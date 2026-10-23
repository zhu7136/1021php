import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
from kinematics import Model,robot_config
from retarget_hud04 import G1_URDF,velocities
import train_hud04


class RepairChecks(unittest.TestCase):
    def setUp(self):
        self.model=Model();self.config=robot_config()

    def test_limits_match_urdf(self):
        np.testing.assert_allclose(self.config['_HU_D04_POS_LOWER'],self.model.lower,atol=1e-9)
        np.testing.assert_allclose(self.config['_HU_D04_POS_UPPER'],self.model.upper,atol=1e-9)
        np.testing.assert_allclose(self.config['_HU_D04_VEL_LIMIT'],self.model.velocity,atol=1e-9)

    def test_default_foot_centers_above_ground(self):
        q=np.array([[self.config['_HU_D04_DEFAULT_JOINT_ANGLES'][n] for n in self.model.names]])
        p,_,_,_=self.model.forward(q,np.array([[0.,0.,.91]]),np.array([[1.,0,0,0]]))
        for foot in ['contact_foot_center_L','contact_foot_center_R']:
            self.assertGreater(p[foot][0,2],.03)

    def test_analytic_position_jacobian(self):
        q=np.array([[self.config['_HU_D04_DEFAULT_JOINT_ANGLES'][n] for n in self.model.names]])
        root=np.array([[0.,0.,1.]]);quat=np.array([[1.,0,0,0]])
        p,r,jp,ja=self.model.forward(q,root,quat)
        for body in ['contact_foot_center_L','contact_foot_center_R','left_wrist_roll_link','right_wrist_roll_link']:
            cols=np.arange(29);jac,_=self.model.jacobian(body,cols,p,jp,ja)
            for i in cols:
                changed=q.copy();changed[0,i]+=1e-6
                pp,_,_,_=self.model.forward(changed,root,quat)
                np.testing.assert_allclose((pp[body]-p[body])/1e-6,jac[:,:,i],atol=4e-7)

    def test_distal_wrist_semantic_mapping(self):
        g1=Model(G1_URDF)
        for side in ['left','right']:
            hparents={j['child']:j for j in self.model.joints};gparents={j['child']:j for j in g1.joints}
            self.assertEqual(hparents[f'{side}_wrist_roll_link']['parent'],f'{side}_wrist_pitch_link')
            self.assertEqual(gparents[f'{side}_wrist_yaw_link']['parent'],f'{side}_wrist_pitch_link')
            self.assertGreater(self.model.chain_length(f'{side}_shoulder_pitch_link',f'{side}_wrist_roll_link'),.5)

    def test_training_requires_separate_candidate(self):
        import argparse
        with self.assertRaisesRegex(ValueError,'Training blocked'):
            train_hud04.require_candidate_dataset(argparse.Namespace(dataset=None,acknowledge_kinematic_candidate=False))
        with self.assertRaisesRegex(ValueError,'recovered G1'):
            train_hud04.require_candidate_dataset(argparse.Namespace(dataset=ROOT/'flux/data/motions-low-step-hu-d04',acknowledge_kinematic_candidate=True))

    def test_world_angular_velocity(self):
        from scipy.spatial.transform import Rotation
        times=np.arange(20)/50
        rot=Rotation.from_rotvec(np.column_stack([np.zeros(20),np.zeros(20),times*.3])).as_matrix()
        np.testing.assert_allclose(velocities(rot,50),np.tile([0,0,.3],(20,1)),atol=1e-10)

    def test_fast_arm_matches_full_fk_and_jacobian(self):
        from continuous_arms import Arm
        rng=np.random.default_rng(27)
        for side in ['left','right']:
            arm=Arm(self.model,side)
            for _ in range(4):
                q=rng.uniform(self.model.lower+.001,self.model.upper-.001,size=(1,29))
                p,r,jp,ja=self.model.forward(q,np.array([[.1,.2,.9]]),np.array([[1.,0,0,0]]))
                point,jac=arm.forward(q[0,arm.cols],p[arm.base][0],r[arm.base][0])
                np.testing.assert_allclose(point,p[arm.body][0],atol=1e-12)
                np.testing.assert_allclose(jac,self.model.jacobian(arm.body,arm.cols,p,jp,ja)[0][0],atol=1e-12)

    def test_sequential_arm_ik_preserves_smooth_endpoint_targets(self):
        from continuous_arms import sequential_arms
        g1=Model(G1_URDF);bodies=self.config['_HU_D04_BODY_NAMES']
        default=np.array([self.config['_HU_D04_DEFAULT_JOINT_ANGLES'][n] for n in self.model.names])
        truth=np.tile(default,(12,1));truth[:,15]+=.02*np.sin(np.linspace(0,1,12));truth[:,22]+=.02*np.cos(np.linspace(0,1,12))
        root=np.tile([0.,0.,.91],(12,1));quat=np.tile([1.,0,0,0],(12,1))
        p,_,_,_=self.model.forward(truth,root,quat)
        bodypos=np.stack([p[b] for b in bodies],axis=1)
        for side in ['left','right']:
            sh=f'{side}_shoulder_pitch_link';hand=f'{side}_wrist_roll_link';source_hand=f'{side}_wrist_yaw_link'
            scale=self.model.chain_length(sh,hand)/g1.chain_length(sh,source_hand)
            bodypos[:,bodies.index(source_hand)]=p[sh]+(p[hand]-p[sh])/scale
        solved,diag=sequential_arms(self.model,g1,{'body_pos_w':bodypos},np.tile(default,(12,1)),root,quat,bodies)
        self.assertTrue(np.all(solved>=self.model.lower))
        self.assertTrue(np.all(solved<=self.model.upper))
        for info in diag.values():
            self.assertLess(info['max_source_grid_hand_error_m'],1e-3)
            self.assertLess(info['max_source_grid_joint_step_rad'],.1)


if __name__=='__main__':
    unittest.main()
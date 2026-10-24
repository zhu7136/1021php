import ast,io,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from control_chain_recorder import Recorder
import diagnose_control_chain as entry

def fixture():
    names=['left_ankle_roll_link','right_ankle_roll_link'];joints=['left_hip_roll_joint','right_hip_roll_joint']
    data=NS(joint_pos_target=torch.tensor([[.3,-.2],[.1,-.1]]),applied_torque=torch.tensor([[1.,2.],[3.,4.]]),computed_torque=torch.ones(2,2))
    term=NS(_applied_position_targets=data.joint_pos_target[:,[1,0]].clone())
    sim=NS(_robot=NS(data=data),dof_ids=[1,0],dof_pos=torch.zeros(2,2),dof_vel=torch.ones(2,2),hard_dof_pos_limits=torch.tensor([[-.15,.15],[-.15,.15]]),contact_sensor=NS(body_names=names,data=NS(net_forces_w=torch.ones(2,2,3))))
    env=NS(device='cpu',robot_config=NS(dof_names=joints,dof_effort_limit_list=[10.,10.]),simulator=sim,action_manager=NS(_term_instances={'joint':term}),reset_buf=torch.tensor([1,0]),episode_length_buf=torch.tensor([6,4]),reward_manager=NS(_episode_sums={'tracking':torch.zeros(2),'penalty':torch.zeros(2)}),rew_buf=torch.zeros(2),calls=0)
    def check():env.calls+=1
    def reward():
        env.calls+=1;env.reward_manager._episode_sums['tracking']+=torch.tensor([.1,.2]);env.reward_manager._episode_sums['penalty']+=torch.tensor([-.2,-.3]);env.rew_buf[:]=-.1
    env._check_termination=check;env._compute_reward=reward
    motion=NS(joint_pos=torch.zeros(2,2),time_steps=torch.tensor([10,20]),motion_cfg=NS(body_names_to_track=names),body_pos_relative_w=torch.ones(2,2,3)*.2,robot_body_pos_w=torch.zeros(2,2,3))
    sink=io.StringIO();r=Recorder(env,motion,sink);r.install()
    return env,sink,r

class Tests(unittest.TestCase):
    def test_pre_reset_snapshot_and_rewards(self):
        env,sink,r=fixture();a=torch.ones(2,2)*.2;r.begin_step('policy_mean',0,torch.ones(2,5),a,a)
        env._check_termination();env._compute_reward();env.simulator.dof_pos.fill_(999)
        row=json.loads(sink.getvalue());self.assertEqual(row['values']['q_after'],[[0,0],[0,0]])
        self.assertEqual(row['values']['done'],[True,False]);self.assertEqual(env.calls,2)
        self.assertTrue(row['values']['target_outside_limits'][0][0]);self.assertEqual(row['values']['applied_torque'][0],[2.,1.])
        self.assertAlmostEqual(row['reward_total'][0],-.1,places=6);r.uninstall();self.assertFalse(r.hooked)
    def test_target_mismatch_fails_closed(self):
        env,sink,r=fixture();a=torch.zeros(2,2);r.begin_step('policy_mean',0,torch.ones(2,5),a,a)
        env.simulator._robot.data.joint_pos_target.fill_(9)
        with self.assertRaisesRegex(AssertionError,'Native target'):env._check_termination()
        self.assertEqual(sink.getvalue(),'')
    def test_missing_torque_remains_missing(self):
        env,sink,r=fixture();del env.simulator._robot.data.applied_torque
        a=torch.zeros(2,2);r.begin_step('policy_mean',0,torch.ones(2,5),a,a);env._check_termination();env._compute_reward()
        row=json.loads(sink.getvalue());self.assertIsNone(row['values']['applied_torque']);self.assertNotIn('configured_effort_limit_fraction',row['values'])
    def test_inactive_hooks_do_not_emit(self):
        env,sink,r=fixture();env._check_termination();env._compute_reward()
        self.assertEqual(sink.getvalue(),'');self.assertEqual(env.calls,2);r.uninstall()
    def test_episode_ids_increment_and_restart_per_mode(self):
        env,sink,r=fixture();a=torch.zeros(2,2)
        for mode in ['policy_mean','policy_mean','reference_pd']:
            r.begin_step(mode,0,torch.ones(2,5),a,a);env._check_termination();env._compute_reward()
        rows=[json.loads(x) for x in sink.getvalue().splitlines()]
        self.assertEqual([x['values']['episode_ids'] for x in rows],[[0,0],[1,0],[0,0]])
    def test_approved_bounds(self):
        self.assertEqual((entry.NUM_ENVS,entry.STEPS_PER_MODE,entry.WORKER_SECONDS),(32,64,600))
        self.assertEqual(entry.MODES,('policy_mean','reference_pd'))
    def test_no_training_or_save_calls(self):
        for name in ['control_chain_recorder.py','diagnose_control_chain.py']:
            tree=ast.parse((Path(__file__).resolve().parents[1]/name).read_text())
            attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            for forbidden in ['learn','backward','zero_grad','step_optimizer','save_checkpoint']:self.assertNotIn(forbidden,attrs)
    def test_optional_telemetry_preserves_old_default(self):
        self.assertEqual(entry.base.MODES,('policy_mean','policy_sample','reference_pd'))
        self.assertEqual(entry.base.WORKER_SECONDS,900)
        text=(Path(__file__).resolve().parents[1]/'diagnose_short_episodes.py').read_text()
        self.assertIn('recorder_factory=None',text);self.assertIn('step<20',text)
    def test_bootstrap_does_not_import_runtime_dependencies(self):
        # A fresh interpreter has no site packages: configure must stay stdlib-only
        # until worker() verifies/installs the pinned offline dependency directory.
        flux=Path(__file__).resolve().parents[1]
        code=f'import sys;sys.path.insert(0,{str(flux)!r});import diagnose_control_chain as e;e.configure();assert "torch" not in sys.modules;assert "typing_extensions" not in sys.modules;assert "control_chain_recorder" not in sys.modules'
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr[-2000:])

if __name__=='__main__':unittest.main(verbosity=2)
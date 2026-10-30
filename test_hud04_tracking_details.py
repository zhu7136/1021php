import unittest
from types import SimpleNamespace as NS
import torch
from hud04_tracking_details import TrackingProxy,tracking_inputs

class NativeLikeTerm:
    def __init__(self):
        self.calls={};self.body_names_to_track=['left_ankle_roll_link']
        self.bad_motion_body_pos_body_indexes=torch.tensor([0])
        self.bad_ref_pos_threshold=.5;self.bad_ref_ori_threshold=.8;self.bad_motion_body_pos_threshold=.5
        self.answer=torch.tensor([True]);self.reset_calls=0
    def bad_ref_pos(self,c):
        self.calls['pos']=self.calls.get('pos',0)+1;return torch.tensor([False])
    def bad_ref_ori(self,c):
        self.calls['ori']=self.calls.get('ori',0)+1;return torch.tensor([False])
    def bad_motion_body_pos(self,c):
        self.calls['body']=self.calls.get('body',0)+1;return torch.tensor([True])
    def __call__(self,env):
        c=env.command_manager.get_state('motion_command')
        self.bad_ref_pos(c);self.bad_ref_ori(c);self.bad_motion_body_pos(c)
        return self.answer
    def reset(self):self.reset_calls+=1

class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.c=NS(time_steps=torch.tensor([0]),ref_pos_w=torch.tensor([[1.,2.,3.]]),
            robot_ref_pos_w=torch.tensor([[1.,2.,3.]]),ref_quat_w=torch.tensor([[0.,0.,0.,1.]]),
            robot_ref_quat_w=torch.tensor([[0.,0.,0.,1.]]),body_pos_relative_w=torch.zeros(1,1,3),
            robot_body_pos_w=torch.tensor([[[1.,0.,0.]]]))
        self.env=NS(command_manager=NS(get_state=lambda name:self.c));self.t=NativeLikeTerm()
        self.v=[];self.d=[];self.proxy=TrackingProxy(self.t,self.v.append,self.d.append)
    def test_each_native_subcheck_once(self):
        self.proxy(self.env);self.assertEqual(self.t.calls,{'pos':1,'ori':1,'body':1})
    def test_exact_native_return_identity(self):self.assertIs(self.proxy(self.env),self.t.answer)
    def test_methods_restored(self):
        before=dict(self.t.__dict__);self.proxy(self.env)
        for n in ['bad_ref_pos','bad_ref_ori','bad_motion_body_pos']:self.assertNotIn(n,self.t.__dict__)
        self.assertEqual(self.t.bad_motion_body_pos_threshold,before['bad_motion_body_pos_threshold'])
    def test_components_and_actual_inputs_recorded(self):
        self.proxy(self.env);d=self.d[0]
        self.assertEqual(d['native_component_results'],{'bad_ref_pos':[False],'bad_ref_ori':[False],'bad_motion_body_pos':[True]})
        self.assertEqual(d['body_error_l2_m'],[[1.]])
        self.assertEqual(d['root_position_error_l2_m'],[0.])
    def test_input_tensors_not_mutated(self):
        before=self.c.body_pos_relative_w.clone();self.proxy(self.env)
        self.assertTrue(torch.equal(self.c.body_pos_relative_w,before))
    def test_native_exception_propagates_and_methods_restore(self):
        def fail(c):raise ArithmeticError('native')
        self.t.bad_ref_ori=fail
        with self.assertRaises(ArithmeticError):self.proxy(self.env)
        self.assertIs(self.t.bad_ref_ori,fail);self.assertNotIn('bad_ref_pos',self.t.__dict__)
    def test_observation_failure_does_not_change_native_result(self):
        del self.c.robot_body_pos_w
        self.assertIs(self.proxy(self.env),self.t.answer)
        self.assertEqual(self.d[0]['observation_error_type'],'AttributeError')
    def test_stateful_reset_forwarded(self):
        self.proxy.reset();self.assertEqual(self.t.reset_calls,1)

if __name__=='__main__':unittest.main()
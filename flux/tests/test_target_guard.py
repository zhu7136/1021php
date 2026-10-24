import ast,dataclasses,importlib.util,sys,tempfile,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'flux')]
from wbt_training.utils import target_guard as guard
from environment_tensor_boundary import install_environment_tensor_boundary

def native_action_class():
    path=Path('/workspace/task047/fresh-run/native-readonly/src/holosoma/holosoma/managers/action/terms/joint_control.py')
    if not path.exists():
        path=ROOT/'thirdparty/holosoma/src/holosoma/holosoma/managers/action/terms/joint_control.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='JointPositionTargetActionTerm')
    class Base:
        def __init__(self,cfg,env):self.cfg,self.env=cfg,env
        def reset(self,env_ids=None):
            if env_ids is None:self._raw_actions.zero_();self._processed_actions.zero_()
            else:self._raw_actions[env_ids]=0;self._processed_actions[env_ids]=0
    ns=dict(torch=torch,ActionTermBase=Base)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag),ns)
    module=types.ModuleType('holosoma.managers.action.terms.joint_control')
    module.JointPositionTargetActionTerm=ns[cls.name]
    with patch.dict(sys.modules,{'holosoma.managers.action.terms.joint_control':module}):
        spec=importlib.util.spec_from_file_location('guarded_action_test',ROOT/'wbt_training/utils/guarded_action.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.GuardedJointPositionTargetActionTerm

class TargetGuardTests(unittest.TestCase):
    def term(self, late_init=False):
        env=NS(num_envs=2,num_dof=3,device='cpu',log_dict={},default_dof_pos=torch.zeros(2,3),
            robot_config=NS(control=NS(control_mode='implicit_position_target',action_scale=.25,
                action_scales_by_effort_limit_over_p_gain=False,clip_actions=True,action_clip_value=100.)),
            simulator=NS(hard_dof_pos_limits=torch.tensor([[-.8727,2.094],[-1.,1.],[-2.,2.]])))
        env.simulator.apply_position_targets_at_dof=lambda t:setattr(env.simulator,'native_target',t.clone())
        if late_init:
            del env.default_dof_pos
            del env.log_dict
        term=native_action_class()(NS(),env)
        env.action_manager=NS(_term_instances={'joint':term},action=torch.zeros(2,3),prev_action=torch.zeros(2,3))
        return term,env
    def test_constructor_before_native_wbt_default_pose_and_log_dict(self):
        # BaseTask constructs ActionManager before _init_buffers/log_dict;
        # WholeBodyTrackingManager creates default_dof_pos after super returns.
        t,e=self.term(late_init=True)
        self.assertFalse(hasattr(e,'default_dof_pos'))
        self.assertFalse(hasattr(e,'log_dict'))
        e.log_dict={};e.default_dof_pos=torch.zeros(2,3)
        t.process_actions(torch.ones(2,3));t.apply_actions()
        self.assertTrue(torch.allclose(e.simulator.native_target,torch.full((2,3),.25)))
    def test_missing_late_pose_fails_before_processing_or_application(self):
        t,e=self.term(late_init=True);e.log_dict={}
        with self.assertRaises(RuntimeError):t.process_actions(torch.ones(2,3))
        self.assertTrue(torch.equal(t._raw_actions,torch.zeros(2,3)))
        self.assertFalse(hasattr(e.simulator,'native_target'))
    def test_late_invalid_default_pose_fails_before_native_action(self):
        t,e=self.term(late_init=True);e.log_dict={};e.default_dof_pos=torch.full((2,3),100.)
        with self.assertRaises(ValueError):t.process_actions(torch.ones(2,3))
        self.assertTrue(torch.equal(t._raw_actions,torch.zeros(2,3)))
        self.assertFalse(hasattr(e.simulator,'native_target'))
    def test_late_nonfinite_default_pose_fails_closed(self):
        t,e=self.term(late_init=True);e.log_dict={};e.default_dof_pos=torch.full((2,3),float('nan'))
        with self.assertRaises(FloatingPointError):t.process_actions(torch.ones(2,3))
        self.assertFalse(hasattr(e.simulator,'native_target'))
    def test_no_application_before_first_pose_validation(self):
        t,e=self.term(late_init=True)
        with self.assertRaises(RuntimeError):t.apply_actions()
        self.assertFalse(hasattr(e.simulator,'native_target'))
    def test_reset_before_late_pose_initialization_is_allowed(self):
        t,e=self.term(late_init=True);t.reset()
        e.log_dict={};e.default_dof_pos=torch.zeros(2,3)
        t.process_actions(torch.zeros(2,3));t.apply_actions()
        self.assertTrue(torch.equal(e.simulator.native_target,torch.zeros(2,3)))
    def test_inside_identity_and_input_preservation(self):
        x=torch.tensor([[.1,-.2]]);old=x.clone();y=guard.protect_targets(x,torch.tensor([[-1.,1.],[-1.,1.]]))
        self.assertTrue(torch.equal(x,old));self.assertTrue(torch.equal(y,x))
    def test_uses_asymmetric_per_joint_limits(self):
        x=torch.tensor([[-8.,8.]])
        y=guard.protect_targets(x,torch.tensor([[-.8727,2.094],[-.3,.4]]))
        self.assertAlmostEqual(float(y[0,0]),-.8527,places=5);self.assertAlmostEqual(float(y[0,1]),.38,places=5)
    def test_invalid_limits_or_margin_fail_closed(self):
        for limits,margin in [(torch.tensor([[0.,0.]]),.02),(torch.tensor([[-1.,1.]]),-1.),(torch.tensor([[0.,.01]]),.02)]:
            with self.assertRaises(ValueError):guard.protect_targets(torch.zeros(1,1),limits,margin)
    def test_nonfinite_fail_closed(self):
        for v in [float('nan'),float('inf')]:
            with self.assertRaises(FloatingPointError):guard.protect_targets(torch.tensor([[v]]),torch.tensor([[-1.,1.]]))
    def test_native_action_scale_and_raw_action_preserved(self):
        t,e=self.term();a=torch.full((2,3),.8);t.process_actions(a);t.apply_actions()
        self.assertTrue(torch.equal(t._raw_actions,a));self.assertTrue(torch.equal(a,torch.full((2,3),.8)))
        self.assertTrue(torch.allclose(e.simulator.native_target,torch.full((2,3),.2)))
    def test_extreme_raw_requests_never_reach_native_target(self):
        t,e=self.term();a=torch.tensor([[-1000.,1000.,1000.],[1000.,-1000.,-1000.]])
        t.process_actions(a);t.apply_actions();lim=e.simulator.hard_dof_pos_limits
        self.assertTrue((e.simulator.native_target>=lim[:,0]+.02).all())
        self.assertTrue((e.simulator.native_target<=lim[:,1]-.02).all())
        self.assertTrue(torch.equal(t._raw_actions,a));self.assertTrue((guard.requested_target_violation(e)>0).all())
    def test_constant_outside_request_is_penalized(self):
        t,e=self.term();a=torch.full((2,3),-10.16);t.process_actions(a)
        e.action_manager.action=a;e.action_manager.prev_action=a
        self.assertTrue(torch.equal(guard.physical_action_rate(e),torch.zeros(2)))
        self.assertTrue((guard.requested_target_violation(e)>0).all())
    def test_safe_request_has_zero_violation(self):
        t,e=self.term();t.process_actions(torch.zeros(2,3))
        self.assertTrue(torch.equal(guard.requested_target_violation(e),torch.zeros(2)))
    def test_rate_penalty_uses_physical_units(self):
        t,e=self.term();e.action_manager.action=torch.ones(2,3)
        self.assertTrue(torch.allclose(guard.physical_action_rate(e),torch.full((2,),3*.25**2)))
    def test_terminal_charge_is_dt_independent_and_ignores_timeouts(self):
        e=NS(dt=.02,termination_manager=NS(terminated=torch.tensor([True,False]),time_outs=torch.tensor([False,True])))
        self.assertTrue(torch.equal(guard.failure_event(e)*e.dt,torch.tensor([1.,0.])))
        e.dt=.01;self.assertTrue(torch.equal(guard.failure_event(e)*e.dt,torch.tensor([1.,0.])))
    def test_guard_detects_mutation_before_application(self):
        t,e=self.term();t.process_actions(torch.zeros(2,3));t._processed_actions[0,0]=-4
        with self.assertRaises(RuntimeError):t.apply_actions()
    def test_partial_reset_clears_requested_cache(self):
        t,e=self.term();t.process_actions(torch.ones(2,3));t.reset(torch.tensor([0]))
        self.assertEqual(float(t._requested_position_targets[0].sum()),0.)
        self.assertGreater(float(t._requested_position_targets[1].sum()),0.)
    def test_protected_entrypoint_does_not_import_torch_for_bootstrap(self):
        import subprocess
        code=f'import sys;sys.path.insert(0,{str(ROOT/"flux")!r});import train_protected_pilot as p;assert p.fresh.PROTECTED;assert "torch" not in sys.modules'
        p=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr)

class ConfigTests(unittest.TestCase):
    def test_configuration_is_opt_in_and_retains_ppo_and_tracking_rewards(self):
        @dataclasses.dataclass(frozen=True)
        class Term:
            func:str='original'
            params:dict=dataclasses.field(default_factory=dict)
            weight:float=1.
            scale:float=1.
            clip:object=None
        @dataclasses.dataclass(frozen=True)
        class Manager:
            terms:dict
        @dataclasses.dataclass(frozen=True)
        class Control:
            action_scale:float=1.
            action_scales_by_effort_limit_over_p_gain:bool=False
        @dataclasses.dataclass(frozen=True)
        class Robot:
            control:Control
        @dataclasses.dataclass(frozen=True)
        class Config:
            robot:Robot
            action:Manager
            reward:Manager
            termination:Manager
            algo:object
        modules={}
        for name,attrs in [('holosoma.config_types.action',dict(ActionTermCfg=Term,ActionManagerCfg=Manager)),
                           ('holosoma.config_types.reward',dict(RewardTermCfg=Term))]:
            m=types.ModuleType(name)
            for k,v in attrs.items():setattr(m,k,v)
            modules[name]=m
        reward=Manager({'action_rate_l2':Term(weight=-.1),'tracking':Term(weight=1.),'limits_dof_pos':Term(weight=-10.)})
        original=Config(Robot(Control()),Manager({'joint_control':Term()}),reward,
            Manager({'bad_tracking':Term(),'timeout':Term()}),object())
        with patch.dict(sys.modules,modules):
            out=guard.protected_config(original)
            with self.assertRaises(ValueError):guard.protected_config(out)
        self.assertEqual(original.robot.control.action_scale,1.)
        self.assertEqual(out.robot.control.action_scale,.25)
        self.assertIs(out.algo,original.algo)
        self.assertIs(out.reward.terms['tracking'],original.reward.terms['tracking'])
        self.assertIs(out.reward.terms['limits_dof_pos'],original.reward.terms['limits_dof_pos'])
        self.assertEqual(out.reward.terms['action_rate_l2'].weight,-.1)
        self.assertEqual(out.reward.terms['failure_event'].weight,-1.)
        self.assertEqual(out.reward.terms['requested_target_violation'].weight,-1.)

class TensorBoundaryTests(unittest.TestCase):
    def env(self):
        class Env:
            def step(self,state):self.buffer=state['actions'].clone();return self.buffer
            def reset_all(self):self.buffer.zero_();return self.buffer
        return Env()
    def test_reproduces_previous_inference_buffer_failure(self):
        e=self.env()
        with torch.inference_mode():e.step({'actions':torch.ones(2)})
        with self.assertRaises(RuntimeError):e.reset_all()
    def test_rollout_eval_reset_and_backward_after_boundary(self):
        e=self.env();install_environment_tensor_boundary(e)
        p=torch.nn.Parameter(torch.ones(2))
        with torch.inference_mode():
            a=p*2;out=e.step({'actions':a})
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference(out));self.assertFalse(torch.is_inference(e.buffer))
        with torch.no_grad():e.reset_all()
        (p*out.detach()).sum().backward();self.assertIsNotNone(p.grad)
        self.assertTrue(torch.equal(a,torch.full((2,),2.)))
    def test_duplicate_install_rejected(self):
        e=self.env();install_environment_tensor_boundary(e)
        with self.assertRaises(ValueError):install_environment_tensor_boundary(e)

if __name__=='__main__':unittest.main(verbosity=2)
"""CPU accounting tests, including pinned native clip-reset branch, not GPU validation."""
import ast
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
from continuous_segments import SegmentLedger,ContinuousSegmentCapture


def native_step_prefix():
    relative='src/holosoma/holosoma/managers/command/terms/wbt.py'
    path=Path('/workspace/task047/prior-diagnostics/native')/relative
    if not path.is_file():path=ROOT/'thirdparty/holosoma'/relative
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!='4acfdb0d14e2877a1503bfb4508c6751db97d064ff270e99bdf74f489fe45d0a':
        raise ValueError('Pinned native command source changed')
    tree=ast.parse(data)
    fn=next(n for c in tree.body if isinstance(c,ast.ClassDef)
        for n in c.body if isinstance(n,ast.FunctionDef) and c.name=='MotionCommand' and n.name=='step')
    # Execute EXACT native prefix through reset+simulator flush. Alignment math
    # following this branch is not exercised or reimplemented by these CPU tests.
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If)
        and 'ended_env_ids.numel()' in ast.unparse(n.test))
    fn.body=fn.body[:end+1]
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['step']


class LedgerTests(unittest.TestCase):
    def test_no_reset_native_metrics_equivalent(self):
        ledger=SegmentLedger(2,.02,3)
        for k in range(1,4):ledger.advance(k,[k>=2,False],[])
        r=ledger.finish()
        self.assertEqual([s['duration_steps'] for s in sorted(r['segments'],key=lambda s:s['env'])],[2,3])
        self.assertEqual(r['first_segment_native_terminations'],1)
        self.assertEqual(r['first_segment_horizon_survivors'],1)

    def test_reset_is_censoring_not_survival(self):
        ledger=SegmentLedger(1,.02,3)
        ledger.advance(1,[False],[dict(env=0,frame_before=10,frame_after=999)])
        ledger.advance(2,[False],[]);ledger.advance(3,[True],[])
        r=ledger.finish()
        self.assertEqual([s['duration_steps'] for s in r['segments']],[1,2])
        self.assertEqual(r['segments'][0]['outcome'],'censored_internal_reset')
        self.assertEqual(r['segments'][1]['outcome'],'native_termination')
        self.assertEqual(r['first_segment_horizon_survivors'],0)

    def test_same_step_native_failure_has_precedence(self):
        ledger=SegmentLedger(1,.02,1)
        ledger.advance(1,[True],[dict(env=0)])
        r=ledger.finish()
        self.assertEqual(r['first_segment_native_terminations'],1)
        self.assertEqual(r['first_segment_internal_reset_censored'],0)
        self.assertTrue(r['internal_reset_events'][0]['native_termination_same_step'])

    def test_final_step_reset_is_not_success_or_zero_segment(self):
        ledger=SegmentLedger(1,.02,1);ledger.advance(1,[False],[dict(env=0)])
        r=ledger.finish()
        self.assertEqual(len(r['segments']),1)
        self.assertEqual(r['first_segment_horizon_survivors'],0)

    def test_repeated_resets_preserve_total_exposure(self):
        ledger=SegmentLedger(1,.02,4)
        for k in range(1,5):ledger.advance(k,[False],[dict(env=0)] if k in (1,3) else [])
        r=ledger.finish()
        self.assertEqual([s['duration_steps'] for s in r['segments']],[1,2,1])
        self.assertEqual(sum(s['duration_steps'] for s in r['segments']),4)

    def test_post_failure_resets_ignored(self):
        ledger=SegmentLedger(1,.02,2);ledger.advance(1,[True],[])
        ledger.advance(2,[False],[dict(env=0)])
        self.assertEqual(ledger.finish()['internal_reset_events'],[])

    def test_incomplete_and_double_finalization_rejected(self):
        ledger=SegmentLedger(1,.02,1)
        with self.assertRaises(ValueError):ledger.finish()
        ledger.advance(1,[False],[]);ledger.finish()
        with self.assertRaises(ValueError):ledger.finish()

    def test_invalid_order_vector_and_duplicate_reset_rejected(self):
        for step,done,events in [(2,[False],[]),(1,[0],[]),(1,[False],[dict(env=0),dict(env=0)]),(1,[False],[dict(env=1)])]:
            with self.subTest(step=step,done=done,events=events):
                with self.assertRaises(ValueError):SegmentLedger(1,.02,2).advance(step,done,events)


class NativeBoundaryTests(unittest.TestCase):
    def make(self):
        env=NS(num_envs=2,dt=.02,_update_tasks_before_termination=False,
            reset_buf=torch.zeros(2,dtype=torch.bool))
        calls=[]
        env.simulator=NS(robot_root_states=torch.zeros(2,13),dof_state=torch.zeros(2,4),
            set_actor_root_state_tensor_robots=lambda *a:calls.append('root'),
            set_dof_state_tensor_robots=lambda *a:calls.append('dof'),
            refresh_sim_tensors=lambda:calls.append('refresh'))
        motion=NS(num_envs=2,device='cpu',_env=env,
            time_steps=torch.tensor([1,4]),motion_ids=torch.tensor([0,0]),
            motion_end_reset=torch.zeros(2,dtype=torch.bool),
            motion_cfg=NS(freeze_at_timestep_zero_prob=0.,resample_on_motion_end=True),
            motion=NS(time_step_total=20,motion_ends=torch.tensor([i in (2,8,19) for i in range(20)]),
                motion_start_idx=torch.tensor([0]),motion_end_idx=torch.tensor([20])))
        def reset(ids):
            motion.time_steps[ids]=torch.randint(10,15,(len(ids),))
            env.simulator.robot_root_states[ids]=torch.rand(len(ids),13)
            calls.append('reset')
        motion.reset=reset
        step=native_step_prefix()
        env._update_tasks_callback=lambda:step(motion)
        return env,motion,calls

    def test_native_branch_explicitly_marks_boundary_after_step_one(self):
        env,m,calls=self.make();c=ContinuousSegmentCapture(env,m,horizon=1)
        c.begin_step(1);env._update_tasks_callback();c.end_step(env.reset_buf)
        r=c.finish();c.close()
        self.assertEqual(r['first_segment_internal_reset_censored'],1)
        event=r['internal_reset_events'][0]
        self.assertEqual((event['env'],event['frame_before'],event['boundary_after_step']),(0,1,1))
        self.assertEqual(calls,['reset','root','dof','refresh'])

    def test_instrumentation_preserves_native_rng_state_and_simulator_writes(self):
        a,ma,ca=self.make();b,mb,cb=self.make()
        torch.manual_seed(101);a._update_tasks_callback();rng=torch.get_rng_state().clone()
        torch.manual_seed(101);c=ContinuousSegmentCapture(b,mb,horizon=1)
        c.begin_step(1);b._update_tasks_callback();c.end_step(b.reset_buf);c.finish();c.close()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(ma.time_steps,mb.time_steps))
        self.assertTrue(torch.equal(a.simulator.robot_root_states,b.simulator.robot_root_states))
        self.assertEqual(ca,cb)

    def test_explicit_flag_catches_reset_even_if_frame_looks_contiguous(self):
        env,m,_=self.make()
        def callback():m.time_steps.add_(1);m.motion_end_reset[0]=True
        env._update_tasks_callback=callback
        c=ContinuousSegmentCapture(env,m,horizon=1)
        c.begin_step(1);env._update_tasks_callback();c.end_step(env.reset_buf)
        self.assertEqual(c.finish()['first_segment_internal_reset_censored'],1);c.close()

    def test_frame_jump_without_native_flag_not_invented_as_reset(self):
        env,m,_=self.make();env._update_tasks_callback=lambda:m.time_steps.add_(100)
        c=ContinuousSegmentCapture(env,m,horizon=1)
        c.begin_step(1);env._update_tasks_callback();c.end_step(env.reset_buf)
        self.assertEqual(c.finish()['internal_reset_events'],[]);c.close()

    def test_missing_flag_and_unsupported_callback_order_fail_closed(self):
        env,m,_=self.make();m.motion_end_reset=None
        with self.assertRaisesRegex(ValueError,'Explicit native'):ContinuousSegmentCapture(env,m)
        env,m,_=self.make();env._update_tasks_before_termination=True
        with self.assertRaisesRegex(ValueError,'post-termination'):ContinuousSegmentCapture(env,m)

    def test_missing_duplicate_and_changed_done_fail_closed(self):
        env,m,_=self.make();c=ContinuousSegmentCapture(env,m,horizon=1);c.begin_step(1)
        with self.assertRaisesRegex(ValueError,'Missing'):c.end_step(env.reset_buf)
        env._update_tasks_callback()
        with self.assertRaisesRegex(ValueError,'count'):env._update_tasks_callback()
        with self.assertRaisesRegex(ValueError,'changed'):c.end_step(torch.ones(2,dtype=torch.bool))
        c.close()

    def test_close_idempotent_restores_callback(self):
        env,m,_=self.make();original=env._update_tasks_callback
        c=ContinuousSegmentCapture(env,m);c.close();c.close()
        self.assertIs(env._update_tasks_callback,original)

    def test_json_serializable_no_nonfinite_metrics(self):
        env,m,_=self.make();c=ContinuousSegmentCapture(env,m,horizon=1)
        c.begin_step(1);env._update_tasks_callback();c.end_step(env.reset_buf)
        json.dumps(c.finish(),allow_nan=False);c.close()


class EvaluationIntegrationTests(unittest.TestCase):
    def run_evaluation(self,directory,enabled,reset_step=None,throw=False):
        import low_noise_tools
        import test_paired_evaluation as previous
        from test_root_state_proxy_contract import native_proxy_class,MotionConfig
        capture,env,motion,stats=previous.CaptureTests().make(Path(directory)/'steps.jsonl')
        env.simulator.robot_root_states=native_proxy_class()(torch.zeros(32,13))
        env.is_evaluating=False;env.num_envs=32;env.device='cpu'
        env._update_tasks_before_termination=False
        motion.motion_cfg=MotionConfig();motion.motion_end_reset=torch.zeros(32,dtype=torch.bool)
        original_cfg=motion.motion_cfg
        stats.clear=lambda:None
        stats.summary=lambda:dict(environment_steps=capture.samples,mean_reward=0.,reward_terms={},termination_flags={})
        def callback():
            if throw and stats.active:raise RuntimeError('synthetic callback failure')
            motion.motion_end_reset.zero_();motion.time_steps.add_(1)
            if stats.active and capture.steps==reset_step:
                motion.motion_end_reset[0]=True;motion.time_steps[0]=999
            torch.rand(1) # account for unchanged RNG consumption
        env._update_tasks_callback=callback
        def step(actor_state):
            if stats.active and capture.steps>=9:motion.body_pos_relative_w[:,0,2]=.6
            env._check_termination();env._update_tasks_callback()
            return dict(actor_obs=torch.zeros(32,4)),torch.zeros(32),env.reset_buf.clone(),{}
        env.step=step
        algo=NS(env=env,device='cpu',actor_obs_keys=['actor_obs'],_eval_mode=lambda:None,
            _normalize_actor_obs=lambda raw,update=False:raw,
            actor=NS(act_inference=lambda obs:torch.zeros(32,29)))
        try:
            result=low_noise_tools.evaluate(algo,motion,stats,seed=1402,continuous_segments=enabled)
            diagnostics=capture.finish(result)
            self.assertEqual(len(diagnostics['first_failures']),32)
            self.assertEqual(capture.steps,128)
            return result,torch.get_rng_state().clone()
        finally:
            capture.sink.close()
            self.assertIs(env._update_tasks_callback,callback)
            self.assertIs(motion.motion_cfg,original_cfg)
            self.assertFalse(stats.active);self.assertIsNone(stats.mask)
            self.assertFalse(env.is_evaluating)

    def test_128_steps_real_root_proxy_and_legacy_metrics_unchanged(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as a,tempfile.TemporaryDirectory(dir='/workspace') as b:
            old,rng_old=self.run_evaluation(a,False)
            new,rng_new=self.run_evaluation(b,True)
            segments=new.pop('continuous_segments')
            self.assertEqual(new,old);self.assertTrue(torch.equal(rng_old,rng_new))
            self.assertEqual(segments['first_segment_native_terminations'],32)
            self.assertEqual(segments['internal_reset_events'],[])

    def test_128_steps_reset_does_not_extend_first_segment(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            result,_=self.run_evaluation(d,True,reset_step=3)
            self.assertEqual(result['first_episode_steps'],[10]*32)
            segments=result['continuous_segments']
            self.assertEqual(segments['first_segment_internal_reset_censored'],1)
            e0=[r for r in segments['segments'] if r['env']==0]
            self.assertEqual([r['duration_steps'] for r in e0],[3,7])
            self.assertEqual(len(segments['internal_reset_events']),1)

    def test_exception_restores_all_evaluation_hooks_and_configuration(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            with self.assertRaisesRegex(RuntimeError,'synthetic callback failure'):
                self.run_evaluation(d,True,throw=True)


if __name__=='__main__':unittest.main()
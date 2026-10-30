"""CPU-only opt-in integration and non-invasive numeric logging tests."""
from dataclasses import dataclass, replace
import hashlib
import ast
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

from hud04_depth_diagnostics import (NumericDepthEventsMixin, configure_depth_diagnostics,
                                     depth_diagnostic_status, options)
from test_hud04_depth_reset_guard import FakeTask, NativeDepth, image
from hud04_diagnostic_logging import emit_depth_reset_diagnostics

@dataclass(frozen=True)
class Term:
    func: str
    params: dict
    noise: float=0.
    scale: float=1.
    clip: object=None
@dataclass(frozen=True)
class Group:
    terms: dict
    enable_noise: bool=False
    concatenate: bool=False
    history_length: int=1
@dataclass(frozen=True)
class Observation:
    groups: dict
@dataclass(frozen=True)
class Training:
    num_envs: int=1
    seed: int=0
@dataclass(frozen=True)
class Config:
    observation: Observation
    training: Training
    command: object=None
    termination: object=None
    algo: object=None
    logger: object=None

@dataclass(frozen=True)
class Command:
    setup_terms: dict
    reset_terms: dict
    step_terms: dict
@dataclass(frozen=True)
class Algo:
    _target_: str='original'
@dataclass(frozen=True)
class Video:
    enabled: bool=True
@dataclass(frozen=True)
class Logger:
    headless_recording: bool
    video: Video

def config(n=1):
    depth=Term('holosoma.managers.observation.terms.depth:WarpDepthImageObsTerm',
               {'latency_frame':[3,4],'buffer_len':6,'resize':[58,87],'enable_holes':True})
    return Config(Observation({'depth_camera':Group({'depth_cam':depth}),
                               'actor_obs':Group({'actor':Term('actor',{})},enable_noise=True)}),
                  Training(n),object(),object(),object())

def request(fix=False,limit=128):
    return {'evaluation_role':'student','scope':{'training_updates':0,'automatic_retries':0,'max_policy_steps':1000},
            'depth_reset_diagnostics':{'fix_pending_reset':fix,'max_events':limit}}

class CapturedDepth(NumericDepthEventsMixin,NativeDepth):
    def __init__(self,cfg,env):
        self.records=[]
        super().__init__(cfg,env)
    def _write_depth_event(self,record):
        self.records.append(record)

def instrument(fix=False,limit=128,latency_range=None):
    env=FakeTask(guarded=False,latency=4 if latency_range is None else None,latency_range=latency_range)
    cfg=Term('test',{'hud04_depth_diagnostics':{'fix_pending_reset':fix,'max_events':limit}})
    term=CapturedDepth(cfg,env)
    env.observation_manager._term_instances['depth_camera']['depth_cam']=term
    env.depth=term
    return env

class ConfigTests(unittest.TestCase):
    def test_disabled_returns_identical_config_without_scope(self):
        original=config();self.assertIs(configure_depth_diagnostics(original,{}),original)
    def test_enabled_changes_only_depth_term_binding_and_options(self):
        original=config();updated=configure_depth_diagnostics(original,request(True))
        self.assertIs(updated.training,original.training);self.assertIs(updated.command,original.command)
        self.assertIs(updated.termination,original.termination);self.assertIs(updated.algo,original.algo)
        self.assertIs(updated.observation.groups['actor_obs'],original.observation.groups['actor_obs'])
        old=original.observation.groups['depth_camera'].terms['depth_cam']
        new=updated.observation.groups['depth_camera'].terms['depth_cam']
        self.assertEqual(new.func,'hud04_depth_runtime:HuD04DiagnosticDepth')
        self.assertEqual({k:v for k,v in new.params.items() if k!='hud04_depth_diagnostics'},old.params)
        self.assertNotIn('hud04_depth_diagnostics',old.params)
        self.assertEqual(new.noise,old.noise);self.assertEqual(new.scale,old.scale)
    def test_trace_only_mode_does_not_imply_fix(self):
        updated=configure_depth_diagnostics(config(),request(False))
        self.assertIs(updated.observation.groups['depth_camera'].terms['depth_cam'].params['hud04_depth_diagnostics']['fix_pending_reset'],False)
    def test_teacher_control_scope_allowed(self):
        req=request(True);req['evaluation_role']='teacher'
        self.assertIsNotNone(configure_depth_diagnostics(config(),req))
    def test_invalid_options_fail_closed(self):
        values=[None,False,{}, {'fix_pending_reset':True,'max_events':31},
          {'fix_pending_reset':True,'max_events':513}, {'fix_pending_reset':'true','max_events':128},
          {'fix_pending_reset':True,'max_events':128.,},
          {'fix_pending_reset':True,'max_events':True},
          {'fix_pending_reset':True,'max_events':128,'url':'secret'}]
        for value in values:
            with self.subTest(value=value),self.assertRaises(ValueError):options(value)
    def test_training_retry_or_multienv_rejected(self):
        for field,value in [('training_updates',1),('automatic_retries',1),('training_updates',False),
                            ('max_policy_steps',1001),('max_policy_steps',True),('max_policy_steps',0)]:
            req=request();req['scope'][field]=value
            with self.assertRaises(PermissionError):configure_depth_diagnostics(config(),req)
        with self.assertRaises(PermissionError):configure_depth_diagnostics(config(10),request())
        with self.assertRaises(PermissionError):configure_depth_diagnostics(config(True),request())
    def test_unknown_native_term_refused(self):
        cfg=config();cfg.observation.groups['depth_camera'].terms['depth_cam']=Term('other',{})
        with self.assertRaises(ValueError):configure_depth_diagnostics(cfg,request())
    def test_original_runtime_unimported_during_config(self):
        with patch.dict(sys.modules,{'hud04_depth_runtime':None}):
            self.assertIsNotNone(configure_depth_diagnostics(config(),request()))

class TraceTests(unittest.TestCase):
    def test_trace_only_preserves_original_failure_and_records_erased_flag(self):
        env=instrument(False);self.assertTrue(image(env.reset_all()).eq(0).all())
        resets=[r for r in env.depth.records if r['kind']=='reset_before']
        self.assertEqual([r['reset_count'] for r in resets],[1,0])
        self.assertEqual([r['pending_reset'] for r in resets],[False,True])
        self.assertFalse(next(r for r in env.depth.records if r['kind']=='capture_before')['pending_reset'])
    def test_guard_trace_records_preserved_flag_then_consumption(self):
        env=instrument(True);self.assertTrue(image(env.reset_all()).ne(0).all())
        self.assertTrue(next(r for r in env.depth.records if r['kind']=='capture_before')['pending_reset'])
        last=[r for r in env.depth.records if r['kind']=='observation_after'][-1]
        self.assertFalse(last['pending_reset']);self.assertFalse(last['output_all_zero'])
        self.assertEqual(last['completed_captures'],1)
    def test_nonmutating_reads_do_not_capture_or_consume_flag(self):
        env=instrument(True);env.depth.reset();env.observation_manager.get_obs_dims()
        status=depth_diagnostic_status(env)
        self.assertEqual(status['capture_calls'],0);self.assertTrue(env.depth.reset_episodes.all())
        self.assertFalse(env.depth.records[-1]['history_write'])
    def test_logging_does_not_change_rng_or_buffer_or_output(self):
        def run(traced):
            with torch.random.fork_rng():
                torch.manual_seed(47)
                env=instrument(False,latency_range=[3,4]) if traced else FakeTask(guarded=False,latency=None,latency_range=[3,4])
                outputs=[image(env.reset_all()).clone()]
                for _ in range(12):outputs.append(image(env.step({'actions':torch.zeros(1,29)})[0]).clone())
                return outputs,env.depth.depth_buffer.clone(),torch.get_rng_state().clone(),env.command.time_steps.clone()
        baseline=run(False);traced=run(True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline[0],traced[0])))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline[1:],traced[1:])))
    def test_event_cap_does_not_stop_captures(self):
        env=instrument(True,limit=32);env.reset_all()
        for _ in range(20):env.step({'actions':torch.zeros(1,29)})
        self.assertEqual(len(env.depth.records),32)
        self.assertEqual(env.depth.records[-1]['kind'],'event_limit')
        status=depth_diagnostic_status(env);self.assertTrue(status['truncated'])
        self.assertFalse(status['logging_failed']);self.assertEqual(status['completed_captures'],21)
    def test_log_write_failure_does_not_hide_successful_observation(self):
        env=instrument(True)
        def fail(_):raise OSError('private_signed_url_should_not_be_printed')
        env.depth._write_depth_event=fail
        self.assertTrue(image(env.reset_all()).ne(0).all())
        status=depth_diagnostic_status(env);self.assertTrue(status['logging_failed'])
        self.assertEqual(status['logging_failures'],1)
        self.assertNotIn('private_signed_url',json.dumps(env.depth.records))
    def test_native_capture_exception_propagates_identically(self):
        env=instrument(True);failure=RuntimeError('private_capture_error')
        with patch.object(NativeDepth,'_get_depth_images',side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:env.reset_all()
        self.assertIs(caught.exception,failure)
        self.assertEqual(depth_diagnostic_status(env)['completed_captures'],0)
        self.assertIn('native_capture_failed',[r['kind'] for r in env.depth.records])
        self.assertNotIn('private_capture_error',json.dumps(env.depth.records))
    def test_records_are_bounded_numeric_fixed_schema(self):
        env=instrument(True);env.reset_all()
        text=json.dumps(env.depth.records,allow_nan=False)
        self.assertNotIn('/workspace',text);self.assertNotIn('http',text)
        self.assertLess(max(len(json.dumps(r)) for r in env.depth.records),1024)
        self.assertEqual([r['sequence'] for r in env.depth.records],list(range(1,len(env.depth.records)+1)))
        times=[r['host_elapsed_ns'] for r in env.depth.records]
        self.assertEqual(times,sorted(times))
    def test_status_disabled_without_wrapper(self):
        self.assertEqual(depth_diagnostic_status(FakeTask(guarded=False)),{'enabled':False})
    def test_multienvironment_wrapper_rejected(self):
        env=FakeTask(guarded=False,n=2)
        with self.assertRaises(PermissionError):CapturedDepth(Term('test',request()['depth_reset_diagnostics']),env)
    def test_runtime_source_mismatch_rejected_before_type_binding(self):
        fake=types.ModuleType('holosoma.managers.observation.terms.depth');fake.WarpDepthImageObsTerm=NativeDepth
        path=Path(__file__).with_name('hud04_depth_runtime.py')
        with patch.dict(sys.modules,{'holosoma.managers.observation.terms.depth':fake}):
            spec=importlib.util.spec_from_file_location('isolated_depth_runtime',path)
            module=importlib.util.module_from_spec(spec)
            with self.assertRaisesRegex(ValueError,'reviewed version'):spec.loader.exec_module(module)

class ChildIntegrationTests(unittest.TestCase):
    def child(self,cfg):
        lifecycle=types.ModuleType('offline_video_child')
        obstacle=types.ModuleType('obstacle_video_child');obstacle.configuration=lambda req:cfg
        with patch.dict(sys.modules,{'offline_video_child':lifecycle,'obstacle_video_child':obstacle}):
            spec=importlib.util.spec_from_file_location('isolated_eval_child',Path(__file__).with_name('hud04_eval_child.py'))
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod
    def run_configuration(self,enabled,bad_hash=False):
        command=Command({'motion_command':Term('old',{})},{'motion_command':Term('old',{})},{'motion_command':Term('old',{})})
        cfg=replace(config(),command=command,algo=Algo(),logger=Logger(True,Video()))
        req=request(True);req['evaluation_seed']=1
        if not enabled:del req['depth_reset_diagnostics']
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path=Path(temporary)/'synthetic-checkpoint';path.write_bytes(b'CPU-only synthetic fixture')
            req['checkpoint']=str(path)
            req['source_checkpoint_sha256']='0'*64 if bad_hash else hashlib.sha256(path.read_bytes()).hexdigest()
            return cfg,self.child(cfg).configuration(req)
    def test_actual_child_default_keeps_original_depth_and_noise_groups(self):
        cfg,updated=self.run_configuration(False)
        self.assertIs(updated.observation,cfg.observation)
        self.assertEqual(updated.algo._target_,'hud04_single_episode_algo.HuD04SingleEpisode')
        self.assertFalse(updated.logger.video.enabled)
        self.assertEqual(updated.training.seed,1)
    def test_actual_child_explicit_option_selects_wrapper_only_in_new_config(self):
        cfg,updated=self.run_configuration(True)
        self.assertEqual(updated.observation.groups['depth_camera'].terms['depth_cam'].func,
                         'hud04_depth_runtime:HuD04DiagnosticDepth')
        self.assertIs(updated.observation.groups['actor_obs'],cfg.observation.groups['actor_obs'])
        self.assertNotIn('hud04_depth_diagnostics',cfg.observation.groups['depth_camera'].terms['depth_cam'].params)
    def test_actual_child_rejects_wrong_checkpoint_before_diagnostic_binding(self):
        with self.assertRaisesRegex(ValueError,'Checkpoint hash mismatch'):self.run_configuration(True,bad_hash=True)

class WorkerTransportTests(unittest.TestCase):
    def test_actual_worker_option_forwarding_is_explicit(self):
        tree=ast.parse(Path(__file__).with_name('obstacle_video_worker.py').read_text())
        candidates=[node for node in ast.walk(tree) if isinstance(node,ast.If)
                    and ast.unparse(node.test)=="'depth_reset_diagnostics' in c"]
        self.assertEqual(len(candidates),1)
        code=compile(ast.Module(body=candidates,type_ignores=[]),'worker-depth-forwarding','exec')
        disabled={'c':{},'req':{}};exec(code,disabled);self.assertEqual(disabled['req'],{})
        enabled={'c':request(True),'req':{}};exec(code,enabled)
        self.assertEqual(enabled['req']['depth_reset_diagnostics'],request(True)['depth_reset_diagnostics'])
    def test_native_log_recovery_roundtrip_discards_unrelated_lines(self):
        env=instrument(True);env.reset_all()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path=Path(temporary)/'native.log'
            path.write_text('private unrelated line\n'+'\n'.join(json.dumps(r,separators=(',',':')) for r in env.depth.records))
            output=io.StringIO()
            with contextlib.redirect_stdout(output):emit_depth_reset_diagnostics(temporary)
        self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()],env.depth.records)
        self.assertNotIn('private',output.getvalue())
    def test_native_log_recovery_rejects_arbitrary_extra_text(self):
        env=instrument(True);record=dict(env.depth.records[0],secret='must not escape')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            (Path(temporary)/'native.log').write_text(json.dumps(record,separators=(',',':')))
            output=io.StringIO()
            with contextlib.redirect_stdout(output),self.assertRaisesRegex(ValueError,'schema'):
                emit_depth_reset_diagnostics(temporary)
            self.assertEqual(output.getvalue(),'')

if __name__=='__main__':unittest.main()
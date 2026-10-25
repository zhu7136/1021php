"""CPU-only formal contracts; not simulator, policy-quality or throughput proof."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from flux import formal_contract as contract, formal_input
from flux.test_checkpoint_roundtrip import algorithm

ROOT=Path(__file__).resolve().parent


def formal_fixture(output, *, stop_after=3, update=True, corrupt=False):
    base=algorithm()
    def native_learn(self):
        for index in range(self.current_learning_iteration,self.current_learning_iteration+self.config.num_learning_iterations):
            self.current_learning_iteration=index
            if update:
                for optimizer in (self.actor_optimizer,self.critic_optimizer):
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.grad=torch.ones_like(param)
                    optimizer.step()
                    optimizer.zero_grad()
            if self._flux_completed_updates+1==stop_after:
                (output/'STOP_AFTER_UPDATE').touch()
            self._post_epoch_logging(index,{'loss':0.1})
    base.__mro__[1].learn=native_learn
    teacher=types.ModuleType('flux.teacher_algo')
    teacher.BoundedPPO=base
    teacher._require_same_state=base.learn.__globals__['_require_same_state']
    # resume_algo helpers must use the same teacher base and exact comparator.
    resume_spec=importlib.util.spec_from_file_location('formal_resume_fixture',ROOT/'resume_algo.py')
    resume=importlib.util.module_from_spec(resume_spec)
    with patch.dict(sys.modules,{'flux.teacher_algo':teacher}):
        resume_spec.loader.exec_module(resume)
    spec=importlib.util.spec_from_file_location('formal_algo_fixture',ROOT/'formal_algo.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'flux.teacher_algo':teacher,'flux.resume_algo':resume}):
        spec.loader.exec_module(module)
    algo=module.FormalContinuationPPO(output)
    algo.empirical_normalization=False
    algo.actor_obs_normalizer=torch.nn.Identity()
    algo.critic_obs_normalizer=torch.nn.Identity()
    algo.config.num_learning_iterations=10000
    algo.current_learning_iteration=20
    source=Path(algo.log_dir)/'source.pt'
    algo.save(str(source))
    saved=torch.load(source,weights_only=True,map_location='cpu')
    before=module.snapshot(algo)
    for name,key in resume.SOURCE_KEYS.items():
        saved[key]=None if name.endswith('obs_normalizer') else before[name]
    torch.save(saved,source)
    module.CHECKPOINT_SHA256=hashlib.sha256(source.read_bytes()).hexdigest()
    if corrupt:
        with torch.no_grad():
            algo.actor.weight.add_(1)
    return algo,module,source


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.output=Path(tempfile.mkdtemp(prefix='php-formal-cpu-',dir='/workspace'))

    def scope(self):
        return {'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
                'gpu':'1x4090D-24G','num_envs':128,'updates':10000,
                'outer_worker_seconds':1800,'training_phase':'formal_30m_continuation',
                'worker_seconds':1600,'code_commit':'a'*40,'billing_acceptance':'bounded test',
                'output_upload_probe_passed':True}

    def run_algo(self,algo,source):
        with patch.dict(os.environ,{'PHP_FLUX_OUTPUT':str(self.output),'PHP_FLUX_RESUME_CHECKPOINT':str(source)}):
            algo.learn()

    def test_soft_stop_saves_reloads_and_accepts_partial_native_budget(self):
        algo,module,source=formal_fixture(self.output)
        raw=source.read_bytes()
        self.run_algo(algo,source)
        report=json.loads((self.output/'resume_validation.json').read_text())
        self.assertTrue(report['passed'])
        self.assertEqual(report['completed_updates'],3)
        self.assertEqual(report['native_final_iteration'],22)
        self.assertEqual(report['termination_reason'],'worker_time_budget')
        self.assertTrue(all(report['parameters_changed'].values()))
        self.assertTrue(all(x['min_step_increment']==3 for x in report['optimizer_step_advancement'].values()))
        self.assertEqual(source.read_bytes(),raw)
        self.assertEqual(len((self.output/'metrics.jsonl').read_text().splitlines()),3)
        self.assertTrue(json.loads((self.output/'native_reload.json').read_text())['native_load_passed'])

    def test_corrupt_native_load_rejected_before_training(self):
        algo,module,source=formal_fixture(self.output,corrupt=True)
        with self.assertRaises(ValueError):
            self.run_algo(algo,source)
        self.assertFalse((self.output/'resume_loaded.json').exists())
        self.assertFalse((self.output/'progress.json').exists())

    def test_source_hash_rejected(self):
        algo,module,source=formal_fixture(self.output)
        module.CHECKPOINT_SHA256='bad'
        with self.assertRaises(ValueError):
            self.run_algo(algo,source)

    def test_no_weight_updates_rejected(self):
        algo,module,source=formal_fixture(self.output,update=False)
        with self.assertRaisesRegex(ValueError,'No parameter update'):
            self.run_algo(algo,source)
        self.assertFalse((self.output/'resume_validation.json').exists())

    def test_optimizer_restore_must_be_enabled(self):
        algo,module,source=formal_fixture(self.output)
        algo.config.load_optimizer=False
        with self.assertRaises(ValueError):
            self.run_algo(algo,source)

    def test_scope_accepts_only_bounded_authorized_shape(self):
        contract.validate_scope(self.scope())
        for key,value in [('gpu','2x4090D-24G'),('num_envs',4096),('num_envs',True),
                          ('trial_count',2),('automatic_retries',1),('outer_worker_seconds',1801),
                          ('worker_seconds',1651),('worker_seconds',True),('worker_seconds',179),
                          ('updates',10001),('gpu_start_authorized',False),('code_commit',''),
                          ('output_upload_probe_passed',False)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                contract.validate_scope(self.scope()|{key:value})

    def test_all_generated_entrypoints_compile(self):
        for builder in (contract.teacher_source,contract.bootstrap_source,contract.worker_source):
            source=builder(ROOT)
            compile(source,'formal_entry_cpu_contract','exec')

    def test_original_smoke_and_resume_caps_not_modified(self):
        self.assertIn("('worker_seconds', 1200)",(ROOT/'bootstrap.py').read_text())
        self.assertIn('args.updates != 2',(ROOT/'teacher_worker.py').read_text())
        self.assertIn('self._flux_completed_updates != 2',(ROOT/'resume_algo.py').read_text())

    def test_generated_worker_budget_and_no_replay_requirement(self):
        source=contract.worker_source(ROOT)
        self.assertIn('signal.alarm(1680)',source)
        self.assertIn('int(1798-',source)
        self.assertIn('int(1650-',source)
        self.assertNotIn("training/replay_validation.json",source)
        self.assertIn("from flux.formal_input import fetch_checkpoint",source)
        self.assertIn("ROOT/'flux/formal_bootstrap.py'",source)
        self.assertIn("'.jsonl'",source)

    def test_formal_teacher_preserves_native_training_preset(self):
        source=contract.teacher_source(ROOT)
        self.assertIn('g1_wbt_terrain_ref_noscale',source)
        self.assertIn('save_interval=50',source)
        self.assertIn('FormalContinuationPPO',source)
        self.assertIn('if not resume_checkpoint or args.updates != 10000:',source)

    def test_reporting_flags_no_longer_unconditional_false(self):
        source=contract.bootstrap_source(ROOT)
        self.assertIn("'acceptance_passed': passed",source)
        self.assertIn("'checkpoint_reload_verified': bool(native.get('native_load_passed'))",source)
        self.assertIn("'teacher_converged': False",source)

    def supervisor_fixture(self, *, hard_timeout=False):
        namespace={'__file__':str(ROOT/'formal_bootstrap.py'),'__name__':'formal_supervisor_cpu'}
        exec(compile(contract.bootstrap_source(ROOT),'formal_bootstrap_cpu','exec'),namespace)
        approval=self.output/'approval.json'
        approval.write_text(json.dumps(self.scope()))
        output=self.output/'worker'
        output.mkdir()
        (output/'progress.json').write_text(json.dumps({'completed_updates':3,'stopped_early':True}))
        (output/'native_reload.json').write_text(json.dumps({'native_load_passed':True,'completed_updates':3}))
        (output/'resume_validation.json').write_text(json.dumps({'passed':True,'completed_updates':3}))
        (output/'model_smoke_final.pt').write_bytes(b'fixture only; not a real checkpoint')
        namespace['static_checks']=lambda **kw:{'php_commit':'a'*40}
        class Child:
            pid=12345678
            returncode=0
            calls=0
            def poll(self):
                self.calls+=1
                return None if hard_timeout or self.calls==1 else 0
            def wait(self,**kw):
                return 0
        clock=[0,0,1570 if hard_timeout else 1480,1600]
        argv=['formal_bootstrap.py','--approval-file',str(approval),'--data',str(self.output),'--output',str(output)]
        with patch.object(sys,'argv',argv),patch.dict(os.environ,{'PHP_SMOKE_OUTER_GROUP':'1'}),\
             patch('subprocess.Popen',return_value=Child()),patch('os.kill') as kill,\
             patch('time.monotonic',side_effect=clock),patch('time.sleep'):
            code=namespace['main']()
        return code,output,kill

    def test_supervisor_soft_deadline_accepts_verified_partial_updates(self):
        code,output,kill=self.supervisor_fixture()
        self.assertEqual(code,0)
        self.assertTrue((output/'STOP_AFTER_UPDATE').exists())
        report=json.loads((output/'completion.json').read_text())
        self.assertTrue(report['pipeline_completed'])
        self.assertTrue(report['acceptance_passed'])
        self.assertTrue(report['checkpoint_reload_verified'])
        self.assertFalse(report['hard_timeout'])
        self.assertFalse(report['teacher_converged'])
        self.assertEqual(report['completed_updates'],3)

    def test_supervisor_hard_deadline_is_failure_even_with_prior_checkpoint(self):
        code,output,kill=self.supervisor_fixture(hard_timeout=True)
        self.assertEqual(code,1)
        report=json.loads((output/'completion.json').read_text())
        self.assertTrue(report['hard_timeout'])
        self.assertFalse(report['pipeline_completed'])
        self.assertFalse(report['acceptance_passed'])
        self.assertEqual(kill.call_count,2)

    def test_contract_change_rejected_without_guessing(self):
        for text in ('unrelated','x x'):
            with self.assertRaises(ValueError):
                contract.replace_once(text,'x','y')

    def test_source_control_pin_and_field_validation(self):
        control={'url':'https://storage.googleapis.com/a/b?signature=fixture',
                 'source_task':formal_input.SOURCE_TASK,'checkpoint_sha256':formal_input.CHECKPOINT_SHA256}
        formal_input.validate_control(control)
        for change in ({'source_task':'TASK_20261025_027'},{'checkpoint_sha256':'wrong'},
                       {'extra':1},{'url':'https://example.com/a?signature=no'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                formal_input.validate_control(control|change)


if __name__=='__main__':
    unittest.main()
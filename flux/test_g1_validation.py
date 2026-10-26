"""CPU contracts; these are NOT simulator or real GPU acceptance evidence."""
import ast
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux.control_codec import encode_control, decode_control
from flux.g1_validation_contract import worker_source, teacher_source, bootstrap_source, validate_scope


def worker_namespace():
    ns = {'__file__': str(ROOT/'flux/run_smoke.py'), '__name__': 'cpu_fixture'}
    exec(compile(worker_source(ROOT/'flux'), str(ROOT/'flux/run_g1_validation.py'), 'exec'), ns)
    return ns


class ValidationContracts(unittest.TestCase):
    def test_padding_regression(self):
        self.assertEqual(encode_control({}), 'e30=')
        with self.assertRaises(Exception):
            base64.b64decode('e30', validate=True)
        self.assertEqual(decode_control('e30='), {})

    def test_real_worker_decoder_and_shell_roundtrip(self):
        decoder = worker_namespace()['decode_control']
        values = [{}, {'Content-Type': 'application/octet-stream'},
                  {'thread_id': 'fixture-thread', 'turn_id': 'fixture-turn'},
                  {'url': 'https://storage.googleapis.com/fixture/input?x=1&y=two+three%2F4', 'bytes': 43034938, 'sha256': 'a'*64},
                  {'unicode': '教师/低速台阶', 'punctuation': '+/_- ='}]
        for value in values:
            token = encode_control(value)
            self.assertEqual(len(token) % 4, 0)
            command = shlex.join(['gm-run', '1021php/flux/run_g1_validation.py', '--control', token])
            forwarded = shlex.split(command)[-1]
            self.assertEqual(decoder(forwarded), value)

    def test_actual_cli_parse_reaches_source_preflight(self):
        ns = worker_namespace()
        with tempfile.TemporaryDirectory(dir='/workspace', prefix='g1-codec-entry-') as temp:
            out = Path(temp)/'output'
            argv = ['run_g1_validation.py', '--task-id', 'TASK_20261026_999999', '--expected-commit', 'a'*40,
                    '--upload-url', 'https://storage.googleapis.com/fixture/output', '--probe-url', 'https://storage.googleapis.com/fixture/probe',
                    '--upload-headers-b64', encode_control({}), '--aria-provenance-b64', encode_control({'thread_id':'cpu'}),
                    '--lowstep-input-b64', encode_control({'url':'https://storage.googleapis.com/fixture/input','bytes':10,'sha256':'b'*64})]
            events = []
            ns['task_output'] = lambda _: out
            ns['upload'] = lambda *args: None
            ns['emit'] = lambda event, **kwargs: events.append((event, kwargs))
            with patch.object(sys, 'argv', argv), patch.object(ns['signal'], 'signal'), patch.object(ns['signal'], 'alarm'), patch.object(ns['subprocess'], 'check_output', side_effect=RuntimeError('CPU stop before source/network/simulator')):
                self.assertEqual(ns['main'](), 1)
            failure = json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['stage'], 'source_preflight')
            self.assertEqual(failure['exception_type'], 'RuntimeError')
            self.assertEqual(events[0][1]['num_envs'], 4096)
            self.assertEqual(events[0][1]['updates'], 20)
            self.assertFalse(json.loads((out/'result-manifest.json').read_text())['worker_passed'])

    def test_malformed_control_fails_before_output_or_gpu(self):
        ns = worker_namespace()
        argv = ['test', '--task-id', 'TASK_20261026_999999', '--expected-commit', 'a'*40,
                '--upload-url', 'https://storage.googleapis.com/fixture/output', '--probe-url', 'https://storage.googleapis.com/fixture/probe',
                '--upload-headers-b64', 'e30', '--lowstep-input-b64', 'e30=']
        with patch.object(sys, 'argv', argv), patch.dict(ns, {'task_output': lambda _: self.fail('Must reject before output/GPU')}):
            with self.assertRaises(ValueError): ns['main']()

    def test_bad_encoding_and_types(self):
        for token in ['e30', 'e30=\n', '!!!!', '____', 'bnVsbA==', 'W10=', 'InRlc3Qi', 'a'*12000]:
            with self.subTest(token=token[:10]), self.assertRaises(ValueError): decode_control(token)
        for value in [[], None, {'x':'z'*9000}, {'x':float('nan')}]:
            with self.assertRaises(ValueError): encode_control(value)

    def test_generated_sources_compile_and_stay_validation_only(self):
        for builder in [worker_source, bootstrap_source, teacher_source]:
            source = builder(ROOT/'flux')
            ast.parse(source)
            self.assertNotIn('G1LowstepPPO', source)
            self.assertNotIn('formal_started_monotonic', source)
            self.assertNotIn('updates = 20020', source)
        worker = worker_source(ROOT/'flux')
        self.assertIn('updates = 20', worker)
        self.assertIn("'formal_seconds':0", worker)
        self.assertIn('signal.alarm(1140)', worker)
        teacher = teacher_source(ROOT/'flux')
        self.assertIn('G1ValidationPPO', teacher)
        self.assertIn('save_interval=5', teacher)
        self.assertIn('resume_checkpoint = None', teacher)

    def test_scope_fails_closed(self):
        scope = {'gpu_start_authorized':True, 'trial_count':1, 'automatic_retries':0, 'gpu':'1x4090D-24G',
                 'num_envs':4096, 'updates':20, 'formal_seconds':0, 'dataset':'motions-low-step', 'fresh_training':True,
                 'output_upload_probe_passed':True, 'worker_seconds':1050, 'code_commit':'a'*40, 'billing_acceptance':'test'}
        self.assertEqual(validate_scope(scope), scope)
        for key, bad in [('updates',20020), ('formal_seconds',1800), ('num_envs',128), ('trial_count',2),
                         ('automatic_retries',1), ('worker_seconds',1081), ('fresh_training',False), ('worker_seconds',True)]:
            with self.subTest(key=key), self.assertRaises(ValueError): validate_scope({**scope, key:bad})

    def test_real_entry_help_without_gpu(self):
        for entry in ['run_g1_validation.py','g1_validation_bootstrap.py','g1_validation_teacher.py']:
            result = subprocess.run([sys.executable, str(ROOT/'flux'/entry), '--help'], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


class AlgorithmFixtures(unittest.TestCase):
    """CPU mocked CUDA fixtures, never real GPU evidence."""
    def run_fixture(self, updates=20, corruption=None, cuda=True, recipe_updates=20):
        import importlib.util, os, types
        import torch
        from flux.test_checkpoint_roundtrip import algorithm
        base = algorithm()
        def native_learn(algo):
            for index in range(updates):
                algo.current_learning_iteration = index
                algo._post_epoch_logging(index, {'loss':0.1})
        base.__mro__[1].learn = native_learn
        module = types.ModuleType('flux.teacher_algo')
        module.BoundedPPO = base
        spec = importlib.util.spec_from_file_location('g1_validation_fixture', ROOT/'flux/g1_validation_algo.py')
        loaded = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'flux.teacher_algo':module}): spec.loader.exec_module(loaded)
        out = Path(tempfile.mkdtemp(dir='/workspace', prefix='g1-validation-fixture-'))
        algo = loaded.G1ValidationPPO(out, corruption=corruption)
        algo.env = types.SimpleNamespace(num_envs=4096)
        algo.device = 'cpu'
        for key, value in dict(num_steps_per_env=24,num_learning_epochs=5,num_mini_batches=4,num_learning_iterations=recipe_updates,save_interval=5).items():
            setattr(algo.config, key, value)
        with patch.dict(os.environ, {'PHP_FLUX_OUTPUT':str(out)}), patch.object(torch.cuda,'is_available',return_value=cuda), patch.object(torch.cuda,'reset_peak_memory_stats'), patch.object(torch.cuda,'mem_get_info',return_value=(1,2)), patch.object(torch.cuda,'max_memory_allocated',return_value=1):
            if updates != 20 or corruption or not cuda or recipe_updates != 20:
                with self.assertRaises(ValueError): algo.learn()
                self.assertFalse((out/'validation_result.json').exists())
            else:
                algo.learn()
                result = json.loads((out/'validation_result.json').read_text())
                self.assertTrue(result['passed'])
                self.assertEqual(result['completed_updates'],20)
                self.assertEqual(result['formal_updates'],0)
                self.assertFalse((out/'formal_result.json').exists())

    def test_exact20_only(self): self.run_fixture()
    def test_incomplete_rejected(self): self.run_fixture(updates=19)
    def test_extra_update_rejected(self): self.run_fixture(updates=21)
    def test_requires_cuda(self): self.run_fixture(cuda=False)
    def test_refuses_formal_recipe(self): self.run_fixture(recipe_updates=20020)
    def test_corruption_blocks_success(self):
        for field in ['actor','critic','actor_obs_normalizer','critic_obs_normalizer','actor_optimizer','critic_optimizer','iteration','learning_rate']:
            with self.subTest(field=field): self.run_fixture(corruption=field)


if __name__ == '__main__': unittest.main()
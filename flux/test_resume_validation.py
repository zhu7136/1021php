"""CPU-only mock environment and contract tests; NOT real IsaacSim/CUDA proof."""
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
from flux import resume_input
from flux.test_checkpoint_roundtrip import algorithm

ROOT = Path(__file__).resolve().parents[1]


class ResumeInputTests(unittest.TestCase):
    def control(self):
        return {'url': 'https://storage.googleapis.com/bucket/object?signature=fixture',
                'source_task': 'TASK_20261025_027', 'checkpoint_sha256': resume_input.CHECKPOINT_SHA256}

    def test_valid_scoped_control(self):
        resume_input.validate_control(self.control())

    def test_bad_source_hash_and_fields(self):
        for key, value in [('source_task', 'TASK_20261025_026'), ('checkpoint_sha256', 'bad'), ('extra', 'no')]:
            control = self.control() | {key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume_input.validate_control(control)

    def test_non_scoped_or_credential_bearing_urls_rejected(self):
        for url in ('http://storage.googleapis.com/a?x=1', 'https://example.com/a?x=1',
                    'https://storage.googleapis.com/a', 'https://user:password@storage.googleapis.com/a?x=1',
                    'https://storage.googleapis.com:8080/a?x=1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                resume_input.validate_control(self.control() | {'url': url})

    def test_archive_corruption_rejected_before_output(self):
        directory = Path(tempfile.mkdtemp(prefix='php-resume-input-', dir='/workspace'))
        archive = directory/'corrupt.zip'
        archive.write_bytes(b'bad archive')
        with self.assertRaises(ValueError):
            resume_input.extract_checkpoint(archive, directory/'must-not-exist.pt')
        self.assertFalse((directory/'must-not-exist.pt').exists())

    def test_redirect_is_rejected(self):
        with self.assertRaises(ValueError):
            resume_input.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com')


def fixture(output, *, nonfinite=False, update=True, persistent_state=False):
    base = algorithm()
    def native_learn(self):
        self._train_mode()
        # Native PPO resets the same environment BEFORE its rollout context.
        self.env.reset_all()
        for index in range(self.current_learning_iteration, self.current_learning_iteration + 2):
            self.current_learning_iteration = index
            if update:
                for optimizer in (self.actor_optimizer, self.critic_optimizer):
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.grad = torch.ones_like(param)
                    optimizer.step()
                    optimizer.zero_grad()
            self._post_epoch_logging(index, {'loss': .1})
    base.__mro__[1].learn = native_learn
    teacher = types.ModuleType('flux.teacher_algo')
    teacher.BoundedPPO = base
    teacher._require_same_state = base.learn.__globals__['_require_same_state']
    spec = importlib.util.spec_from_file_location('resume_cpu_fixture', ROOT/'flux/resume_algo.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'flux.teacher_algo': teacher}):
        spec.loader.exec_module(module)
    algo = module.ReplayResumePPO(output)
    algo.empirical_normalization = True
    algo.config.num_learning_iterations = 2
    algo.current_learning_iteration = 19
    source = output/'source.pt'
    states = module.snapshot(algo)
    torch.save({**states, **{key: states[name] for name, key in module.SOURCE_KEYS.items()}, 'iter':19}, source)
    module.CHECKPOINT_SHA256 = resume_input.digest(source)
    algo.load(str(source))
    def mode(training):
        for name in module.MODULES:
            getattr(algo, name).train(training)
    algo._eval_mode = lambda: mode(False)
    algo._train_mode = lambda: mode(True)
    algo.get_inference_policy = lambda: lambda obs: algo.actor(obs['actor_obs'])
    algo.device = 'cpu'
    algo.num_act = 1
    algo.actor_obs_keys = ['a']
    class Environment:
        num_envs = 128
        calls = 0
        modes = None
        def reset_all(self):
            if persistent_state:
                if self.modes is None:
                    self.modes = []
                self.modes.append((torch.is_grad_enabled(),torch.is_inference_mode_enabled()))
                if not hasattr(self,'dof_pos'):
                    self.dof_pos = torch.ones(128,2)
                # Reproduces native reset's in-place persistent-tensor update.
                self.dof_pos[0] = 0
            return {'a': torch.ones(128,2)}
        def step(self, state):
            self.calls += 1
            if persistent_state:
                self.dof_pos = torch.ones(128,2) * 2
            obs = self.reset_all()
            if nonfinite:
                obs['a'][0,0] = float('nan')
            return obs, torch.ones(128), torch.zeros(128,dtype=torch.bool), {}
    algo.env = Environment()
    return algo, module, source


class ResumeAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.output = Path(tempfile.mkdtemp(prefix='php-resume-fixture-',dir='/workspace'))

    def run_with_source(self, algo, source):
        with patch.dict(os.environ, {'PHP_FLUX_OUTPUT':str(self.output), 'PHP_FLUX_RESUME_CHECKPOINT':str(source)}):
            algo.learn()

    def test_replay_then_two_actual_cpu_optimizer_updates_and_roundtrip(self):
        algo, module, source = fixture(self.output)
        original = source.read_bytes()
        self.run_with_source(algo,source)
        report = json.loads((self.output/'resume_validation.json').read_text())
        replay = json.loads((self.output/'replay_validation.json').read_text())
        self.assertTrue(report['passed'])
        self.assertEqual(report['completed_updates'], 2)
        self.assertEqual(report['native_final_iteration'],20)
        self.assertEqual(algo.env.calls,32)
        self.assertEqual(replay['environment_steps'],4096)
        self.assertTrue(replay['model_normalizers_optimizer_scalars_unchanged'])
        self.assertEqual(source.read_bytes(),original)
        self.assertTrue(all(report['parameters_changed'].values()))
        self.assertTrue(all(r['min_step_increment']==2 for r in report['optimizer_step_advancement'].values()))

    def disable_normalizers_as_native(self, algo, module, source):
        algo.empirical_normalization = False
        algo.actor_obs_normalizer = torch.nn.Identity()
        algo.critic_obs_normalizer = torch.nn.Identity()
        saved = torch.load(source, weights_only=True, map_location='cpu')
        for name in ('actor_obs_normalizer', 'critic_obs_normalizer'):
            saved[module.SOURCE_KEYS[name]] = None
        torch.save(saved, source)
        module.CHECKPOINT_SHA256 = resume_input.digest(source)
        return saved

    def test_shared_simulator_reset_after_no_grad_replay_then_updates(self):
        algo, module, source = fixture(self.output,persistent_state=True)
        self.disable_normalizers_as_native(algo,module,source)
        self.run_with_source(algo,source)
        self.assertEqual(algo.env.calls,32)
        self.assertEqual(algo.env.modes[:33],[(False,False)] * 33)
        self.assertEqual(algo.env.modes[33],(True,False))
        self.assertFalse(torch.is_inference(algo.env.dof_pos))
        self.assertTrue(json.loads((self.output/'resume_validation.json').read_text())['passed'])

    def test_old_inference_mode_reproduces_reset_failure_before_update(self):
        algo, module, source = fixture(self.output,persistent_state=True)
        self.disable_normalizers_as_native(algo,module,source)
        old_inference_mode = torch.inference_mode
        with patch.object(module.torch,'no_grad',old_inference_mode):
            with self.assertRaisesRegex(RuntimeError,'inference tensor'):
                self.run_with_source(algo,source)
        self.assertEqual(algo.env.calls,32)
        self.assertTrue((self.output/'replay_validation.json').exists())
        self.assertFalse((self.output/'resume_validation.json').exists())
        self.assertFalse((self.output/'progress.json').exists())

    def test_native_disabled_normalizers_none_identity_passes_full_cpu_flow(self):
        algo, module, source = fixture(self.output)
        self.disable_normalizers_as_native(algo, module, source)
        self.run_with_source(algo, source)
        report = json.loads((self.output/'resume_loaded.json').read_text())
        self.assertTrue(report['disabled_normalizers_verified_as_native_none_and_stateless_identity'])
        self.assertFalse(report['empirical_normalization'])
        self.assertEqual(algo.env.calls,32)
        self.assertTrue(json.loads((self.output/'resume_validation.json').read_text())['passed'])

    def test_enabled_normalizer_none_is_not_silently_accepted(self):
        algo, module, source = fixture(self.output)
        self.disable_normalizers_as_native(algo, module, source)
        algo.empirical_normalization = True
        with self.assertRaisesRegex(ValueError, 'Missing enabled native normalizer'):
            self.run_with_source(algo, source)
        self.assertEqual(algo.env.calls,0)

    def test_disabled_normalizer_cannot_hide_a_checkpoint_state(self):
        algo, module, source = fixture(self.output)
        self.disable_normalizers_as_native(algo, module, source)
        saved = torch.load(source, weights_only=True, map_location='cpu')
        saved['actor_obs_normalizer_state_dict'] = {'unexpected':torch.ones(1)}
        torch.save(saved,source)
        module.CHECKPOINT_SHA256 = resume_input.digest(source)
        with self.assertRaisesRegex(ValueError, 'Invalid disabled native normalizer'):
            self.run_with_source(algo, source)
        self.assertEqual(algo.env.calls,0)

    def test_disabled_none_rejects_nonidentity_runtime_module(self):
        algo, module, source = fixture(self.output)
        self.disable_normalizers_as_native(algo, module, source)
        algo.actor_obs_normalizer = torch.nn.Linear(2,2)
        with self.assertRaisesRegex(ValueError, 'Invalid disabled native normalizer'):
            self.run_with_source(algo, source)
        self.assertEqual(algo.env.calls,0)

    def test_disabled_identity_with_unexpected_buffer_rejected(self):
        algo, module, source = fixture(self.output)
        self.disable_normalizers_as_native(algo, module, source)
        algo.actor_obs_normalizer.register_buffer('unexpected',torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'Invalid disabled native normalizer'):
            self.run_with_source(algo, source)
        self.assertEqual(algo.env.calls,0)

    def test_hash_mismatch_fails_before_replay(self):
        algo, module, source = fixture(self.output)
        module.CHECKPOINT_SHA256 = 'wrong'
        with self.assertRaises(ValueError):
            self.run_with_source(algo, source)
        self.assertEqual(algo.env.calls,0)
        self.assertFalse((self.output/'resume_validation.json').exists())

    def test_loaded_model_corruption_fails_before_replay(self):
        algo, module, source = fixture(self.output)
        with torch.no_grad():
            algo.actor.weight.add_(1)
        with self.assertRaises(ValueError):
            self.run_with_source(algo, source)
        self.assertEqual(algo.env.calls,0)

    def test_nonfinite_replay_does_not_mark_success_or_resume(self):
        algo, module, source = fixture(self.output,nonfinite=True)
        with self.assertRaises(FloatingPointError):
            self.run_with_source(algo,source)
        self.assertFalse((self.output/'replay_validation.json').exists())
        self.assertFalse((self.output/'resume_validation.json').exists())
        self.assertFalse((self.output/'progress.json').exists())

    def test_no_parameter_updates_rejected(self):
        algo, module, source = fixture(self.output,update=False)
        with self.assertRaisesRegex(ValueError,'No parameter update'):
            self.run_with_source(algo,source)
        self.assertFalse((self.output/'resume_validation.json').exists())

    def test_disabled_optimizer_restore_rejected(self):
        algo, module, source = fixture(self.output)
        algo.config.load_optimizer = False
        with self.assertRaises(ValueError):
            self.run_with_source(algo,source)
        self.assertEqual(algo.env.calls,0)

    def test_replay_optimizer_mutation_rejected(self):
        algo, module, source = fixture(self.output)
        original = algo.env.step
        def mutated(state):
            next(iter(algo.actor_optimizer.state.values()))['exp_avg'].add_(1)
            return original(state)
        algo.env.step = mutated
        with self.assertRaisesRegex(ValueError,'read_only_replay'):
            self.run_with_source(algo,source)
        self.assertFalse((self.output/'replay_validation.json').exists())


if __name__ == '__main__':
    unittest.main()
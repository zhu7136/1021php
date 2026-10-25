"""CPU contract fixtures, NOT real Holosoma/IsaacSim native reload evidence."""
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

ROOT = Path(__file__).resolve().parents[1]


def algorithm():
    upstream = types.ModuleType('holosoma.agents.ppo.ppo')

    class NativeContractPPO:
        """Model the pinned upstream save-dir guard and checkpoint schema."""
        def __init__(self, output, *, corruption=None, save_error=False, load_error=False):
            self.log_dir = str(output / 'HuD04_WBT' / 'timestamped-run')
            Path(self.log_dir).mkdir(parents=True)
            self.config = types.SimpleNamespace(load_optimizer=True)
            self.actor = torch.nn.Linear(2, 1)
            self.critic = torch.nn.Linear(2, 1)
            self.actor_obs_normalizer = torch.nn.BatchNorm1d(2)
            self.critic_obs_normalizer = torch.nn.BatchNorm1d(2)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-5)
            # Populate nonempty optimizer buffers with tiny CPU fixture tensors.
            for opt in (self.actor_optimizer, self.critic_optimizer):
                for group in opt.param_groups:
                    for param in group['params']:
                        param.grad = torch.ones_like(param)
                opt.step()
                opt.zero_grad()
            self.actor_learning_rate = 1e-5
            self.critic_learning_rate = 2e-5
            self.current_learning_iteration = 0
            self.corruption = corruption
            self.save_error = save_error
            self.load_error = load_error
            self.saved = []
            self.loaded = []

        def learn(self):
            for index in range(3):
                self.current_learning_iteration = index
                self._post_epoch_logging(index, {'loss': 0.1})
            return 'fixture-complete'

        def _post_epoch_logging(self, it, loss_dict):
            pass

        def save(self, path, infos=None):
            # Exact pinned upstream guard; old adapter failed here on task023.
            if not path.startswith(self.log_dir):
                raise ValueError('Path is not in the logging directory')
            if self.save_error:
                raise OSError('fixture save failed')
            self.saved.append(path)
            state = {name: getattr(self, name).state_dict() for name in (
                'actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer',
                'actor_optimizer', 'critic_optimizer') if getattr(self, name) is not None}
            state.update(iter=self.current_learning_iteration)
            torch.save(state, path)

        def load(self, path):
            self.loaded.append(path)
            if self.load_error:
                raise RuntimeError('fixture load failed')
            state = torch.load(path, weights_only=True, map_location='cpu')
            for name in ('actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer'):
                if getattr(self, name) is not None:
                    getattr(self, name).load_state_dict(state[name])
            if self.config.load_optimizer:
                for name in ('actor_optimizer', 'critic_optimizer'):
                    getattr(self, name).load_state_dict(state[name])
                self.actor_learning_rate = self.actor_optimizer.param_groups[0]['lr']
                self.critic_learning_rate = self.critic_optimizer.param_groups[0]['lr']
            self.current_learning_iteration = state['iter']
            if self.corruption in ('actor', 'critic'):
                with torch.no_grad():
                    getattr(self, self.corruption).weight.add_(1)
            elif self.corruption in ('actor_obs_normalizer', 'critic_obs_normalizer'):
                getattr(self, self.corruption).running_mean.add_(1)
            elif self.corruption in ('actor_optimizer', 'critic_optimizer'):
                opt = getattr(self, self.corruption)
                next(iter(opt.state.values()))['exp_avg'].add_(1)
            elif self.corruption == 'optimizer_scalar':
                self.actor_optimizer.param_groups[0]['lr'] *= 2
            elif self.corruption == 'iteration':
                self.current_learning_iteration += 1
            elif self.corruption == 'learning_rate':
                self.actor_learning_rate *= 2

    upstream.PPO = NativeContractPPO
    spec = importlib.util.spec_from_file_location('flux_roundtrip_fixture', ROOT / 'flux/teacher_algo.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'holosoma.agents.ppo.ppo': upstream}):
        spec.loader.exec_module(module)
    return module.BoundedPPO


class CheckpointRoundtripTests(unittest.TestCase):
    def setUp(self):
        # Retain all test evidence under /workspace, no cleanup/deletion.
        self.output = Path(tempfile.mkdtemp(prefix='php-roundtrip-', dir='/workspace'))
        self.env = patch.dict(os.environ, {'PHP_FLUX_OUTPUT': str(self.output)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def report(self):
        return json.loads((self.output / 'native_reload.json').read_text())

    def assert_no_success(self):
        self.assertFalse((self.output / 'native_reload.json').exists())

    def test_nested_log_dir_roundtrip_and_relative_report(self):
        algo = algorithm()(self.output)
        self.assertEqual(algo.learn(), 'fixture-complete')
        final = Path(algo.log_dir) / 'model_smoke_final.pt'
        self.assertEqual(algo.saved, [str(final)])
        self.assertEqual(algo.loaded, [str(final)])
        self.assertGreater(final.stat().st_size, 0)
        self.assertFalse((self.output / 'model_smoke_final.pt').exists())
        report = self.report()
        self.assertTrue(report['native_load_passed'])
        self.assertEqual(report['path'], final.relative_to(self.output).as_posix())
        self.assertEqual(report['completed_updates'], 3)
        self.assertEqual(report['iteration'], 2)
        self.assertEqual(report['checked_optimizers'], ['actor_optimizer', 'critic_optimizer'])
        self.assertEqual(report['checked_modules'], ['actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer'])
        self.assertEqual(algo.current_learning_iteration, 2)

    def test_old_upper_directory_save_reproduces_worker_error(self):
        algo = algorithm()(self.output)
        with self.assertRaisesRegex(ValueError, 'not in the logging directory'):
            algo.save(str(self.output / 'model_smoke_final.pt'))
        self.assert_no_success()

    def test_state_corruption_fails_without_success_marker(self):
        for corruption in ('actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer',
                           'actor_optimizer', 'critic_optimizer', 'optimizer_scalar', 'iteration', 'learning_rate'):
            with self.subTest(corruption=corruption):
                directory = self.output / corruption
                directory.mkdir()
                with patch.dict(os.environ, {'PHP_FLUX_OUTPUT': str(directory)}):
                    algo = algorithm()(directory, corruption=corruption)
                    with self.assertRaisesRegex(ValueError, 'roundtrip changed'):
                        algo.learn()
                self.assertFalse((directory / 'native_reload.json').exists())

    def test_save_error_propagates_without_load_or_success(self):
        algo = algorithm()(self.output, save_error=True)
        with self.assertRaisesRegex(OSError, 'save failed'):
            algo.learn()
        self.assertEqual(algo.loaded, [])
        self.assert_no_success()

    def test_load_error_propagates_without_success(self):
        algo = algorithm()(self.output, load_error=True)
        with self.assertRaisesRegex(RuntimeError, 'load failed'):
            algo.learn()
        self.assertEqual(len(algo.loaded), 1)
        self.assert_no_success()

    def test_soft_stop_keeps_one_update_and_reloads_final(self):
        (self.output / 'STOP_AFTER_UPDATE').touch()
        algo = algorithm()(self.output)
        self.assertIsNone(algo.learn())
        self.assertTrue((Path(algo.log_dir) / 'model_00000.pt').is_file())
        self.assertEqual(self.report()['completed_updates'], 1)
        self.assertEqual(self.report()['iteration'], 0)
        self.assertTrue(json.loads((self.output / 'progress.json').read_text())['stopped_early'])

    def test_normalizers_can_be_disabled(self):
        algo = algorithm()(self.output)
        algo.actor_obs_normalizer = algo.critic_obs_normalizer = None
        algo.learn()
        self.assertEqual(self.report()['checked_modules'], ['actor', 'critic'])

    def test_optimizer_loading_disabled_is_reported_not_overridden(self):
        algo = algorithm()(self.output)
        algo.config.load_optimizer = False
        algo.learn()
        self.assertFalse(algo.config.load_optimizer)
        self.assertEqual(self.report()['checked_optimizers'], [])

    def test_log_dir_outside_output_rejected(self):
        algo = algorithm()(self.output)
        algo.log_dir = str(self.output.parent / (self.output.name + '-sibling'))
        with self.assertRaises(ValueError):
            algo.learn()
        self.assertEqual(algo.saved, [])
        self.assert_no_success()

    def test_nonfinite_final_checkpoint_rejected(self):
        algo = algorithm()(self.output)
        with torch.no_grad():
            algo.actor.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            algo.learn()
        self.assertEqual(algo.loaded, [])
        self.assert_no_success()

    def test_save_without_file_fails_before_load(self):
        algo = algorithm()(self.output)
        with patch.object(type(algo).__mro__[1], 'save', return_value=None):
            with self.assertRaisesRegex(ValueError, 'missing or empty'):
                algo.learn()
        self.assertEqual(algo.loaded, [])
        self.assert_no_success()

    def test_empty_saved_file_fails_before_load(self):
        algo = algorithm()(self.output)
        def empty_save(instance, path, infos=None):
            Path(path).write_bytes(b'')
        with patch.object(type(algo).__mro__[1], 'save', empty_save):
            with self.assertRaisesRegex(ValueError, 'missing or empty'):
                algo.learn()
        self.assertEqual(algo.loaded, [])
        self.assert_no_success()

    def test_training_error_is_not_swallowed_or_marked_success(self):
        algo = algorithm()(self.output)
        with patch.object(type(algo).__mro__[1], 'learn', side_effect=RuntimeError('fixture training failed')):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                algo.learn()
        self.assertEqual(algo.saved, [])
        self.assertEqual(algo.loaded, [])
        self.assert_no_success()

    def test_symlink_log_directory_escape_rejected(self):
        algo = algorithm()(self.output)
        other = Path(tempfile.mkdtemp(prefix='php-roundtrip-outside-', dir='/workspace'))
        link = self.output / 'linked-run'
        link.symlink_to(other, target_is_directory=True)
        algo.log_dir = str(link)
        with self.assertRaises(ValueError):
            algo.learn()
        self.assertEqual(algo.saved, [])
        self.assert_no_success()


if __name__ == '__main__':
    unittest.main()
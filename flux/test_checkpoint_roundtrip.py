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


class _DeviceTaggedCPUFixture:
    """CPU storage with mocked device labels; this is NOT a CUDA tensor."""
    def __init__(self, tensor, device, *, transfer_error=False):
        self.tensor = tensor
        self.device = torch.device(device)
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.transfer_error = transfer_error
        self.detached = False
        self.cpu_calls = 0

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        if not self.detached:
            raise AssertionError('Comparison must detach its input')
        self.cpu_calls += 1
        if self.transfer_error:
            raise RuntimeError('fixture CPU copy failed')
        return self.tensor.detach().clone()


class CrossDeviceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.compare = algorithm().learn.__globals__['_require_same_state']
        self.real_is_tensor = torch.is_tensor
        self.real_equal = torch.equal
        self.equal_calls = []
        self.is_tensor_patch = patch.object(torch, 'is_tensor', side_effect=self.is_tensor)
        self.equal_patch = patch.object(torch, 'equal', side_effect=self.equal)
        self.is_tensor_patch.start()
        self.equal_patch.start()
        self.addCleanup(self.is_tensor_patch.stop)
        self.addCleanup(self.equal_patch.stop)

    def is_tensor(self, value):
        return isinstance(value, _DeviceTaggedCPUFixture) or self.real_is_tensor(value)

    def equal(self, left, right):
        # Make the original code reproduce task026 even without a GPU.
        if left.device != right.device:
            raise RuntimeError('Expected all tensors to be on the same device, cpu and cuda:0')
        self.equal_calls.append((left, right))
        if isinstance(left, _DeviceTaggedCPUFixture):
            left = left.tensor
        if isinstance(right, _DeviceTaggedCPUFixture):
            right = right.tensor
        return self.real_equal(left, right)

    def tagged(self, value, device='cuda:0'):
        return _DeviceTaggedCPUFixture(torch.tensor(value), device)

    def assert_cpu_comparisons(self):
        self.assertTrue(self.equal_calls)
        for left, right in self.equal_calls:
            self.assertNotIsInstance(left, _DeviceTaggedCPUFixture)
            self.assertNotIsInstance(right, _DeviceTaggedCPUFixture)
            self.assertEqual(left.device.type, 'cpu')
            self.assertEqual(right.device.type, 'cpu')
            self.assertFalse(left.requires_grad)
            self.assertFalse(right.requires_grad)

    def test_cpu_cuda_label_equal_values_compare_on_cpu(self):
        expected = self.tagged(20., 'cpu')
        actual = self.tagged(20., 'cuda:0')
        self.compare(expected, actual, 'actor_optimizer.state.0.step')
        self.assert_cpu_comparisons()
        self.assertEqual(expected.cpu_calls, 1)
        self.assertEqual(actual.cpu_calls, 1)
        self.assertEqual(actual.device.type, 'cuda')

    def test_cuda_cpu_label_equal_values(self):
        self.compare(self.tagged([1., 2.]), self.tagged([1., 2.], 'cpu'), 'optimizer.buffer')
        self.assert_cpu_comparisons()

    def test_same_cuda_labels_still_compare_cpu_copies(self):
        self.compare(self.tagged([1., 2.]), self.tagged([1., 2.]), 'optimizer.buffer')
        self.assert_cpu_comparisons()

    def test_cross_device_value_mismatch_not_hidden(self):
        with self.assertRaisesRegex(ValueError, r'actor_optimizer.state.0.step'):
            self.compare(self.tagged(19., 'cpu'), self.tagged(20.), 'actor_optimizer.state.0.step')
        self.assert_cpu_comparisons()

    def test_dtype_and_shape_remain_exact(self):
        for left, right in ((torch.tensor([1.], dtype=torch.float32), torch.tensor([1.], dtype=torch.float64)),
                            (torch.tensor([1.]), torch.tensor([[1.]]))):
            with self.subTest(shape=tuple(right.shape), dtype=right.dtype):
                with self.assertRaises(ValueError):
                    self.compare(left, right, 'state')
        self.assertEqual(self.equal_calls, [])

    def test_nested_mixed_optimizer_state_and_parameter_groups(self):
        expected = {'state': {0: {'step': self.tagged(20., 'cpu'), 'exp_avg': self.tagged([.1, .2])}},
                    'param_groups': [{'lr': 1e-5, 'params': [0], 'betas': (.9, .999)}]}
        actual = {'state': {0: {'step': self.tagged(20.), 'exp_avg': self.tagged([.1, .2])}},
                  'param_groups': [{'lr': 1e-5, 'params': [0], 'betas': (.9, .999)}]}
        self.compare(expected, actual, 'optimizer')
        self.assert_cpu_comparisons()
        actual['param_groups'][0]['lr'] = 2e-5
        with self.assertRaisesRegex(ValueError, r'param_groups\[0\].lr'):
            self.compare(expected, actual, 'optimizer')

    def test_real_cpu_inputs_unchanged_and_gradients_detached(self):
        left = torch.tensor([1., 2.], requires_grad=True)
        right = torch.tensor([1., 2.], requires_grad=True)
        left.grad = torch.tensor([3., 4.])
        self.compare(left, right, 'actor')
        self.assert_cpu_comparisons()
        self.assertTrue(left.requires_grad and right.requires_grad)
        self.assertTrue(self.real_equal(left.grad, torch.tensor([3., 4.])))
        self.assertTrue(self.real_equal(left.detach(), torch.tensor([1., 2.])))
        self.assertEqual(left.device.type, 'cpu')

    def test_nonfinite_values_are_not_accepted(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.compare(self.tagged(value, 'cpu'), self.tagged(value), 'optimizer.state')

    def test_transfer_failure_propagates(self):
        actual = _DeviceTaggedCPUFixture(torch.tensor(1.), 'cuda:0', transfer_error=True)
        with self.assertRaisesRegex(RuntimeError, 'CPU copy failed'):
            self.compare(torch.tensor(1.), actual, 'optimizer.step')

    def run_roundtrip(self, *, transfer_error=False, corrupt=False):
        directory = Path(tempfile.mkdtemp(prefix='php-mixed-device-', dir='/workspace'))
        algo = algorithm()(directory)
        native_load = algo.load
        moved = []
        def load_with_tagged_optimizer(path):
            result = native_load(path)
            for opt in (algo.actor_optimizer, algo.critic_optimizer):
                state = next(iter(opt.state.values()))
                value = state['step'].clone()
                if corrupt:
                    value.add_(1)
                wrapped = _DeviceTaggedCPUFixture(value, 'cuda:0', transfer_error=transfer_error)
                state['step'] = wrapped
                moved.append((state, wrapped))
            return result
        algo.load = load_with_tagged_optimizer
        with patch.dict(os.environ, {'PHP_FLUX_OUTPUT': str(directory)}):
            if transfer_error or corrupt:
                error = RuntimeError if transfer_error else ValueError
                with self.assertRaises(error):
                    algo.learn()
                self.assertFalse((directory / 'native_reload.json').exists())
            else:
                algo.learn()
                report = json.loads((directory / 'native_reload.json').read_text())
                self.assertTrue(report['native_load_passed'])
                self.assertEqual(report['checked_optimizers'], ['actor_optimizer', 'critic_optimizer'])
        self.assertEqual(len(moved), 2)
        for state, wrapped in moved:
            self.assertIs(state['step'], wrapped)
            self.assertEqual(wrapped.device.type, 'cuda')

    def test_full_fixture_roundtrip_preserves_live_tagged_optimizer(self):
        self.run_roundtrip()
        self.assert_cpu_comparisons()

    def test_full_fixture_copy_failure_never_marks_success(self):
        self.run_roundtrip(transfer_error=True)

    def test_full_fixture_corruption_never_marks_success(self):
        self.run_roundtrip(corrupt=True)


if __name__ == '__main__':
    unittest.main()
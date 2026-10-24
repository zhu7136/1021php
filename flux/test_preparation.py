"""CPU-only tests. Passing these does NOT establish IsaacSim compatibility."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux.bootstrap import load_approval
from flux.preflight import static_checks


def scope():
    return dict(gpu_start_authorized=True, trial_count=1, automatic_retries=0,
                gpu='1x4090D-24G', worker_seconds=1200, num_envs=128, updates=20,
                billing_currency='test-only', billing_unit='test-only',
                billing_acceptance='test fixture, not real approval', code_commit='test-only',
                output_retrieval_verified=True)


def algorithm():
    import torch
    upstream = types.ModuleType('holosoma.agents.ppo.ppo')
    class FakePPO:
        def __init__(self, output):
            self.log_dir = str(output)
            self.actor = torch.nn.Linear(2, 1)
        def learn(self):
            for index in range(3):
                self._post_epoch_logging(index, {'loss': 0.1})
        def _post_epoch_logging(self, it, loss_dict):
            pass
        def save(self, path, infos=None):
            torch.save({'actor': self.actor.state_dict()}, path)
    upstream.PPO = FakePPO
    spec = importlib.util.spec_from_file_location('flux_test_algo', ROOT / 'flux/teacher_algo.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'holosoma.agents.ppo.ppo': upstream}):
        spec.loader.exec_module(module)
    return module.BoundedPPO


class PreparationTests(unittest.TestCase):
    def setUp(self):
        # Keep all scratch evidence; no cleanup/deletion is performed.
        self.path = Path(tempfile.mkdtemp(prefix='php-flux-test-', dir='/workspace'))

    def test_rejects_unapproved_or_excess_scope(self):
        changes = [dict(gpu_start_authorized=False), dict(updates=21), dict(num_envs=4096),
                   dict(automatic_retries=1), dict(worker_seconds=1201),
                   dict(billing_unit=''), dict(output_retrieval_verified=False)]
        for change in changes:
            with self.subTest(change=change):
                value = scope()
                value.update(change)
                path = self.path / 'scope.json'
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_approval(path)

    def test_bounded_scope_parser(self):
        path = self.path / 'scope.json'
        path.write_text(json.dumps(scope()))
        self.assertEqual(load_approval(path)['updates'], 20)

    def test_static_checks(self):
        checks = static_checks()
        self.assertEqual(checks['holosoma'], '4a0bf04cde5930af5ed59da797568d32be972525')
        self.assertGreater(checks['declared_dataset_files'], 0)

    def test_counts_completed_updates(self):
        with patch.dict('os.environ', {'PHP_FLUX_OUTPUT': str(self.path)}):
            algorithm()(self.path).learn()
        self.assertEqual(json.loads((self.path / 'progress.json').read_text())['completed_updates'], 3)

    def test_soft_stop_saves_at_update_boundary(self):
        (self.path / 'STOP_AFTER_UPDATE').touch()
        with patch.dict('os.environ', {'PHP_FLUX_OUTPUT': str(self.path)}):
            algorithm()(self.path).learn()
        self.assertEqual(json.loads((self.path / 'progress.json').read_text())['completed_updates'], 1)
        self.assertGreater((self.path / 'model_00000.pt').stat().st_size, 0)

    def test_nonfinite_parameters_fail(self):
        import torch
        algo = algorithm()(self.path)
        with torch.no_grad():
            algo.actor.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            algo.save(str(self.path / 'bad.pt'))


if __name__ == '__main__':
    unittest.main()
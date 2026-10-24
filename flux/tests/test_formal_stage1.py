"""CPU tests of authorized stage boundary; no simulator, GPU or remote task."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

FLUX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLUX))
import formal_training_hooks as hooks
import git_train_stage1 as stage


class FormalStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.pt'
        self.source.write_bytes(b'verified original input')
        self.out = self.root / 'stage'
        self.out.mkdir()
        self.clock = 0
        self.hash = hooks.digest(self.source)
        self.sha_patch = patch.object(hooks, 'CHECKPOINT_SHA', self.hash)
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)

    def algo(self, error=False):
        fixture = self
        class FakePPO:
            def __init__(self):
                self.env = SimpleNamespace(num_envs=32)
                self.config = SimpleNamespace(num_learning_iterations=hooks.ITERATION_CEILING,
                    save_interval=10, load_optimizer=True)
                self.log_dir = str(fixture.root)
                self.logging_helper = SimpleNamespace(writer=Mock())
                self.current_learning_iteration = 0
            def load(self, path):
                self.current_learning_iteration = 1
                self.loaded = Path(path)
            def _post_epoch_logging(self, it, losses):
                self.logging_helper.writer.add_scalar('Train/mean_reward', float(it), it)
            def learn(self):
                for it in range(self.current_learning_iteration, 100):
                    self.current_learning_iteration = it
                    fixture.clock += 1
                    if error:
                        raise ValueError('mock failure')
                    self._post_epoch_logging(it, {})
            def save(self, path):
                Path(path).write_bytes(('trained ' + str(self.current_learning_iteration)).encode())
        cls = hooks.instrument(FakePPO, self.out, self.source, 3, clock=lambda: fixture.clock)
        return cls()

    def test_resume_stops_at_completed_epoch_and_saves(self):
        algo = self.algo()
        algo.learn()
        summary = json.loads((self.out / 'training_stage_completed.json').read_text())
        self.assertEqual(summary['completed_updates'], 3)
        self.assertEqual(summary['last_iteration'], 4)
        self.assertEqual(summary['stop_reason'], 'approved_time_budget')
        self.assertEqual(hooks.digest(self.source), self.hash)
        self.assertEqual(hooks.digest(summary['final_checkpoint']), summary['final_checkpoint_sha256'])
        self.assertEqual(len((self.out / 'metrics.jsonl').read_text().splitlines()), 3)
        self.assertEqual(algo.loaded, self.source)

    def test_config_environment_mismatch_fails_closed(self):
        algo = self.algo()
        algo.env.num_envs = 128
        with self.assertRaises(ValueError): algo.learn()

    def test_optimizer_disabled_rejected(self):
        algo = self.algo()
        algo.config.load_optimizer = False
        with self.assertRaises(ValueError): algo.learn()

    def test_iteration_mismatch_rejected(self):
        algo = self.algo()
        algo.config.num_learning_iterations = 2
        with self.assertRaises(ValueError): algo.learn()

    def test_save_interval_mismatch_rejected(self):
        algo = self.algo()
        algo.config.save_interval = 100
        with self.assertRaises(ValueError): algo.learn()

    def test_checkpoint_tampering_prevents_resume(self):
        algo = self.algo()
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError): algo.learn()

    def test_exhausted_budget_never_starts_ppo(self):
        algo = self.algo()
        self.clock = 4
        with self.assertRaises(TimeoutError): algo.learn()

    def test_training_failure_has_no_success_manifest(self):
        algo = self.algo(error=True)
        with self.assertRaises(ValueError): algo.learn()
        self.assertTrue((self.out / 'failure.json').is_file())
        self.assertFalse((self.out / 'training_stage_completed.json').exists())

    def test_wrong_resume_iteration_rejected(self):
        algo = self.algo()
        algo.load = lambda path: setattr(algo, 'current_learning_iteration', 0)
        with self.assertRaises(ValueError): algo.learn()

    def test_scalar_nonfinite_stops_instead_of_logging(self):
        mirror = hooks.ScalarMirror(Mock(), self.out / 'bad.jsonl')
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError): mirror.add_scalar('loss', value, 0)
        self.assertFalse((self.out / 'bad.jsonl').exists())

    def test_scalar_mirror_preserves_original_writer(self):
        writer = Mock()
        mirror = hooks.ScalarMirror(writer, self.out / 'metrics.jsonl')
        mirror.add_scalar('reward', 3.5, 2)
        writer.add_scalar.assert_called_once_with('reward', 3.5, 2)
        self.assertEqual(json.loads((self.out / 'metrics.jsonl').read_text())['value'], 3.5)

    def test_output_upload_rejects_general_credentials(self):
        with patch.object(stage, 'ROOT', self.root), patch.object(stage, 'OUT', self.out):
            with self.assertRaises(ValueError):
                stage.seal_and_upload('https://storage.googleapis.com/scoped', {'Authorization': 'forbidden'})

    def test_output_upload_rejects_other_hosts(self):
        with patch.object(stage, 'ROOT', self.root), patch.object(stage, 'OUT', self.out):
            for url in ('http://storage.googleapis.com/file', 'https://example.com/file',
                        'https://user:password@storage.googleapis.com/file'):
                with self.assertRaises(ValueError): stage.seal_and_upload(url, {})

    def test_artifacts_are_sealed_and_uploaded_once(self):
        self.algo().learn()
        (self.root / 'holosoma_config.yaml').write_text('unchanged_config: true')
        (self.root / 'outputs').mkdir()
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(stage, 'ROOT', self.root), patch.object(stage, 'OUT', self.out), \
                patch('urllib.request.build_opener', return_value=opener):
            self.assertTrue(stage.seal_and_upload('https://storage.googleapis.com/scoped', {}))
        opener.open.assert_called_once()
        self.assertTrue((self.root / 'outputs/training_stage1_bundle.zip').is_file())

    def test_parent_requires_manifest_even_after_exit_zero(self):
        (self.root / 'outputs').mkdir()
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(stage, 'ROOT', self.root), patch.object(stage, 'OUT', self.out), \
                patch('urllib.request.build_opener', return_value=opener):
            self.assertFalse(stage.seal_and_upload('https://storage.googleapis.com/scoped', {}))

    def test_parent_rejects_tampered_final_checkpoint(self):
        self.algo().learn()
        (self.root / 'model_stage1_final.pt').write_bytes(b'tampered')
        with patch.object(stage, 'ROOT', self.root), patch.object(stage, 'OUT', self.out):
            with self.assertRaises(ValueError): stage.seal_and_upload('https://storage.googleapis.com/scoped', {})

    def test_authorized_worker_constants(self):
        self.assertEqual(stage.WORKER_SECONDS, 3600)
        self.assertLess(stage.SOFT_SECONDS, 3540)
        self.assertGreater(hooks.ITERATION_CEILING, 2)


if __name__ == '__main__':
    unittest.main()
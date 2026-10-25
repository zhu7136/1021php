"""Synthetic CPU fixtures only: never launch/import the simulator or CUDA."""
import builtins
import contextlib
import io
import importlib.machinery
import importlib.metadata
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux import runtime_versions as rv


class RuntimeVersionTests(unittest.TestCase):
    def setUp(self):
        # Retain fixtures under /workspace; no destructive cleanup.
        self.root = Path(tempfile.mkdtemp(prefix='php-runtime-version-', dir='/workspace'))
        self.lab = self.root / 'isaaclab'
        self.sim = self.lab / '_isaac_sim'
        self.exe = self.sim / 'kit/python/bin/python3'
        self.sim_origin = self.sim / 'python_packages/isaacsim/__init__.py'
        self.lab_origin = self.lab / 'source/isaaclab/isaaclab/__init__.py'
        for p, text in [(self.exe, ''), (self.sim / 'python.sh', ''),
                        (self.sim / 'VERSION', '5.1.0-rc.19\n'),
                        (self.sim_origin, 'raise AssertionError("must not import simulator")'),
                        (self.lab_origin, 'raise AssertionError("must not import lab")'),
                        (self.lab / 'VERSION', '2.3.1\n'), (self.lab / 'isaaclab.sh', ''),
                        (self.lab / 'source/isaaclab/config/extension.toml', '[package]\nversion="0.48.0"\n')]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self.dists = {'torch': '2.7.0+cu128', 'isaaclab': '0.48.0'}
        self.origins = {'isaacsim': self.sim_origin, 'isaaclab': self.lab_origin}

    def version(self, name):
        if name not in self.dists:
            raise importlib.metadata.PackageNotFoundError(name)
        return self.dists[name]

    def spec(self, name):
        origin = self.origins.get(name)
        return importlib.machinery.ModuleSpec(name, loader=None, origin=str(origin)) if origin else None

    def check(self):
        with patch.object(rv.metadata, 'version', side_effect=self.version), \
             patch.object(rv.importlib.util, 'find_spec', side_effect=self.spec), \
             patch.object(rv.sys, 'executable', str(self.exe)), \
             patch.object(rv.sys, 'version_info', (3, 11, 14)):
            return rv.check_runtime_versions()

    def test_actual_failure_fixture_standalone_without_pip_metadata(self):
        result = self.check()
        self.assertEqual(result['packages']['isaacsim'], '5.1.0')
        self.assertIsNone(result['runtime_version_evidence']['isaacsim']['distribution'])

    def test_lab_release_differs_from_extension_distribution(self):
        result = self.check()['runtime_version_evidence']['isaaclab']
        self.assertEqual((result['release'], result['extension'], result['distribution']),
                         ('2.3.1', '0.48.0', '0.48.0'))

    def test_evidence_records_raw_and_paths(self):
        result = self.check()['runtime_version_evidence']
        self.assertEqual(result['isaacsim']['standalone_versions'],
                         [{'path': str(self.sim / 'VERSION'), 'raw': '5.1.0-rc.19'}])
        self.assertEqual(result['isaaclab']['version_file'], str(self.lab / 'VERSION'))

    def test_stable_sim_release_with_build_suffix(self):
        (self.sim / 'VERSION').write_text('5.1.0+release.123')
        self.check()

    def test_pip_and_standalone_agree(self):
        self.dists['isaacsim'] = '5.1.0'
        self.check()

    def test_pip_only_importable_sim(self):
        self.exe = self.root / 'venv/bin/python'
        self.origins['isaacsim'] = self.root / 'venv/lib/isaacsim/__init__.py'
        self.dists['isaacsim'] = '5.1.0'
        self.assertEqual(self.check()['runtime_version_evidence']['isaacsim']['standalone_versions'], [])

    def test_standalone_interpreter_can_identify_before_module_visibility(self):
        self.origins['isaacsim'] = None
        self.check()

    def test_missing_all_sim_evidence_rejected(self):
        self.exe = self.root / 'venv/bin/python'
        self.origins['isaacsim'] = None
        with self.assertRaisesRegex(ValueError, 'Cannot verify IsaacSim'):
            self.check()

    def test_pip_metadata_without_importable_sim_rejected(self):
        self.exe = self.root / 'venv/bin/python'
        self.origins['isaacsim'] = None
        self.dists['isaacsim'] = '5.1.0'
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_sim_versions_and_malformed_text_rejected(self):
        for value in ('5.0.0', '5.1.1', '5.1.0-rc.18', '5.1.0 arbitrary', '', 'v5.1.0', '5.1.0\n5.0.0'):
            with self.subTest(value=value):
                (self.sim / 'VERSION').write_text(value)
                with self.assertRaises(ValueError):
                    self.check()

    def test_sim_pip_file_conflict_rejected(self):
        self.dists['isaacsim'] = '5.0.0'
        with self.assertRaises(ValueError):
            self.check()

    def test_unrelated_sim_module_rejected(self):
        self.origins['isaacsim'] = self.root / 'shadow/isaacsim/__init__.py'
        with self.assertRaisesRegex(ValueError, 'shadows'):
            self.check()

    def test_missing_standalone_version_not_hidden_by_pip(self):
        (self.sim / 'VERSION').rename(self.sim / 'VERSION.saved')
        self.dists['isaacsim'] = '5.1.0'
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.check()

    def test_wrong_lab_release_rejected_despite_matching_package(self):
        (self.lab / 'VERSION').write_text('2.3.0')
        with self.assertRaisesRegex(ValueError, 'product release'):
            self.check()

    def test_lab_distribution_conflict_rejected(self):
        self.dists['isaaclab'] = '2.3.1'
        with self.assertRaisesRegex(ValueError, 'conflict'):
            self.check()

    def test_lab_extension_mismatch_rejected(self):
        (self.lab / 'source/isaaclab/config/extension.toml').write_text('[package]\nversion="0.47.0"')
        with self.assertRaisesRegex(ValueError, 'extension'):
            self.check()

    def test_lab_without_pip_metadata_uses_anchored_files(self):
        self.dists.pop('isaaclab')
        self.assertIsNone(self.check()['runtime_version_evidence']['isaaclab']['distribution'])

    def test_missing_lab_release_rejected(self):
        (self.lab / 'VERSION').rename(self.lab / 'VERSION.saved')
        with self.assertRaises(ValueError):
            self.check()

    def test_lab_module_shadow_or_missing_rejected(self):
        for origin in (None, self.root / 'fake/isaaclab/__init__.py'):
            with self.subTest(origin=origin):
                self.origins['isaaclab'] = origin
                with self.assertRaises(ValueError):
                    self.check()

    def test_wrong_torch_and_missing_torch_rejected(self):
        for value in ('2.6.0', '2.7.0rc1', ''):
            self.dists['torch'] = value
            with self.assertRaises(ValueError):
                self.check()
        self.dists.pop('torch')
        with self.assertRaises(ValueError):
            self.check()

    def test_metadata_service_errors_not_swallowed(self):
        with patch.object(rv.sys, 'version_info', (3, 11, 14)), \
             patch.object(rv.metadata, 'version', side_effect=RuntimeError('broken metadata')):
            with self.assertRaisesRegex(RuntimeError, 'broken metadata'):
                rv.check_runtime_versions()

    def test_no_native_imports_no_processes_no_network(self):
        real_import = builtins.__import__
        def guarded(name, *args, **kw):
            if name.split('.')[0] in ('isaacsim', 'isaaclab', 'torch', 'subprocess', 'socket', 'requests'):
                raise AssertionError('Forbidden import: ' + name)
            return real_import(name, *args, **kw)
        with patch('builtins.__import__', side_effect=guarded):
            self.check()

    def test_oversized_version_evidence_rejected(self):
        (self.sim / 'VERSION').write_text('5.1.0' + ' ' * 4096)
        with self.assertRaisesRegex(ValueError, 'oversized'):
            self.check()

    def test_python_version_rejected(self):
        with patch.object(rv.sys, 'version_info', (3, 10, 0)):
            with self.assertRaisesRegex(ValueError, 'Python 3.11'):
                rv.check_runtime_versions()

    def test_bad_runtime_stops_before_downloads_install_and_teacher(self):
        from flux import preflight, run_smoke
        import json
        output = self.root / 'attempt'
        # Git invocation is mocked; no process, network, Flux call or GPU.
        commit = 'a' * 40
        argv = ['run_smoke.py', '--task-id', 'TASK_20261025_999', '--expected-commit', commit,
                '--upload-url', 'https://storage.googleapis.com/fake',
                '--probe-url', 'https://storage.googleapis.com/fake']
        checks = {'source_verification': 'cpu-fixture', 'shallow_checkout': True}
        with patch.object(sys, 'argv', argv), \
             patch.object(run_smoke, 'task_output', return_value=output), \
             patch.object(run_smoke.subprocess, 'check_output', return_value=commit), \
             patch.object(preflight, 'static_checks', return_value=checks), \
             patch.object(run_smoke, 'launcher', return_value=self.exe), \
             patch.object(run_smoke, 'run', side_effect=RuntimeError('cpu-fixture-version-failure')) as runner, \
             patch.object(run_smoke, 'ThreadPoolExecutor') as downloads, \
             patch.object(run_smoke, 'upload'), \
             patch.object(run_smoke.signal, 'signal'), patch.object(run_smoke.signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_smoke.main(), 1)
        downloads.assert_not_called()
        runner.assert_called_once()
        self.assertEqual(Path(runner.call_args.args[0][1]).name, 'runtime_versions.py')
        env = runner.call_args.args[1]
        self.assertFalse(set(env).intersection({'WANDB_API_KEY', 'OPENAI_API_KEY', 'FLUX_API_KEY', 'GITHUB_TOKEN', 'github', 'key'}))
        self.assertEqual(json.loads((output / 'failure.json').read_text())['stage'], 'runtime_version_preflight')
        self.assertFalse((output / 'inputs').exists())
        self.assertFalse((output / 'training').exists())


if __name__ == '__main__':
    unittest.main()
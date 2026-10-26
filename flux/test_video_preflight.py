import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from flux.video_preflight import encoder_preflight


class EncoderRepairTests(unittest.TestCase):
    def scratch(self):
        return tempfile.TemporaryDirectory(prefix='encoder-test-', dir='/workspace')

    def test_real_h264_encoding(self):
        with self.scratch() as d:
            env, evidence = encoder_preflight(Path(d)/'encoder', os.environ)
            self.assertTrue(evidence['encoder_preflight_passed'])
            self.assertGreater(evidence['probe_bytes'], 100)
            self.assertEqual(evidence['codec'], 'libx264')

    def test_bundled_encoder_without_system_path(self):
        with self.scratch() as d:
            env, evidence = encoder_preflight(Path(d)/'encoder', {'PATH': ''})
            self.assertTrue(evidence['encoder_preflight_passed'])
            self.assertTrue((Path(d)/'encoder/bin/ffmpeg').is_symlink())
            self.assertIn(str(Path(d)/'encoder/bin'), env['PATH'])

    def test_absent_dependency_fails_before_simulation(self):
        with self.scratch() as d, patch.dict(sys.modules, {'imageio_ffmpeg': None}):
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg unavailable'):
                encoder_preflight(Path(d)/'encoder', {'PATH': ''})

    def test_encoder_error_is_not_suppressed(self):
        with self.scratch() as d, patch('flux.video_preflight.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffmpeg')):
            with self.assertRaises(subprocess.CalledProcessError):
                encoder_preflight(Path(d)/'encoder', os.environ)

    def test_failure_exits_without_running_hanging_atexit(self):
        code = ('import atexit,time; from flux.video_preflight import fail_closed_entrypoint; '
                'atexit.register(lambda: time.sleep(60)); '
                'fail_closed_entrypoint(lambda: (_ for _ in ()).throw(ValueError("deliberate-test")))')
        p = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, text=True, timeout=6)
        self.assertEqual(p.returncode, 1)
        self.assertIn('deliberate-test', p.stderr)

    def test_final_progress_saved_before_video_encoding(self):
        source = (Path(__file__).parent/'g1_eval283_teacher.py').read_text()
        start = source.index('        for i,r in enumerate(block):')
        save = source.index('        save_progress()', start)
        stop = source.index('        if recorder is not None:recorder.stop_recording()', start)
        self.assertLess(save, stop)

    def test_preflight_precedes_simulator_evaluation(self):
        source = (Path(__file__).parent/'run_g1_eval283.py').read_text()
        self.assertLess(source.index("stage='encoder_preflight'"), source.index("stage='frozen_policy_evaluation'"))
        ast.parse(source)

    def test_encoder_dependency_is_pinned(self):
        row = json.loads((Path(__file__).parent/'eval-encoder-wheel.json').read_text())
        self.assertEqual(row['version'], '0.6.0')
        self.assertEqual(row['file'], 'imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl')
        self.assertEqual(row['bytes'], 29498237)
        self.assertEqual(row['sha256'], 'c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282')
        self.assertTrue(row['url'].startswith('https://files.pythonhosted.org/packages/'))

    def test_encoder_download_verified_before_offline_install(self):
        source = (Path(__file__).parent/'run_g1_eval283.py').read_text()
        self.assertLess(source.index('        download(encoder)'), source.index("str(private/encoder['file'])"))
        self.assertIn("'--isolated','install','--no-index','--no-deps'", source)


if __name__ == '__main__':
    unittest.main()
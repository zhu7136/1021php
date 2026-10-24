"""Fault-injected CPU tests: no external network, GPU or dependency install."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import git_smoke_v8 as smoke

URL = 'https://storage.googleapis.com/test/archive.zip?signature=never-log-this-marker'
PAYLOAD = b'mocked ZIP bytes, not production data' * 100


class Response(io.BytesIO):
    def __init__(self, payload=PAYLOAD, headers=None):
        super().__init__(payload)
        self.headers = {} if headers is None else headers


class DownloadDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/workspace/hud04_download_diagnostics_tests')
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=scratch))
        self.output = self.root/'archive.zip'
        self.stdout = io.StringIO()

    def download(self, response=None, error=None, timeout=1):
        class Opener:
            def open(self, *args, **kwargs):
                if error:
                    raise error
                return response
        with patch.object(smoke.urllib.request, 'build_opener', return_value=Opener()), \
             patch.object(smoke, 'ARCHIVE_SHA256', hashlib.sha256(PAYLOAD).hexdigest()), \
             redirect_stdout(self.stdout):
            smoke.fetch_archive(URL, self.output, timeout_seconds=timeout)

    def events(self):
        return [json.loads(line) for line in self.stdout.getvalue().splitlines() if line.startswith('{')]

    def test_success_reports_bytes_and_verified_hash_without_url(self):
        self.download(Response(headers={'Content-Length':str(len(PAYLOAD))}))
        self.assertEqual(self.output.read_bytes(), PAYLOAD)
        events = self.events()
        self.assertEqual(events[0]['event'], 'download_start')
        self.assertIn('download_progress', [e['event'] for e in events])
        self.assertEqual(events[-1]['event'], 'download_complete')
        self.assertEqual(events[-1]['bytes_downloaded'], len(PAYLOAD))
        self.assertTrue(events[-1]['sha256_verified'])
        self.assertNotIn('never-log-this-marker', self.stdout.getvalue())
        self.assertNotIn('https://', self.stdout.getvalue())

    def test_unknown_content_length_still_verifies_hash(self):
        self.download(Response())
        self.assertIsNone(self.events()[1]['content_length_bytes'])
        self.assertEqual(self.events()[-1]['event'], 'download_complete')

    def test_http_error_keeps_status_but_redacts_signed_url(self):
        error = urllib.error.HTTPError(URL, 403, 'Forbidden '+URL, {}, None)
        with self.assertRaises(RuntimeError) as raised:
            self.download(error=error)
        self.assertEqual(self.events()[-1]['http_status'], 403)
        self.assertEqual(self.events()[-1]['bytes_downloaded'], 0)
        self.assertNotIn('never-log-this-marker', str(raised.exception)+self.stdout.getvalue())

    def test_transport_exception_valueerror_is_also_redacted(self):
        with self.assertRaises(RuntimeError) as raised:
            self.download(error=ValueError(URL))
        self.assertNotIn('never-log-this-marker', str(raised.exception)+self.stdout.getvalue())

    def test_socket_timeout_is_reported_without_success(self):
        with self.assertRaisesRegex(RuntimeError, 'TimeoutError'):
            self.download(error=TimeoutError(URL))
        self.assertEqual(self.events()[-1]['event'], 'download_failed')
        self.assertNotIn('download_complete', [e['event'] for e in self.events()])

    def test_hash_mismatch_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.download(Response(b'corrupted'))
        self.assertEqual(self.events()[-1]['event'], 'download_failed')

    def test_truncated_content_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Content-Length'):
            self.download(Response(headers={'Content-Length':str(len(PAYLOAD)+1)}))
        self.assertEqual(self.events()[-1]['bytes_downloaded'], len(PAYLOAD))

    def test_oversized_header_is_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError, 'Content-Length'):
            self.download(Response(headers={'Content-Length':str(smoke.LIMIT_BYTES+1)}))
        self.assertEqual(self.events()[-1]['bytes_downloaded'], 0)

    def test_malformed_header_does_not_leak_value(self):
        with self.assertRaises(RuntimeError) as raised:
            self.download(Response(headers={'Content-Length':URL}))
        self.assertNotIn('never-log-this-marker', str(raised.exception)+self.stdout.getvalue())

    def test_stream_without_header_cannot_exceed_byte_ceiling(self):
        with patch.object(smoke, 'LIMIT_BYTES', 10), \
             self.assertRaisesRegex(ValueError, 'archive limit'):
            self.download(Response())
        self.assertLessEqual(self.output.stat().st_size, 10)

    def test_total_deadline_interrupts_stalled_read_and_restores_signal(self):
        class Stalled(Response):
            def read(self, count):
                time.sleep(5)
                return super().read(count)
        previous = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'DownloadDeadlineExceeded'):
            self.download(Stalled(), timeout=0.1)
        self.assertLess(time.monotonic()-started, 1)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(self.events()[-1]['bytes_downloaded'], 0)
        self.download(Response())
        self.assertEqual(self.events()[-1]['event'], 'download_complete')

    def test_total_deadline_also_bounds_connection_setup(self):
        class StalledOpener:
            def open(self, *args, **kwargs):
                time.sleep(5)
                return Response()
        started = time.monotonic()
        with patch.object(smoke.urllib.request, 'build_opener', return_value=StalledOpener()), \
             redirect_stdout(self.stdout), self.assertRaisesRegex(RuntimeError, 'DownloadDeadlineExceeded'):
            smoke.fetch_archive(URL, self.output, timeout_seconds=0.1)
        self.assertLess(time.monotonic()-started, 1)
        self.assertEqual(self.events()[-1]['bytes_downloaded'], 0)

    def test_slow_trickle_cannot_reset_total_deadline(self):
        class Trickle(Response):
            def read(self, count):
                time.sleep(0.03)
                return super().read(16)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'DownloadDeadlineExceeded'):
            self.download(Trickle(), timeout=0.13)
        self.assertLess(time.monotonic()-started, 1)
        self.assertGreater(self.events()[-1]['bytes_downloaded'], 0)
        self.assertLess(self.events()[-1]['bytes_downloaded'], len(PAYLOAD))
        self.assertEqual(self.output.stat().st_size, self.events()[-1]['bytes_downloaded'])

    def test_redirect_to_other_host_is_rejected(self):
        with self.assertRaises(ValueError):
            smoke.SafeRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com/file')

    def test_cli_rejects_invalid_deadline_without_launch(self):
        for value in ['0', '301', '-1']:
            with self.subTest(value=value):
                result = subprocess.run([sys.executable, str(Path(smoke.__file__)),
                    '--download-timeout-seconds', value], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn('between 1 and 300', result.stderr)

    def test_download_failure_never_installs_or_trains(self):
        with patch.object(smoke, 'ROOT', self.root), \
             patch.object(smoke, 'fetch_archive', side_effect=RuntimeError('mock download failure')), \
             patch.object(smoke.subprocess, 'run') as install, \
             patch('runpy.run_path') as train, redirect_stdout(self.stdout), \
             self.assertRaises(RuntimeError):
            smoke.worker(URL)
        install.assert_not_called()
        train.assert_not_called()
        self.assertEqual(self.events()[-1]['stage'], 'download_archive')
        self.assertEqual(self.events()[-1]['event'], 'stage_failed')

    def test_validation_failure_never_installs_or_trains(self):
        with patch.object(smoke, 'ROOT', self.root), patch.object(smoke, 'fetch_archive'), \
             patch.object(smoke, 'extract_candidates', side_effect=ValueError('mock invalid candidates')), \
             patch.object(smoke.subprocess, 'run') as install, \
             patch('runpy.run_path') as train, redirect_stdout(self.stdout), \
             self.assertRaises(ValueError):
            smoke.worker(URL)
        install.assert_not_called()
        train.assert_not_called()
        self.assertEqual(self.events()[-1]['stage'], 'extract_and_verify_candidates')

    def test_mock_worker_keeps_training_bounds_and_reports_phases(self):
        with patch.object(smoke, 'ROOT', self.root), patch.object(smoke, 'fetch_archive'), \
             patch.object(smoke, 'extract_candidates'), patch('offline_dependencies.install') as install, \
             patch('runpy.run_path', side_effect=SystemExit(0)) as train, \
             patch.object(sys, 'argv', []), patch.object(sys, 'path', list(sys.path)), \
             patch.dict(smoke.os.environ), redirect_stdout(self.stdout):
            smoke.worker(URL)
            argv = list(sys.argv)
        self.assertEqual(argv[argv.index('--num-envs')+1], '32')
        self.assertEqual(argv[argv.index('--iterations')+1], '2')
        self.assertEqual(argv[argv.index('--time-limit-seconds')+1], '900')
        install.assert_called_once_with(self.root)
        self.assertEqual(train.call_count, 1)
        stages = [e['stage'] for e in self.events() if e['event']=='stage_complete']
        self.assertEqual(stages, ['download_archive', 'extract_and_verify_candidates',
            'install_pinned_dependencies', 'training_entrypoint'])
        self.assertEqual(self.events()[-1]['event'], 'smoke_entrypoint_complete')

    def test_dependency_failure_never_calls_training(self):
        with patch.object(smoke, 'ROOT', self.root), patch.object(smoke, 'fetch_archive'), \
             patch.object(smoke, 'extract_candidates'), \
             patch('offline_dependencies.install', side_effect=subprocess.CalledProcessError(1, ['mock-pip'])), \
             patch('runpy.run_path') as train, redirect_stdout(self.stdout), \
             self.assertRaises(subprocess.CalledProcessError):
            smoke.worker(URL)
        train.assert_not_called()
        self.assertEqual(self.events()[-1]['stage'], 'install_pinned_dependencies')
        self.assertEqual(self.events()[-1]['event'], 'stage_failed')

    def test_nonzero_training_exit_is_not_marked_complete(self):
        with patch.object(smoke, 'ROOT', self.root), patch.object(smoke, 'fetch_archive'), \
             patch.object(smoke, 'extract_candidates'), patch('offline_dependencies.install'), \
             patch('runpy.run_path', side_effect=SystemExit(2)), \
             patch.object(sys, 'argv', []), patch.object(sys, 'path', list(sys.path)), \
             patch.dict(smoke.os.environ), redirect_stdout(self.stdout), \
             self.assertRaises(SystemExit):
            smoke.worker(URL)
        self.assertEqual(self.events()[-1]['event'], 'stage_failed')
        self.assertEqual(self.events()[-1]['stage'], 'training_entrypoint')
        self.assertNotIn('smoke_entrypoint_complete', [e['event'] for e in self.events()])


if __name__ == '__main__':
    unittest.main()
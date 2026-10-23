"""CPU-only Drive confirmation, safe redirects, wrapper and deadline tests."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import git_smoke_v8 as smoke

URL = 'https://drive.google.com/file/d/'+smoke.DRIVE_FILE_ID+'/view?usp=sharing'
DOWNLOAD = 'https://drive.usercontent.google.com/download?id='+smoke.DRIVE_FILE_ID+'&export=download'
PAYLOAD = b'PK\x03\x04approved immutable mock ZIP bytes'*100
HTML = ('<form id="download-form" method="get" action="https://drive.usercontent.google.com/download">'
        '<input name="id" value="'+smoke.DRIVE_FILE_ID+'"><input name="export" value="download">'
        '<input name="confirm" value="t"><input name="uuid" value="never-log-this-marker"></form>')


class Response(io.BytesIO):
    def __init__(self, payload, headers=None, url=DOWNLOAD):
        super().__init__(payload)
        self.headers = headers or {'Content-Length': str(len(payload)), 'Content-Type': 'application/zip'}
        self.url = url

    def geturl(self):
        return self.url


class GoogleDriveSourceTests(unittest.TestCase):
    def setUp(self):
        base = Path('/workspace/hud04_drive_unit_tests')
        base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=base))
        self.destination = self.root/'verified.zip'
        self.output = io.StringIO()

    def fetch(self, responses, timeout=1):
        self.calls = []
        owner = self
        class Opener:
            def open(self, request, **kwargs):
                owner.calls.append(request)
                value = responses.pop(0)
                if isinstance(value, BaseException):
                    raise value
                return value
        with patch.object(smoke.urllib.request, 'build_opener', return_value=Opener()), \
             patch.object(smoke, 'ARCHIVE_BYTES', len(PAYLOAD)), \
             patch.object(smoke, 'ARCHIVE_SHA256', hashlib.sha256(PAYLOAD).hexdigest()), \
             redirect_stdout(self.output):
            smoke.fetch_archive(URL, self.destination, timeout)

    def test_approved_view_url(self):
        smoke.check_url(URL)

    def test_approved_direct_download_url(self):
        smoke.check_url(DOWNLOAD)

    def test_wrong_file_id_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(URL.replace(smoke.DRIVE_FILE_ID, 'another-file'))

    def test_duplicate_file_id_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(DOWNLOAD+'&id='+smoke.DRIVE_FILE_ID)

    def test_unapproved_drive_path_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(DOWNLOAD.replace('/download?', '/login?'))

    def test_host_credentials_port_scheme_fragment_rejected(self):
        for url in [URL.replace('drive.google.com', 'drive.google.com.evil.example'),
                    URL.replace('https://', 'https://user:password@'),
                    URL.replace('drive.google.com', 'drive.google.com:8443'),
                    URL.replace('https:', 'http:'), URL+'#fragment']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                smoke.check_url(url)

    def test_confirmation_parser(self):
        form = smoke.DriveConfirmation()
        form.feed(HTML)
        result = form.next_url(URL)
        self.assertIn('confirm=t', result)
        smoke.check_drive_url(result)

    def test_login_form_not_accepted(self):
        form = smoke.DriveConfirmation()
        form.feed('<form id="login"><input name="password"></form>')
        with self.assertRaises(ValueError):
            form.next_url(URL)

    def test_cross_host_confirmation_rejected(self):
        form = smoke.DriveConfirmation()
        form.feed(HTML.replace('drive.usercontent.google.com', 'evil.example'))
        with self.assertRaises(ValueError):
            form.next_url(URL)

    def test_confirmation_post_rejected(self):
        form = smoke.DriveConfirmation()
        form.feed(HTML.replace('method="get"', 'method="post"'))
        with self.assertRaises(ValueError):
            form.next_url(URL)

    def test_duplicate_confirmation_input_rejected(self):
        form = smoke.DriveConfirmation()
        with self.assertRaises(ValueError):
            form.feed(HTML.replace('</form>', '<input name="id" value="another-file"></form>'))

    def test_drive_redirect_cannot_reach_other_allowed_provider(self):
        request = urllib.request.Request(DOWNLOAD)
        with self.assertRaises(ValueError):
            smoke.DriveRedirect().redirect_request(request, None, 302, 'Found', {},
                                                  'https://storage.googleapis.com/other.zip')

    def test_direct_download_verified(self):
        self.fetch([Response(PAYLOAD)])
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertIn('"download_complete"', self.output.getvalue())
        self.assertNotIn('https://', self.output.getvalue())
        self.assertIsNone(self.calls[0].get_header('Authorization'))

    def test_public_confirmation_then_wrapper_verified(self):
        wrapper = io.BytesIO()
        with zipfile.ZipFile(wrapper, 'w') as bundle:
            bundle.writestr('REPAIR_REPORT.md', b'ignored metadata')
            bundle.writestr('export/'+smoke.ARCHIVE_NAME, PAYLOAD)
        self.fetch([Response(HTML.encode(), {'Content-Type': 'text/html'}), Response(wrapper.getvalue())])
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn('never-log-this-marker', self.output.getvalue())
        self.assertNotIn('https://', self.output.getvalue())
        self.assertFalse((self.root/'REPAIR_REPORT.md').exists())

    def test_bad_payload_rejected(self):
        with self.assertRaises(ValueError):
            self.fetch([Response(b'PK\x03\x04'+b'X'*(len(PAYLOAD)-4))])
        self.assertNotIn('"download_complete"', self.output.getvalue())

    def test_non_zip_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not a ZIP'):
            self.fetch([Response(b'not a file')])

    def test_truncated_transfer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Content-Length'):
            self.fetch([Response(PAYLOAD, {'Content-Length': str(len(PAYLOAD)+1)})])

    def test_oversized_header_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Content-Length'):
            self.fetch([Response(PAYLOAD, {'Content-Length': str(smoke.LIMIT_BYTES+1)})])

    def test_oversized_confirmation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'page exceeds'):
            self.fetch([Response(b'X'*262145, {'Content-Type': 'text/html'})])

    def test_confirmation_loop_is_bounded(self):
        with self.assertRaisesRegex(ValueError, 'Too many'):
            self.fetch([Response(HTML.encode(), {'Content-Type': 'text/html'}) for _ in range(3)])
        self.assertEqual(len(self.calls), 3)

    def test_http_failure_no_retry_and_no_url_leak(self):
        error = urllib.error.HTTPError(DOWNLOAD+'&uuid=never-log-this-marker', 403, 'Forbidden', {}, None)
        with self.assertRaises(RuntimeError) as raised:
            self.fetch([error])
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('never-log-this-marker', str(raised.exception)+self.output.getvalue())

    def test_deadline_interrupts_stalled_read(self):
        class Stalled(Response):
            def read(self, count):
                time.sleep(5)
                return super().read(count)
        previous = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'DownloadDeadlineExceeded'):
            self.fetch([Stalled(PAYLOAD)], timeout=0.1)
        self.assertLess(time.monotonic()-started, 1)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
"""CPU-only tests; no network calls, installation or GPU execution."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import git_smoke_v8 as smoke


class GitBootstrapChecks(unittest.TestCase):
    def scratch(self):
        root = Path('/workspace/hud04_git_bootstrap_tests')
        root.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(dir=root))

    def test_only_approved_https_storage_urls(self):
        smoke.check_url('https://storage.googleapis.com/bucket/file?signature=test')
        for url in ['http://storage.googleapis.com/a', 'https://example.com/a',
                    'https://storage.googleapis.com.evil.example/a',
                    'https://user:password@storage.googleapis.com/a',
                    'https://storage.googleapis.com:8443/a']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                smoke.check_url(url)

    def test_approved_archive_digest_and_download(self):
        payload = b'fixture archive bytes'
        class Opener:
            def open(self, *args, **kwargs):
                return io.BytesIO(payload)
        out = self.scratch()/'download.zip'
        with patch.object(smoke.urllib.request, 'build_opener', return_value=Opener()), \
             patch.object(smoke, 'ARCHIVE_SHA256', hashlib.sha256(payload).hexdigest()):
            smoke.fetch_archive('https://storage.googleapis.com/a', out)
        self.assertEqual(out.read_bytes(), payload)

    def test_download_digest_mismatch_blocks_training(self):
        class Opener:
            def open(self, *args, **kwargs):
                return io.BytesIO(b'wrong file')
        with patch.object(smoke.urllib.request, 'build_opener', return_value=Opener()), \
             self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            smoke.fetch_archive('https://storage.googleapis.com/a', self.scratch()/'bad.zip')

    def test_archive_traversal_rejected(self):
        root = self.scratch()
        archive = root/'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(smoke.PREFIX+'../outside', b'not extracted')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
            smoke.extract_candidates(archive, root/'candidates')
        self.assertFalse((root/'outside').exists())

    def test_incomplete_batch_rejected(self):
        root = self.scratch()
        archive = root/'incomplete.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(smoke.PREFIX+'one_motion.npz', b'not a complete batch')
        with self.assertRaisesRegex(ValueError, 'all56'):
            smoke.extract_candidates(archive, root/'candidates')


if __name__ == '__main__':
    unittest.main()
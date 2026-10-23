"""CPU-only mount/archive checks; no Flux mutation, network or GPU use."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import git_smoke_v8 as smoke


class MountedArchiveTests(unittest.TestCase):
    def setUp(self):
        base = Path('/workspace/hud04_local_archive_tests')
        base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=base))
        self.payload = b'approved archive fixture' * 100
        self.source = self.root/'uploaded.zip'
        self.output = self.root/'staged.zip'
        self.size_patch = patch.object(smoke, 'ARCHIVE_BYTES', len(self.payload))
        self.hash_patch = patch.object(smoke, 'ARCHIVE_SHA256', hashlib.sha256(self.payload).hexdigest())
        self.size_patch.start()
        self.hash_patch.start()
        self.addCleanup(self.size_patch.stop)
        self.addCleanup(self.hash_patch.stop)

    def wrapper(self, names=None, payload=None):
        with zipfile.ZipFile(self.source, 'w') as bundle:
            bundle.writestr('REPAIR_REPORT.md', b'ignored metadata')
            for name in names or [smoke.ARCHIVE_NAME]:
                bundle.writestr(name, self.payload if payload is None else payload)

    def test_direct_archive_is_verified_without_network(self):
        self.source.write_bytes(self.payload)
        capture = io.StringIO()
        with patch.object(smoke, 'fetch_archive', side_effect=AssertionError('network forbidden')), redirect_stdout(capture):
            smoke.stage_local_archive(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.payload)
        event = json.loads(capture.getvalue())
        self.assertTrue(event['sha256_verified'])
        self.assertFalse(event['network_download'])
        self.assertFalse(event['export_wrapper'])

    def test_export_wrapper_preserves_original_payload_hash(self):
        self.wrapper()
        smoke.stage_local_archive(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.payload)
        self.assertFalse((self.root/'REPAIR_REPORT.md').exists())

    def test_export_with_safe_prefix_is_supported(self):
        self.wrapper(['version/'+smoke.ARCHIVE_NAME])
        smoke.stage_local_archive(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.payload)

    def test_corrupted_direct_archive_fails(self):
        self.source.write_bytes(b'X' * len(self.payload))
        with self.assertRaisesRegex(smoke.ArchiveValidationError, 'SHA256 mismatch'):
            smoke.stage_local_archive(self.source, self.output)

    def test_corrupted_inner_archive_fails(self):
        self.wrapper(payload=b'X' * len(self.payload))
        with self.assertRaisesRegex(smoke.ArchiveValidationError, 'SHA256 mismatch'):
            smoke.stage_local_archive(self.source, self.output)

    def test_incorrect_inner_size_fails(self):
        self.wrapper(payload=b'truncated')
        with self.assertRaises(smoke.ArchiveValidationError):
            smoke.stage_local_archive(self.source, self.output)

    def test_duplicate_inner_names_fail(self):
        self.wrapper([smoke.ARCHIVE_NAME, 'other/'+smoke.ARCHIVE_NAME])
        with self.assertRaisesRegex(smoke.ArchiveValidationError, 'exactly one'):
            smoke.stage_local_archive(self.source, self.output)

    def test_unsafe_prefix_fails(self):
        self.wrapper(['../'+smoke.ARCHIVE_NAME])
        with self.assertRaisesRegex(smoke.ArchiveValidationError, 'Unsafe'):
            smoke.stage_local_archive(self.source, self.output)

    def test_missing_file_fails(self):
        with self.assertRaises(smoke.ArchiveValidationError):
            smoke.stage_local_archive(self.source, self.output)

    def test_invalid_wrapper_fails(self):
        self.source.write_bytes(b'not a ZIP')
        with self.assertRaises(smoke.ArchiveValidationError):
            smoke.stage_local_archive(self.source, self.output)

    def test_relative_path_fails(self):
        with self.assertRaises(smoke.ArchiveValidationError):
            smoke.stage_local_archive(Path('personal/uploaded.zip'), self.output)

    def test_same_source_and_destination_fails(self):
        self.source.write_bytes(self.payload)
        with self.assertRaises(smoke.ArchiveValidationError):
            smoke.stage_local_archive(self.source, self.source)


if __name__ == '__main__':
    unittest.main()
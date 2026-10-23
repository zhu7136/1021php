"""CPU-only Git-parts integrity and CLI tests. Never launch Flux or a GPU."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import git_parts_v8 as parts
import git_smoke_v8 as smoke


class PartsTests(unittest.TestCase):
    def setUp(self):
        base = Path('/workspace/hud04_git_parts_unit_tests')
        base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=base))
        self.source = self.root/'parts'
        self.source.mkdir()
        self.output = self.root/'assembled.zip'
        self.payload = b'approved tiny archive payload!!'
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.record = {'format': 'hud04-v8-git-parts-1', 'archive_bytes': len(self.payload),
                       'archive_sha256': self.digest, 'part_bytes': 7, 'parts': []}
        self.assertEqual((len(self.payload)+6)//7, 5)
        for index in range(5):
            content = self.payload[index*7:(index+1)*7]
            name = f'archive.part{index:03d}'
            (self.source/name).write_bytes(content)
            self.record['parts'].append({'name': name, 'bytes': len(content),
                                       'sha256': hashlib.sha256(content).hexdigest()})
        self.write_manifest()
        patcher = patch.object(parts, 'PART_BYTES', 7)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_manifest(self):
        (self.source/'manifest.json').write_text(json.dumps(self.record))

    def assemble(self):
        parts.assemble_parts(self.source, self.output, len(self.payload), self.digest)

    def test_exact_reconstruction_without_network(self):
        with patch.object(smoke, 'fetch_archive', side_effect=AssertionError('network forbidden')):
            self.assemble()
        self.assertEqual(self.output.read_bytes(), self.payload)

    def test_missing_part(self):
        self.record['parts'][0]['name'] = 'missing'
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_missing_directory(self):
        self.source = self.root/'missing'
        with self.assertRaises(ValueError): self.assemble()

    def test_relative_directory(self):
        self.source = Path('relative')
        with self.assertRaises(ValueError): self.assemble()

    def test_part_symlink(self):
        name = 'archive.part000'
        path = self.source/name
        moved = self.root/'preserved-original'
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaises(ValueError): self.assemble()

    def test_manifest_symlink(self):
        path = self.source/'manifest.json'
        moved = self.root/'preserved-manifest'
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaises(ValueError): self.assemble()

    def test_directory_symlink(self):
        link = self.root/'link'
        link.symlink_to(self.source, target_is_directory=True)
        self.source = link
        with self.assertRaises(ValueError): self.assemble()

    def test_oversized_manifest(self):
        (self.source/'manifest.json').write_bytes(b' '*16385)
        with self.assertRaises(ValueError): self.assemble()

    def test_wrong_pinned_archive_hash(self):
        self.record['archive_sha256'] = '0'*64
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_wrong_pinned_archive_size(self):
        self.record['archive_bytes'] += 1
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_path_traversal(self):
        self.record['parts'][0]['name'] = '../archive.part000'
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_reordered_parts(self):
        self.record['parts'].reverse()
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_duplicate_part(self):
        self.record['parts'][1] = self.record['parts'][0]
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_wrong_part_count(self):
        self.record['parts'].pop()
        self.write_manifest()
        with self.assertRaises(ValueError): self.assemble()

    def test_lfs_pointer_instead_of_blob(self):
        (self.source/'archive.part000').write_bytes(b'version https://git-lfs.github.com/spec/v1')
        with self.assertRaises(ValueError): self.assemble()

    def test_corrupt_part(self):
        (self.source/'archive.part000').write_bytes(b'X'*7)
        with self.assertRaisesRegex(ValueError, 'part SHA256'): self.assemble()

    def test_manifest_cannot_approve_altered_payload(self):
        changed = b'X'*7
        (self.source/'archive.part000').write_bytes(changed)
        self.record['parts'][0]['sha256'] = hashlib.sha256(changed).hexdigest()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Assembled archive SHA256'): self.assemble()

    def test_output_never_overwritten(self):
        self.output.write_bytes(b'preserve me')
        with self.assertRaises(FileExistsError): self.assemble()
        self.assertEqual(self.output.read_bytes(), b'preserve me')

    def test_worker_cli_relative_source(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv',
                ['bootstrap', '--worker', '--data-parts-dir', 'flux/data/hud04-v8-parts']), \
                patch.object(smoke, 'worker') as worker:
            smoke.main()
        worker.assert_called_once_with(None, 180, None, str(smoke.ROOT/'flux/data/hud04-v8-parts'))

    def test_conflicting_environment_sources_rejected(self):
        with patch.dict(os.environ, {'HUD04_V8_DATA_FILE': '/some/file'}, clear=True), \
                patch.object(sys, 'argv', ['bootstrap', '--worker', '--data-parts-dir', str(self.source)]), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): smoke.main()

    def test_git_worker_calls_all56_verification_before_dependencies(self):
        class VerifiedStop(Exception): pass
        with patch.object(smoke, 'ROOT', self.root), patch.object(parts, 'assemble_parts') as assemble, \
                patch.object(smoke, 'fetch_archive', side_effect=AssertionError('network forbidden')), \
                patch.object(smoke, 'extract_candidates', side_effect=VerifiedStop) as verify, \
                patch.object(smoke.subprocess, 'run') as dependency_install, redirect_stdout(io.StringIO()):
            with self.assertRaises(VerifiedStop):
                smoke.worker(None, 180, None, str(self.source))
        assemble.assert_called_once()
        verify.assert_called_once()
        dependency_install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
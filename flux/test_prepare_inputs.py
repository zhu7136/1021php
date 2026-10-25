"""CPU-only tests for public input reconstruction; no simulator import."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('php_prepare_inputs', Path(__file__).with_name('prepare_inputs.py'))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PreparationInputTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='php-shard-test-', dir='/workspace'))
        self.source = self.root / 'source'
        self.source.mkdir()
        self.payload = b'verified fixture data\n'
        self.record = {'path': 'motions-locomotion/fixture.txt', 'bytes': len(self.payload),
                       'sha256': hashlib.sha256(self.payload).hexdigest()}
        self.tar = self.root / 'fixture.tar.gz'
        self.make_tar(self.tar, 'motions-locomotion/fixture.txt', self.payload)
        data = self.tar.read_bytes()
        parts = []
        for i, chunk in enumerate([data[:23], data[23:]]):
            path = self.source / f'fixture.part{i:03d}'
            path.write_bytes(chunk)
            parts.append({'index': i, 'path': path.name, 'bytes': len(chunk), 'sha256': module.digest(path)})
        self.manifest = {'schema_version': 1, 'gpu_start_authorized': False,
                         'archives': [{'name': 'fixture.tar.gz', 'bytes': len(data), 'sha256': module.digest(self.tar),
                                       'parts': parts, 'extracted_files': [self.record]}]}

    def make_tar(self, path, name, payload, symlink=False):
        with tarfile.open(path, 'w:gz') as archive:
            info = tarfile.TarInfo(name)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = '../outside'
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    def test_roundtrip(self):
        output = self.root / 'new'
        module.reconstruct(self.source, self.manifest, output)
        self.assertEqual((output / self.record['path']).read_bytes(), self.payload)

    def test_corrupt_part(self):
        (self.source / 'fixture.part000').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            module.verify_parts(self.source, self.manifest)

    def test_missing_part(self):
        bad = copy.deepcopy(self.manifest)
        bad['archives'][0]['parts'][0]['path'] = 'missing.part'
        with self.assertRaises(ValueError):
            module.verify_parts(self.source, bad)

    def test_out_of_order(self):
        bad = copy.deepcopy(self.manifest)
        bad['archives'][0]['parts'].reverse()
        with self.assertRaises(ValueError):
            module.verify_parts(self.source, bad)

    def test_unsafe_paths(self):
        for name in ['../escape', '/absolute', 'a/../b', 'a//b', 'a\\b', './a']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.safe_relative(name)

    def test_symlink_part(self):
        (self.source / 'linked').symlink_to(self.source / 'fixture.part000')
        with self.assertRaises(ValueError):
            module.regular(self.source, 'linked')

    def test_lfs_pointer(self):
        path = self.source / 'lfs'
        path.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaises(ValueError):
            module.verify_file(path, {'bytes': path.stat().st_size, 'sha256': module.digest(path)})

    def test_existing_output_refused(self):
        output = self.root / 'existing'
        output.mkdir()
        with self.assertRaises(FileExistsError):
            module.reconstruct(self.source, self.manifest, output)

    def test_tar_traversal(self):
        path = self.root / 'malicious.tar.gz'
        self.make_tar(path, '../escape', self.payload)
        with self.assertRaises(ValueError):
            module.safe_unpack(path, self.root / 'unpacked', [self.record])

    def test_tar_symlink_refused(self):
        path = self.root / 'symlink.tar.gz'
        self.make_tar(path, self.record['path'], self.payload, symlink=True)
        with self.assertRaises(ValueError):
            module.safe_unpack(path, self.root / 'unpacked', [self.record])

    def test_missing_private_wheels_refused(self):
        with self.assertRaises(ValueError):
            module.install(self.root, {}, Path('/unused-python'), None)

    def test_gpu_authorization_is_not_allowed_in_input_manifest(self):
        bad = copy.deepcopy(self.manifest)
        bad['gpu_start_authorized'] = True
        with self.assertRaises(ValueError):
            module.verify_parts(self.source, bad)


if __name__ == '__main__':
    unittest.main()
import hashlib
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import git_shards as g

ROOT = Path(__file__).parent / ('shard-tests-' + str(time.time_ns()))
ROOT.mkdir()


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self._testMethodName
        self.root.mkdir()
        self.payloads = {'base': b'base-test-123', 'data': b'data-test-456789'}
        self.sources = {}
        for name, data in self.payloads.items():
            p = self.root / (name + '.zip')
            p.write_bytes(data)
            self.sources[name] = p
        pins = {n: (len(b), hashlib.sha256(b).hexdigest()) for n, b in self.payloads.items()}
        self.pins = patch.object(g, 'INPUTS', pins)
        self.chunk = patch.object(g, 'CHUNK', 4)
        self.pins.start()
        self.chunk.start()
        self.addCleanup(self.pins.stop)
        self.addCleanup(self.chunk.stop)
        self.parts = self.root / 'parts'
        self.manifest = g.build(self.sources, self.parts)

    def restore(self):
        return g.restore(self.parts, self.root / 'restored')

    def write_manifest(self):
        (self.parts / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_roundtrip(self):
        result = self.restore()
        for name, data in self.payloads.items():
            self.assertTrue(result[name]['hash_verified'])
            self.assertEqual((self.root / 'restored' / (name + '.zip')).read_bytes(), data)

    def test_corrupt_part(self):
        (self.parts / 'base.zip.part-0000').write_bytes(b'bad!')
        with self.assertRaises(ValueError): self.restore()

    def test_missing_part(self):
        missing = self.root / 'missing'
        missing.mkdir()
        (missing / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError): g.restore(missing, self.root / 'restored')

    def test_lfs_pointer(self):
        (self.parts / 'base.zip.part-0000').write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaises(ValueError): self.restore()

    def test_symlink(self):
        link = self.root / 'linked-parts'
        link.symlink_to(self.parts, target_is_directory=True)
        with self.assertRaises(ValueError): g.restore(link, self.root / 'restored')

    def test_path_traversal(self):
        self.manifest['inputs']['base']['parts'][0]['path'] = '../base.zip'
        self.write_manifest()
        with self.assertRaises(ValueError): self.restore()

    def test_reordered_parts(self):
        self.manifest['inputs']['base']['parts'].reverse()
        self.write_manifest()
        with self.assertRaises(ValueError): self.restore()

    def test_modified_trusted_hash(self):
        self.manifest['inputs']['base']['sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaises(ValueError): self.restore()

    def test_no_overwrite(self):
        target = self.root / 'restored'
        target.mkdir()
        with self.assertRaises(FileExistsError): self.restore()


if __name__ == '__main__':
    unittest.main(verbosity=2)
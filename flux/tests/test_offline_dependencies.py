"""CPU-only unit tests: no real installer, simulator, or remote calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import offline_dependencies as offline


class OfflineDependencyTests(unittest.TestCase):
    def setUp(self):
        base=Path('/workspace/hud04_offline_deps/unit-tests')
        base.mkdir(parents=True,exist_ok=True)
        self.bundle=Path(tempfile.mkdtemp(dir=base))
        (self.bundle/'wheelhouse').mkdir()
        records=[]
        for index in range(53):
            name=f'fixture{index}-1-py3-none-any.whl'
            payload=f'fixture bytes {index}'.encode()
            (self.bundle/'wheelhouse'/name).write_bytes(payload)
            records.append({'filename':name,'bytes':len(payload),
                'sha256':hashlib.sha256(payload).hexdigest()})
        lock=''.join(f"fixture{i}==1 --hash=sha256:{r['sha256']}\n" for i,r in enumerate(records))
        (self.bundle/'requirements.lock').write_text(lock)
        self.manifest={'schema_version':1,'wheel_count':53,'wheels':records,
            'lock_sha256':hashlib.sha256(lock.encode()).hexdigest()}
        self.save_manifest()

    def save_manifest(self):
        (self.bundle/'manifest.json').write_text(json.dumps(self.manifest))

    def test_verified_fixture_bundle(self):
        self.assertEqual(offline.verify_bundle(self.bundle)['wheel_count'],53)

    def test_corrupt_wheel_rejected(self):
        path=self.bundle/'wheelhouse'/self.manifest['wheels'][0]['filename']
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):
            offline.verify_bundle(self.bundle)

    def test_missing_wheel_rejected(self):
        self.manifest['wheels'][0]['filename']='missing.whl'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'missing'):
            offline.verify_bundle(self.bundle)

    def test_symlink_rejected(self):
        link=self.bundle/'wheelhouse'/'linked.whl'
        link.symlink_to(self.bundle/'wheelhouse'/self.manifest['wheels'][0]['filename'])
        self.manifest['wheels'][0]['filename']=link.name
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'symlinked'):
            offline.verify_bundle(self.bundle)

    def test_traversal_rejected(self):
        self.manifest['wheels'][0]['filename']='../outside.whl'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            offline.verify_bundle(self.bundle)

    def test_lock_tampering_rejected(self):
        (self.bundle/'requirements.lock').write_text('changed')
        with self.assertRaisesRegex(ValueError,'lock digest mismatch'):
            offline.verify_bundle(self.bundle)

    def test_url_in_lock_rejected_even_with_matching_hash(self):
        lock=(self.bundle/'requirements.lock').read_text().replace('fixture0==1','https://example.com/package.whl')
        (self.bundle/'requirements.lock').write_text(lock)
        self.manifest['lock_sha256']=hashlib.sha256(lock.encode()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'lock contents'):
            offline.verify_bundle(self.bundle)

    def test_changed_original_pins_rejected(self):
        pins=self.bundle/'original.txt'
        pins.write_text('different pins')
        self.manifest['original_requirements_sha256']='0'*64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Original requirements changed'):
            offline.verify_bundle(self.bundle,pins)

    def test_installer_enforces_offline_hash_checked_binary_only(self):
        root=self.bundle/'repo'
        with patch.object(offline,'check_runtime'), \
             patch.object(offline,'verify_bundle',return_value={'wheel_count':53}), \
             patch.object(offline.subprocess,'run') as execute:
            offline.install(root)
        args=execute.call_args.args[0]
        for flag in ['--no-index','--no-deps','--no-cache-dir','--require-hashes','--only-binary=:all:','--isolated']:
            self.assertIn(flag,args)
        self.assertNotIn('--extra-index-url',args)
        self.assertTrue(execute.call_args.kwargs['check'])

    def test_runtime_rejection_never_invokes_pip(self):
        with patch.object(offline,'check_runtime',side_effect=RuntimeError('wrong runtime')), \
             patch.object(offline.subprocess,'run') as execute, \
             self.assertRaises(RuntimeError):
            offline.install(self.bundle)
        execute.assert_not_called()


if __name__=='__main__':
    unittest.main()
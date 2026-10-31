"""CPU-only import-resolution regression against the ORIGINAL pinned archive.

No IsaacSim/module body imports, no model inference, no training, no GPU.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
from overlay_package_fix import repair_overlay, ORIGINAL, PATCHED

PROBE = '''import importlib.util,json,sys
sys.path[:0]=[sys.argv[1],sys.argv[2]]
names=['wbt_training.config_values.motion_command','wbt_training.config_values.hud04_experiment','wbt_training.config_values.hud04_robot']
print(json.dumps({n:(s.origin if (s:=importlib.util.find_spec(n)) else None) for n in names}))
'''

class OverlayRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='overlay-test-',dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.bundle = Path(self.tmp.name)/'bundle'
        self.archive = ROOT/'git-reassembled-20261031/base.zip'
        with zipfile.ZipFile(self.archive) as z:
            for name in z.namelist():
                if name.endswith('.py') and any(name.startswith(p) for p in (
                    'bundle/overlay/wbt_training/','bundle/runtime/php/wbt_training/')):
                    target = Path(self.tmp.name)/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(name))
        self.target=self.bundle/'overlay/wbt_training/config_values/__init__.py'
        self.upstream=self.bundle/'runtime/php/wbt_training/config_values/motion_command.py'

    def probe(self):
        p=subprocess.run([sys.executable,'-I','-c',PROBE,
            str(self.bundle/'overlay'),str(self.bundle/'runtime/php')],
            capture_output=True,text=True,timeout=15,check=True)
        return json.loads(p.stdout)

    def test_original_archive_reproduces_shadowing(self):
        self.assertEqual(self.target.read_bytes(),ORIGINAL)
        self.assertIsNone(self.probe()['wbt_training.config_values.motion_command'])

    def test_fix_restores_upstream_and_preserves_overlay(self):
        original=self.upstream.read_bytes()
        receipt=repair_overlay(self.bundle)
        found=self.probe()
        self.assertEqual(Path(found['wbt_training.config_values.motion_command']),self.upstream)
        for n in ('hud04_experiment','hud04_robot'):
            self.assertEqual(Path(found['wbt_training.config_values.'+n]),self.target.parent/(n+'.py'))
        self.assertEqual(original,self.upstream.read_bytes())
        self.assertTrue(receipt['changed'])
        tree=ast.parse(original)
        self.assertIn('which_motion',[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))])

    def test_idempotent(self):
        repair_overlay(self.bundle)
        self.assertFalse(repair_overlay(self.bundle)['changed'])
        self.assertEqual(self.target.read_bytes(),PATCHED)

    def test_unknown_initializer_refused_unchanged(self):
        unknown=ORIGINAL+b'# unexpected version\n'
        self.target.write_bytes(unknown)
        with self.assertRaises(ValueError):repair_overlay(self.bundle)
        self.assertEqual(self.target.read_bytes(),unknown)

    def test_missing_upstream_refused(self):
        # Rename in disposable fixture only. No user data is removed.
        self.upstream.rename(self.upstream.with_suffix('.fixture-hidden'))
        with self.assertRaises(ValueError):repair_overlay(self.bundle)
        self.assertEqual(self.target.read_bytes(),ORIGINAL)

    def test_linked_initializer_refused(self):
        moved=self.target.with_suffix('.fixture-original')
        self.target.rename(moved)
        self.target.symlink_to(moved)
        with self.assertRaises(ValueError):repair_overlay(self.bundle)
        self.assertEqual(moved.read_bytes(),ORIGINAL)

    def test_linked_parent_refused(self):
        parent=self.target.parent
        moved=parent.with_name('config_values-fixture-original')
        parent.rename(moved)
        parent.symlink_to(moved,target_is_directory=True)
        with self.assertRaises(ValueError):repair_overlay(self.bundle)

    def test_only_nested_package_overlaps(self):
        with zipfile.ZipFile(self.archive) as z:
            a='bundle/overlay/wbt_training/';b='bundle/runtime/php/wbt_training/'
            sets=[{n[len(p):] for n in z.namelist() if n.startswith(p) and n.endswith('__init__.py')} for p in (a,b)]
            self.assertEqual(sets[0]&sets[1],{'__init__.py','config_values/__init__.py'})

if __name__=='__main__':unittest.main(verbosity=2)
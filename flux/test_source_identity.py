"""Real local depth-1 Git fixtures and CPU-only child diagnostics; no network/GPU."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux import preflight, run_smoke


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.scratch=Path(tempfile.mkdtemp(prefix='php-shallow-test-',dir='/workspace'))
        self.source=self.scratch/'source'
        self.source.mkdir()
        self.git(self.source,'init','-q')
        self.git(self.source,'config','user.name','CPU Fixture')
        self.git(self.source,'config','user.email','cpu@example.invalid')
        (self.source/'source.py').write_text('original=1\n')
        self.git(self.source,'add','source.py')
        self.git(self.source,'commit','-qm','upstream base')
        self.base=self.git(self.source,'rev-parse','HEAD')
        raw=subprocess.check_output(['git','-C',str(self.source),'ls-tree','-rz',self.base])
        self.digest=hashlib.sha256(raw).hexdigest()
        (self.source/'flux').mkdir()
        (self.source/'flux/entry.py').write_text('# integration\n')
        self.git(self.source,'add','flux/entry.py')
        self.git(self.source,'commit','-qm','integration only')
        self.head=self.git(self.source,'rev-parse','HEAD')
        self.shallow=self.scratch/'depth1'
        subprocess.run(['git','clone','-q','--depth','1',self.source.as_uri(),str(self.shallow)],check=True)

    def git(self,root,*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()

    def check(self,root,expected):
        with patch.object(preflight,'ROOT',root),patch.object(preflight,'PHP_COMMIT',self.base),patch.object(preflight,'SOURCE_TREE_SHA256',self.digest):
            return preflight.source_identity(expected)

    def test_reproduces_original_exit128(self):
        result=subprocess.run(['git','-C',str(self.shallow),'merge-base','--is-ancestor',self.base,'HEAD'],capture_output=True)
        self.assertEqual(result.returncode,128)
        self.assertIn(b'Not a valid commit name',result.stderr)

    def test_missing_history_shallow_passes_with_exact_head_and_tree(self):
        result=self.check(self.shallow,self.head)
        self.assertTrue(result['shallow_checkout'])
        self.assertEqual(result['source_verification'],'approved_head_and_pinned_source_tree')

    def test_full_history_checks_real_ancestry(self):
        result=self.check(self.source,self.head)
        self.assertFalse(result['shallow_checkout'])
        self.assertEqual(result['source_verification'],'full_history_and_pinned_source_tree')

    def test_wrong_approved_head_fails(self):
        with self.assertRaisesRegex(ValueError,'authorized commit'):
            self.check(self.shallow,'0'*40)

    def test_shallow_without_approval_fails(self):
        with self.assertRaisesRegex(ValueError,'explicit authorized commit'):
            self.check(self.shallow,None)

    def test_dirty_upstream_fails(self):
        (self.shallow/'source.py').write_text('tampered=1\n')
        with self.assertRaisesRegex(ValueError,'uncommitted'):
            self.check(self.shallow,self.head)

    def test_dirty_integration_fails_with_approved_head(self):
        (self.shallow/'flux/entry.py').write_text('# modified wrapper\n')
        with self.assertRaisesRegex(ValueError,'uncommitted'):
            self.check(self.shallow,self.head)

    def test_committed_upstream_change_fails_even_at_approved_head(self):
        (self.source/'source.py').write_text('tampered=2\n')
        self.git(self.source,'add','source.py')
        self.git(self.source,'commit','-qm','bad upstream change')
        with self.assertRaisesRegex(ValueError,'source tree'):
            self.check(self.source,self.git(self.source,'rev-parse','HEAD'))

    def test_added_nonintegration_file_fails(self):
        (self.source/'surprise.py').write_text('extra=1\n')
        self.git(self.source,'add','surprise.py')
        self.git(self.source,'commit','-qm','unexpected extra source')
        with self.assertRaisesRegex(ValueError,'source tree'):
            self.check(self.source,self.git(self.source,'rev-parse','HEAD'))

    def test_changed_gitlink_fails(self):
        self.git(self.source,'update-index','--add','--cacheinfo','160000,'+'1'*40+',thirdparty/holosoma')
        self.git(self.source,'commit','-qm','unexpected gitlink')
        with self.assertRaisesRegex(ValueError,'source tree'):
            self.check(self.source,self.git(self.source,'rev-parse','HEAD'))

    def test_full_history_invalid_base_fails_closed(self):
        tree=self.git(self.source,'rev-parse',self.base+'^{tree}')
        unrelated=self.git(self.source,'commit-tree',tree,'-m','unrelated root')
        with patch.object(preflight,'ROOT',self.source),patch.object(preflight,'PHP_COMMIT',unrelated),patch.object(preflight,'SOURCE_TREE_SHA256',self.digest):
            with self.assertRaisesRegex(ValueError,'ancestry'):
                preflight.source_identity(self.head)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.scratch=Path(tempfile.mkdtemp(prefix='php-child-log-test-',dir='/workspace'))

    def test_child_stdout_and_traceback_persist_on_failure(self):
        path=self.scratch/'bootstrap.log'
        command=[sys.executable,'-c','print("cpu-fixture",flush=True); raise ValueError("expected-cpu-fixture-error")']
        with self.assertRaisesRegex(RuntimeError,'Child exited with code'):
            run_smoke.run(command,{'PYTHONUNBUFFERED':'1'},10,path)
        text=path.read_text()
        self.assertIn('cpu-fixture',text)
        self.assertIn('Traceback',text)
        self.assertIn('expected-cpu-fixture-error',text)

    def test_bad_source_stops_before_launcher_downloads_and_install(self):
        actual=subprocess.check_output(['git','-C',str(run_smoke.ROOT),'rev-parse','HEAD'],text=True).strip()
        output=self.scratch/'attempt'
        argv=['run_smoke.py','--task-id','TASK_20261025_999','--expected-commit',actual,
              '--upload-url','https://storage.googleapis.com/fake','--probe-url','https://storage.googleapis.com/fake']
        with patch.object(sys,'argv',argv),patch.object(run_smoke,'task_output',return_value=output),\
             patch.object(preflight,'static_checks',side_effect=ValueError('bad-source-fixture')),\
             patch.object(run_smoke,'launcher') as launcher,patch.object(run_smoke,'ThreadPoolExecutor') as downloads,\
             patch.object(run_smoke,'upload'),patch.object(run_smoke.signal,'signal'),patch.object(run_smoke.signal,'alarm'),\
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_smoke.main(),1)
            launcher.assert_not_called()
            downloads.assert_not_called()
        result=json.loads((output/'failure.json').read_text())
        self.assertEqual(result['stage'],'source_preflight')
        self.assertTrue(result['frames'])
        self.assertFalse((output/'inputs').exists())


if __name__=='__main__':
    unittest.main()
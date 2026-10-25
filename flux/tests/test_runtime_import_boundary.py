"""Subprocess-only import-order regressions; never alter the test runner's module cache."""
import ast,json,os,pathlib,shutil,subprocess,sys,tempfile,unittest,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
import runtime_import_boundary as boundary

class RuntimeImportBoundaryTests(unittest.TestCase):
    def fixture(self):
        root=pathlib.Path(tempfile.mkdtemp(prefix='runtime-import-',dir='/workspace'))
        deps=root/'.flux-deps';deps.mkdir()
        wheel_dir=root/'flux/offline-deps/wheelhouse';wheel_dir.mkdir(parents=True)
        original=ROOT/'flux/offline-deps/wheelhouse'/boundary.WHEEL
        shutil.copyfile(original,wheel_dir/boundary.WHEEL)
        with zipfile.ZipFile(original) as z:(deps/'typing_extensions.py').write_bytes(z.read('typing_extensions.py'))
        return root
    def probe(self,root,code,success=True):
        prefix=f"import sys,types,pathlib,json; sys.path.insert(0,{str(ROOT/'flux')!r}); root=pathlib.Path({str(root)!r})\n"
        result=subprocess.run([sys.executable,'-c',prefix+code],capture_output=True,text=True,timeout=35)
        if success:self.assertEqual(result.returncode,0,result.stderr[-5000:])
        else:self.assertNotEqual(result.returncode,0,result.stdout[-1000:])
        return result
    def stale(self):
        return "old=types.ModuleType('typing_extensions'); old.__file__='/fixture/isaac/pip_prebundle/typing_extensions.py'; sys.modules['typing_extensions']=old\n"
    def test_original_path_insert_does_not_fix_preloaded_old_module(self):
        root=self.fixture()
        r=self.probe(root,self.stale()+"sys.path.insert(0,str(root/'.flux-deps'))\nfrom typing_extensions import Sentinel\n",success=False)
        self.assertIn("cannot import name 'Sentinel'",r.stderr)
    def test_preloaded_old_module_replaced_from_verified_wheel(self):
        root=self.fixture()
        self.probe(root,self.stale()+"from runtime_import_boundary import activate\nr=activate(root)\nfrom typing_extensions import Sentinel\nassert r['replaced_preloaded_typing_extensions'] and callable(Sentinel)\nassert sys.modules['typing_extensions'] is not old\n")
    def test_selected_module_has_sentinel_and_verified_origin(self):
        root=self.fixture()
        self.probe(root,"from runtime_import_boundary import activate\nr=activate(root)\nimport typing_extensions as t\nassert pathlib.Path(t.__file__).resolve()==(root/'.flux-deps/typing_extensions.py').resolve()\nassert r['sentinel_available'] and not r['network_download']\n")
    def test_activation_is_idempotent_before_other_dependency_imports(self):
        root=self.fixture()
        self.probe(root,"from runtime_import_boundary import activate\nactivate(root)\nfirst=sys.modules['typing_extensions']\nr=activate(root)\nassert sys.modules['typing_extensions'] is first\nassert not r['replaced_preloaded_typing_extensions']\n")
    def test_torch_and_pydantic_import_after_stale_cache_repair(self):
        root=self.fixture()
        self.probe(root,self.stale()+"from runtime_import_boundary import activate\nactivate(root)\nimport torch\nfrom pydantic import BaseModel\nclass Probe(BaseModel):\n    value:int\nassert Probe(value='2').value==2\nassert torch.tensor([1.]).sum().item()==1.\n")
    def test_already_loaded_torch_is_not_reloaded(self):
        root=self.fixture()
        self.probe(root,"import torch\noriginal=torch\n"+self.stale()+"from runtime_import_boundary import activate\nr=activate(root)\nassert sys.modules['torch'] is original and not r['torch_cuda_reloaded']\nassert torch.ones(2).sum().item()==2\n")
    def test_incompatible_preloaded_pydantic_rejected_not_purged(self):
        root=self.fixture()
        self.probe(root,"p=types.ModuleType('pydantic');p.__file__='/fixture/old/pydantic/__init__.py';sys.modules['pydantic']=p\nfrom runtime_import_boundary import activate\ntry:activate(root)\nexcept RuntimeError as e:assert 'incompatible dependency' in str(e)\nelse:raise AssertionError('must reject old Pydantic')\nassert sys.modules['pydantic'] is p\n")
    def test_corrupt_installed_module_rejected(self):
        root=self.fixture();(root/'.flux-deps/typing_extensions.py').write_text('corrupt=True\n')
        r=self.probe(root,"from runtime_import_boundary import activate\nactivate(root)\n",success=False)
        self.assertIn('differs from verified wheel',r.stderr)
    def test_corrupt_wheel_rejected(self):
        root=self.fixture();(root/'flux/offline-deps/wheelhouse'/boundary.WHEEL).write_bytes(b'corrupt')
        r=self.probe(root,"from runtime_import_boundary import activate\nactivate(root)\n",success=False)
        self.assertIn('wheel digest mismatch',r.stderr)
    def test_missing_installation_rejected_without_fallback(self):
        root=self.fixture();other=root/'elsewhere.py';(root/'.flux-deps/typing_extensions.py').rename(other)
        r=self.probe(root,"from runtime_import_boundary import activate\nactivate(root)\n",success=False)
        self.assertIn('installation required',r.stderr)
    def test_symlinked_installation_rejected(self):
        root=self.fixture();target=root/'.flux-deps/typing_extensions.py';other=root/'elsewhere.py';target.rename(other);target.symlink_to(other)
        r=self.probe(root,"from runtime_import_boundary import activate\nactivate(root)\n",success=False)
        self.assertIn('installation required',r.stderr)
    def test_training_and_eval_boundaries_precede_heavy_imports(self):
        for filename,functions in [('train_protected_resume.py',['train_worker','eval_worker']),
                                   ('evaluate_protected_multiseed.py',['worker'])]:
            tree=ast.parse((ROOT/'flux'/filename).read_text())
            for name in functions:
                fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                text=ast.unparse(fn)
                self.assertIn('activate(ROOT)',text,filename+':'+name)
                self.assertLess(text.index('activate(ROOT)'),text.index('import torch'),filename+':'+name)
                for node in fn.body[:2]:
                    if isinstance(node,ast.Import):self.assertNotIn('torch',[x.name for x in node.names])

if __name__=='__main__':unittest.main()
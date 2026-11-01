"""CPU regression of the actual native learn gates, without simulator/GPU."""
import ast,copy,os,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent

def native_gate():
    tree=ast.parse((ROOT/'native_resume.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='learn'))
    boundary=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='update_budget' for t in n.targets))
    fn.body=fn.body[:boundary]+[ast.Return(value=ast.Constant(True))]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace={'os':os,'time':time,'TRAIN_SECONDS':6600,'SOURCE_ITER':4,'MAX_UPDATES':1000}
    exec(compile(module,'actual_native_learn_gates','exec'),namespace)
    return namespace['learn']

class ScopeTests(unittest.TestCase):
    def fixture(self,num_envs=3000,mini=2):
        return SimpleNamespace(resume_verified=True,is_multi_gpu=False,env=SimpleNamespace(num_envs=num_envs),num_teachers=3,current_learning_iteration=4,
            config=SimpleNamespace(num_learning_iterations=1000,num_steps_per_env=24,num_learning_epochs=2,num_mini_batches=mini,distillation_warmup_steps=0,ppo_start_epoch=0,dagger_end_epoch=10000))
    def run_gate(self,obj):
        with patch.dict(os.environ,HUD04_RESUME_APPROVED='1000_updates_7200_seconds_once',HUD04_TRAIN_DEADLINE=str(time.monotonic()+6000)):
            return native_gate()(obj)
    def test_3000_env_actual_native_gate_accepts(self):self.assertTrue(self.run_gate(self.fixture()))
    def test_stale_4096_env_rejected(self):
        with self.assertRaises(ValueError):self.run_gate(self.fixture(num_envs=4096))
    def test_changed_minibatches_rejected(self):
        with self.assertRaises(ValueError):self.run_gate(self.fixture(mini=8))
    def test_all_entrypoints_bind_3000(self):
        child=(ROOT/'resume_child.py').read_text();parent=(ROOT/'resume_parent.py').read_text()
        self.assertIn('num_envs=3000',child);self.assertIn('num_envs!=3000',child)
        self.assertIn("prov.get('num_envs')!=3000",parent);self.assertIn('num_envs=3000',parent)
        self.assertNotIn('4096',child);self.assertNotIn('4096',parent)

if __name__=='__main__':unittest.main(verbosity=2)
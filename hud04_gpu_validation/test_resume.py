"""CPU-only resume, bound, and entrypoint tests; never submit a task."""
import copy,importlib.util,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from resume_contract import *

class ResumeTests(unittest.TestCase):
    def fixture(self):
        p=torch.nn.Module()
        for n in ('student','critic','depth_backbone'):setattr(p,n,torch.nn.Linear(2,2))
        p.std=torch.nn.Parameter(torch.ones(2));p.loaded_teachers=[True]*3
        opt=torch.optim.Adam([{'params':list(getattr(p,n).parameters())} for n in ('student','critic','depth_backbone')]+[{'params':[p.std]}],lr=3e-4)
        sum(x.square().sum() for x in p.parameters()).backward();opt.step();opt.zero_grad()
        a=SimpleNamespace(policy=p,optimizer=opt,current_learning_iteration=4,learning_rate=3e-4,ppo_coef=.0004)
        saved=copy.deepcopy(dict(model_state_dict=p.state_dict(),optimizer_state_dict=opt.state_dict(),iter=4,learning_rate=3e-4,ppo_coef=.0004,teacher_loaded=[True]*3))
        return a,saved
    def test_real_adam_equality(self):
        a,s=self.fixture();verify_loaded(a,s);self.assertEqual(len(s['optimizer_state_dict']['param_groups']),4)
    def test_reinitialized_optimizer_rejected(self):
        a,s=self.fixture();a.optimizer.state.clear()
        with self.assertRaises(ValueError):verify_loaded(a,s)
    def test_changed_model_rejected(self):
        a,s=self.fixture()
        with torch.no_grad():a.policy.std.add_(1)
        with self.assertRaises(ValueError):verify_loaded(a,s)
    def test_changed_iteration_rejected(self):
        a,s=self.fixture();a.current_learning_iteration=0
        with self.assertRaises(ValueError):verify_loaded(a,s)
    def test_changed_lr_rejected(self):
        a,s=self.fixture();a.learning_rate=1e-3
        with self.assertRaises(ValueError):verify_loaded(a,s)
    def test_changed_readiness_rejected(self):
        a,s=self.fixture();a.policy.loaded_teachers[0]=False
        with self.assertRaises(ValueError):verify_loaded(a,s)
    def test_no_repeated_last_iteration(self):self.assertEqual(next_iteration({'iter':4}),5)
    def test_other_source_rejected(self):
        with self.assertRaises(ValueError):next_iteration({'iter':5})
    def test_1000_update_ceiling(self):
        b=ResumeBudget(clock=lambda:0)
        for _ in range(1000):b.begin();b.complete()
        with self.assertRaises(ResumeBudgetExpired):b.begin()
    def test_time_ceiling(self):
        t=[0];b=ResumeBudget(seconds=2,clock=lambda:t[0]);t[0]=2
        with self.assertRaises(ResumeBudgetExpired):b.begin()
    def test_oversized_scope_rejected(self):
        for kw in ({'max_updates':1001},{'seconds':6601}):
            with self.assertRaises(ValueError):ResumeBudget(**kw)
    def entry_help(self,name,original):
        module=runpy.run_path(str(ROOT/name),run_name='cpu_only_test')
        saved=Path.read_text
        def read(p,*a,**k):
            if p.name==original:return saved(Path('/workspace/three-teacher-gpu/source')/original,*a,**k)
            return saved(p,*a,**k)
        with patch.object(Path,'read_text',read),patch.object(sys,'argv',[name,'--help']):
            with self.assertRaises(SystemExit) as e:module['main']()
            self.assertEqual(e.exception.code,0)
    def test_child_anchors_and_parse(self):self.entry_help('resume_child.py','three_teacher_child.py')
    def test_parent_anchors_and_parse(self):
        sys.path.insert(0,'/workspace/three-teacher-gpu/source')
        sys.path.insert(0,'/workspace/three-teacher-gpu')
        self.entry_help('resume_parent.py','three_teacher_parent.py')

if __name__=='__main__':unittest.main(verbosity=2)
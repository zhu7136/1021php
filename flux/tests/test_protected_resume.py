"""CPU tests of actual pinned native checkpoint load and bounded continuation wiring."""
import ast,copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'flux'),str(ROOT/'flux/tests')]
import train_protected_resume as resume
import checkpoint_contract
import test_fresh_pilot as native_tests

class ResumeTests(unittest.TestCase):
    def fixture(self,directory):
        helper=native_tests.FreshTests();a=helper.make_algo(True);a.current_learning_iteration=200
        a.actor_learning_rate=a.critic_learning_rate=1e-5
        for opt in [a.actor_optimizer,a.critic_optimizer]:
            for group in opt.param_groups:group['lr']=1e-5
        checkpoint_contract.attach_metadata(a,helper.config())
        p=Path(directory)/'source.pt';a.save(str(p));a.load(str(p))
        return a,p
    def verify(self,a,p):
        with patch.object(resume,'SOURCE_SHA',resume.base.digest(p)):
            return resume.verify_source_restore(a,p)
    def test_native_load_preserves_populated_moments_and_lrs(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d);self.assertTrue(self.verify(a,p)['actor_critic_optimizer_exact'])
            self.assertEqual(a.actor_learning_rate,1e-5)
    def test_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d)
            with self.assertRaisesRegex(ValueError,'hash'):resume.verify_source_restore(a,p)
    def test_weight_corruption_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d)
            with torch.no_grad():a.actor.weight.add_(1)
            with self.assertRaisesRegex(ValueError,'actor_model'):self.verify(a,p)
    def test_optimizer_moment_corruption_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d)
            next(iter(a.actor_optimizer.state.values()))['exp_avg'].add_(1)
            with self.assertRaisesRegex(ValueError,'optimizer'):self.verify(a,p)
    def test_lr_attribute_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d);a.actor_learning_rate=.001
            with self.assertRaisesRegex(ValueError,'LR'):self.verify(a,p)
    def test_iteration_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d);a.current_learning_iteration=199
            with self.assertRaisesRegex(ValueError,'iteration'):self.verify(a,p)
    def test_saved_normalization_contract(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            a,p=self.fixture(d);a.empirical_normalization=True
            with self.assertRaisesRegex(ValueError,'normalization'):self.verify(a,p)
    def test_boundaries_and_independent_eval_arms(self):
        self.assertEqual((resume.WORKER_SECONDS,resume.MAX_UPDATES,resume.TRAIN_SECONDS),(1800,200,300))
        self.assertEqual(len(resume.ARMS),7)
        self.assertEqual(resume.ARMS[0][0],'training')
        self.assertEqual(resume.SEEDS,(1402,2402,3402))
    def test_resume_starts_next_iteration_and_uses_alarm(self):
        tree=ast.parse(Path(resume.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_worker')
        text=ast.unparse(fn)
        self.assertIn('algo.current_learning_iteration = SOURCE_ITERATION + 1',text)
        self.assertLess(text.index('verify_source_restore'),text.index('trained = train'))
        self.assertIn('signal.setitimer(signal.ITIMER_REAL, TRAIN_SECONDS)',text)
        self.assertIn('roundtrip(algo, out /',text)
    def test_config_recovers_embedded_recipe_no_reapplication(self):
        tree=ast.parse(Path(resume.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='setup')
        text=ast.unparse(fn)
        self.assertIn('load_saved_experiment_config',text)
        self.assertNotIn('protected_config(',text)
        for field in ['reward','action','termination','robot','randomization','command']:self.assertIn(repr(field),text)
    def test_eval_is_frozen_and_continuous(self):
        source=Path(resume.__file__).read_text()
        self.assertIn('continuous_segments=True',source)
        self.assertIn('algo.actor_optimizer.step=forbid',source)
        self.assertIn('SynchronizedDiagnostics',source)
        self.assertIn("states['before']!=states['after']",source)

if __name__=='__main__':unittest.main()
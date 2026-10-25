"""CPU-only evidence that the dedicated worker cannot run a 200-update/eval job."""
import ast,json,pathlib,subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'flux'),str(ROOT/'flux/tests')]
import train_delivery_smoke as smoke
import train_protected_resume as resume
import low_noise_tools as tools
from test_low_noise import FakeAlgo

class SmokeTests(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        code=f'import sys;sys.path.insert(0,{str(ROOT/"flux")!r});import train_delivery_smoke;assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_configure_is_one_update_no_eval_with_same_bounds_in_subprocess(self):
        code=f'import sys;sys.path.insert(0,{str(ROOT/"flux")!r});import train_delivery_smoke as s;s.configure();r=s.resume;assert r.MAX_UPDATES==1 and r.ARMS==(("training",0.),) and r.SEEDS==() and r.WORKER_SECONDS==1800 and r.TRAIN_SECONDS==300;assert r.OUT.name=="delivery_smoke"'
        r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_wrapper_passed_as_child_entrypoint(self):
        with patch.object(smoke,'configure') as cfg,patch.object(resume,'main') as main:
            smoke.main();cfg.assert_called_once();self.assertEqual(pathlib.Path(main.call_args.kwargs['child_entrypoint']).name,'train_delivery_smoke.py')
    def test_stop_hook_prevents_second_update_even_if_native_loop_wants_hundreds(self):
        a=FakeAlgo();a.current_learning_iteration=201
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            stats=tools.EnvironmentStats(a.env)
            result=tools.train(a,stats,temp,lambda **k:None,clock=lambda:0,max_updates=1)
        self.assertEqual(result['completed_updates'],1)
        self.assertEqual(a.native_logs,1);self.assertEqual(a.native_steps,40)
        self.assertEqual(result['stop_reason'],'iteration_budget')
    def test_invalid_update_caps_rejected_before_learn(self):
        for cap in (0,201,-1,1.5):
            with self.subTest(cap=cap),self.assertRaises(ValueError):tools.train(None,None,'.',None,max_updates=cap)
    def summary_fixture(self,updates=1,iteration=201,restore=True):
        return dict(completed_updates=updates,final_checkpoint=dict(iteration=iteration,sha256='test'),restore=dict(actor_critic_optimizer_exact=restore))
    def run_summary(self,result):
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            root=pathlib.Path(temp);(root/'training').mkdir();(root/'training/summary.json').write_text(json.dumps(result))
            with patch.object(resume,'OUT',root),patch.object(resume,'emit'):
                smoke.summarize();return json.loads((root/'summary.json').read_text())
    def test_smoke_summary_is_not_policy_evaluation(self):
        r=self.run_summary(self.summary_fixture());self.assertEqual(r['model_evaluations'],0);self.assertFalse(r['policy_quality_evaluated'])
    def test_wrong_update_or_iteration_cannot_succeed(self):
        for args in ((2,202),(1,200),(200,400)):
            with self.subTest(args=args),self.assertRaises(ValueError):self.run_summary(self.summary_fixture(*args))
    def test_missing_full_native_restore_cannot_succeed(self):
        with self.assertRaises(ValueError):self.run_summary(self.summary_fixture(restore=False))
    def test_native_loop_range_and_cap_are_both_applied(self):
        source=(ROOT/'thirdparty/holosoma/src/holosoma/holosoma/agents/ppo/ppo.py').read_text()
        tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PPO')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='learn')
        text=ast.unparse(fn);self.assertIn('self.current_learning_iteration + self.config.num_learning_iterations',text)
        source=pathlib.Path(resume.__file__).read_text()
        self.assertIn('max_updates=MAX_UPDATES',source)
        self.assertIn('Path(child_entrypoint or __file__).resolve()',source)

if __name__=='__main__':unittest.main()
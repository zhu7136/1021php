"""CPU contract tests for the explicitly approved TASK042 continuation."""
import ast,copy,json,pathlib,subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'flux'),str(ROOT/'flux/tests')]
import train_task042_continuation as entry
import train_protected_resume as resume
import low_noise_tools as tools
from test_low_noise import FakeAlgo

class Task042Tests(unittest.TestCase):
    URL='https://storage.googleapis.com/wandb-production.appspot.com/zhul136-university-of-science-and-technology-of-china/HuD04_WBT/323f06da/resume_trained_checkpoint.pt?Signature=test'
    def test_import_before_runtime_is_stdlib_only(self):
        code=f'import sys;sys.path.insert(0,{str(ROOT/"flux")!r});import train_task042_continuation;assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_configure_enforces_exact_source_and_bounds(self):
        code=f'import sys;sys.path.insert(0,{str(ROOT/"flux")!r});import train_task042_continuation as e;e.configure();r=e.resume;assert r.SOURCE_ITERATION==201 and r.SOURCE_SHA==e.SOURCE_SHA and r.MAX_UPDATES==200 and r.TRAIN_SECONDS==300 and r.WORKER_SECONDS==1800 and r.ARMS==(("training",0.),) and not r.SEEDS;assert r.frozen.fetch_inputs is e.fetch_inputs'
        r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_exact_signed_source_accepted(self):entry.validate_source_url(self.URL)
    def test_wrong_sources_and_capabilities_rejected(self):
        for url in [self.URL.replace('323f06da','3znlrh25'),self.URL.replace('323f06da','94728bf7'),self.URL.replace('https:','http:'),self.URL.replace('storage.googleapis.com','evil.example'),self.URL.split('?')[0],self.URL+'#fragment',self.URL.replace('resume_trained_checkpoint.pt','model_final.pt')]:
            with self.subTest(url=url),self.assertRaises(ValueError):entry.validate_source_url(url)
    def fixture(self):
        return dict(completed_updates=200,source_iteration=201,source_sha256=entry.SOURCE_SHA,
            final_checkpoint=dict(iteration=401,sha256='test'),training_seconds=250,
            restore=dict(actor_critic_optimizer_exact=True))
    def summary(self,result):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            root=pathlib.Path(d);(root/'training').mkdir();(root/'training/summary.json').write_text(json.dumps(result))
            with patch.object(resume,'OUT',root),patch.object(resume,'emit'):
                entry.summarize();return json.loads((root/'summary.json').read_text())
    def test_summary_not_evaluation_or_convergence(self):
        s=self.summary(self.fixture());self.assertEqual(s['model_evaluations'],0);self.assertFalse(s['policy_quality_evaluated']);self.assertFalse(s['teacher_converged']);self.assertEqual(s['source_task'],'TASK_20261025_042')
    def test_summary_invalid_bounds_and_source_fail(self):
        changes=[('completed_updates',201),('completed_updates',0),('source_iteration',200),('source_sha256','other'),('training_seconds',301),('final_checkpoint',dict(iteration=2260)),('restore',dict(actor_critic_optimizer_exact=False))]
        for k,v in changes:
            r=self.fixture();r[k]=v
            with self.subTest(key=k,value=v),self.assertRaises(ValueError):self.summary(r)
    def test_one_completed_update_summary_allowed(self):
        r=self.fixture();r.update(completed_updates=1,final_checkpoint=dict(iteration=202,sha256='test'))
        self.assertEqual(self.summary(r)['final_checkpoint']['iteration'],202)
    def test_cpu_native_loop_stops_at_200(self):
        a=FakeAlgo();a.current_learning_iteration=202
        with tempfile.TemporaryDirectory(dir='/workspace') as d,patch('test_low_noise.range',side_effect=lambda *args:range(202,600) if args==(791,1100) else range(*args),create=True):
            result=tools.train(a,tools.EnvironmentStats(a.env),d,lambda **k:None,clock=lambda:0,max_updates=200)
        self.assertEqual(result['completed_updates'],200);self.assertEqual(result['last_iteration'],401);self.assertEqual(result['transitions'],153600)
    def test_training_loop_time_stop(self):
        a=FakeAlgo();clock=iter([0,301,301,301])
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            result=tools.train(a,tools.EnvironmentStats(a.env),d,lambda **k:None,clock=lambda:next(clock),max_updates=200)
        self.assertEqual(result['completed_updates'],1);self.assertEqual(result['stop_reason'],'time_budget')
    def test_parent_and_child_both_use_bounded_wrapper(self):
        with patch.object(entry,'configure'),patch.object(resume,'main') as m,patch.object(sys,'argv',['train_task042_continuation.py']):
            entry.main();self.assertEqual(pathlib.Path(m.call_args.kwargs['child_entrypoint']).name,'train_task042_continuation.py')
    def test_base_no_hardcoded200_metadata(self):
        tree=ast.parse(pathlib.Path(resume.__file__).read_text())
        for name in ['train_worker','main']:
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            text=ast.unparse(fn)
            self.assertNotIn('source_iteration=200',text);self.assertNotIn('completed_updates_before=200',text)

if __name__=='__main__':unittest.main()
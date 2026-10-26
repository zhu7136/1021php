"""CPU-only224 continuation contracts; not GPU/simulator evidence."""
import hashlib,importlib.util,json,os,shlex,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from flux import g1_resume224_contract as c
from flux import g1_resume224_input as inp
from flux.control_codec import encode_control

ROOT=Path(__file__).parent
def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class ContinuationContracts(unittest.TestCase):
    def control(self):return dict(url='https://storage.googleapis.com/fixture/input?x=1&y=2',bytes=inp.CHECKPOINT_BYTES,sha256=inp.CHECKPOINT_SHA256,source_task=inp.SOURCE_TASK,source_run=inp.SOURCE_RUN)
    def scope(self):return dict(gpu_start_authorized=True,trial_count=1,automatic_retries=0,gpu='1x4090D-24G',num_envs=4096,updates=20020,formal_seconds=1800,dataset='motions-low-step',fresh_training=False,output_upload_probe_passed=True,code_commit='a'*40,billing_acceptance='approved',worker_seconds=3000,resume_source_task=inp.SOURCE_TASK,resume_source_sha256=inp.CHECKPOINT_SHA256)
    def test_scope(self):c.validate_scope(self.scope())
    def test_scope_rejects_fresh_other_source_or_budget(self):
        for k,v in [('fresh_training',True),('resume_source_task','TASK_20261026_216'),('resume_source_sha256','0'*64),('formal_seconds',3600),('automatic_retries',1),('dataset','motions-locomotion')]:
            with self.subTest(k=k),self.assertRaises(ValueError):c.validate_scope(self.scope()|{k:v})
    def test_checkpoint_control(self):inp.validate_control(self.control())
    def test_wrong_checkpoint_rejected(self):
        for k,v in [('bytes',True),('bytes',10),('sha256','a'*64),('source_task','TASK_20261026_216'),('source_run','r2nio4q8'),('url','http://storage.googleapis.com/x'),('url','https://example.com/x')]:
            with self.subTest(k=k),self.assertRaises(ValueError):inp.validate_control(self.control()|{k:v})
    def test_generated_entrypoints(self):
        for builder in (c.worker_source,c.teacher_source,c.bootstrap_source):compile(builder(ROOT),'generated224','exec')
        t=c.teacher_source(ROOT);w=c.worker_source(ROOT);b=c.bootstrap_source(ROOT)
        self.assertIn("checkpoint=resume_checkpoint",t);self.assertIn('G1Resume224PPO',t);self.assertIn('fresh_training=False',t)
        self.assertNotIn('must train fresh',t);self.assertNotIn('args.updates != 2',t)
        self.assertIn("env['PHP_FLUX_RESUME_CHECKPOINT']=str(source)",w)
        self.assertIn('resume_result.json',b);self.assertIn('budget_times(validation_deadline,gate)',b)
        self.assertIn('signal.alarm(3180)',w);self.assertIn("'formal_seconds':1800",w)
    def test_actual_decoder_and_cli(self):
        ns={'__file__':str(ROOT/'run_smoke.py'),'__name__':'fixture224'}
        exec(compile(c.worker_source(ROOT),'generated224','exec'),ns)
        v=self.control();forwarded=shlex.split(shlex.join(['gm-run','entry','--g1-resume-input-b64',encode_control(v)]))
        self.assertEqual(ns['decode_control'](forwarded[-1]),v)
        out=Path(tempfile.mkdtemp(dir='/workspace',prefix='resume224-cli-'))/'output';events=[]
        argv=['entry','--task-id','TASK_20261026_999999','--expected-commit','a'*40,'--upload-url','https://storage.googleapis.com/x','--probe-url','https://storage.googleapis.com/y','--lowstep-input-b64',encode_control({}),'--g1-resume-input-b64',encode_control(v)]
        ns['task_output']=lambda _:out;ns['upload']=lambda *a:None;ns['emit']=lambda e,**kw:events.append((e,kw))
        with patch.object(sys,'argv',argv),patch.object(ns['signal'],'signal'),patch.object(ns['signal'],'alarm'),patch.object(ns['subprocess'],'check_output',side_effect=RuntimeError('CPU fixture stops before network/simulator')):
            self.assertEqual(ns['main'](),1)
        self.assertEqual(json.loads((out/'failure.json').read_text())['stage'],'source_preflight')
        self.assertEqual(events[0][1]['num_envs'],4096)
    def test_missing_or_wrong_resume_rejected_before_output(self):
        ns={'__file__':str(ROOT/'run_smoke.py'),'__name__':'fixture224'};exec(c.worker_source(ROOT),ns)
        argv=['entry','--task-id','TASK_20261026_999999','--expected-commit','a'*40,'--upload-url','x','--probe-url','y','--lowstep-input-b64','e30=','--g1-resume-input-b64',encode_control(self.control()|{'sha256':'0'*64})]
        with patch.object(sys,'argv',argv),patch.dict(ns,{'task_output':lambda _:self.fail('Must reject before output/GPU')}):
            with self.assertRaises(ValueError):ns['main']()

class ResumeAlgorithmFixtures(unittest.TestCase):
    def fixture(self,corruption=None):
        import torch,copy
        from flux.test_checkpoint_roundtrip import algorithm
        bounded=algorithm();native=bounded.__mro__[1]
        def learn(algo):
            for i in range(21):
                for optimizer in (algo.actor_optimizer,algo.critic_optimizer):
                    for _ in range(20):
                        for group in optimizer.param_groups:
                            for p in group['params']:p.grad=torch.ones_like(p)
                        optimizer.step();optimizer.zero_grad()
                algo.current_learning_iteration=841+i;algo._post_epoch_logging(841+i,{'loss':.1})
        native.learn=learn
        t=types.ModuleType('flux.teacher_algo');t.BoundedPPO=bounded;t._require_same_state=bounded.learn.__globals__['_require_same_state']
        with patch.dict(sys.modules,{'flux.teacher_algo':t}):
            g1=load_file('fixture_g1_lowstep',ROOT/'g1_lowstep_algo.py');resume=load_file('fixture_resume',ROOT/'resume_algo.py')
            with patch.dict(sys.modules,{'flux.g1_lowstep_algo':g1,'flux.resume_algo':resume}):mod=load_file('fixture_resume224',ROOT/'g1_resume224_algo.py')
        out=Path(tempfile.mkdtemp(dir='/workspace',prefix='resume224-algo-'));algo=mod.G1Resume224PPO(out)
        algo.env=types.SimpleNamespace(num_envs=4096);algo.device='cpu';algo.current_learning_iteration=841
        algo.empirical_normalization=False;algo.actor_obs_normalizer=algo.critic_obs_normalizer=torch.nn.Identity()
        for k,v in dict(num_steps_per_env=24,num_learning_epochs=5,num_mini_batches=4,num_learning_iterations=20020,save_interval=500).items():setattr(algo.config,k,v)
        for opt in (algo.actor_optimizer,algo.critic_optimizer):
            for row in opt.state.values():row['step'].fill_(16840)
        snap=resume.snapshot(algo);saved={key:(None if 'normalizer' in name else snap[name]) for name,key in resume.SOURCE_KEYS.items()};saved['iter']=841
        source=out/'source.pt';torch.save(saved,source);mod.CHECKPOINT_SHA256=hashlib.sha256(source.read_bytes()).hexdigest()
        if corruption=='actor':
            with torch.no_grad():algo.actor.weight.add_(1)
        elif corruption=='optimizer':next(iter(algo.actor_optimizer.state.values()))['exp_avg'].add_(1)
        elif corruption=='lr':algo.actor_learning_rate*=2
        elif corruption=='iter':algo.current_learning_iteration=0
        elif corruption=='no_optimizer':algo.config.load_optimizer=False
        with patch.dict(os.environ,{'PHP_FLUX_OUTPUT':str(out),'PHP_FLUX_RESUME_CHECKPOINT':str(source)}),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'mem_get_info',return_value=(1,2)),patch.object(torch.cuda,'max_memory_allocated',return_value=1):
            if corruption:
                with self.assertRaises(ValueError):algo.learn()
            else:algo.learn()
        return out
    def test_resume_advances_real_cpu_optimizers(self):
        out=self.fixture();r=json.loads((out/'resume_result.json').read_text())
        self.assertTrue(r['passed']);self.assertEqual(r['new_completed_updates'],21);self.assertEqual(r['cumulative_completed_updates'],863);self.assertEqual(r['native_final_iteration'],861)
        self.assertEqual(r['optimizer_step_advancement']['actor_optimizer']['min_step_increment'],420)
    def test_bad_restore_never_enters_gate(self):
        for bad in ['actor','optimizer','lr','iter','no_optimizer']:
            with self.subTest(bad=bad):
                out=self.fixture(bad);self.assertFalse((out/'resume_loaded.json').exists());self.assertFalse((out/'capacity_gate.json').exists())

if __name__=='__main__':unittest.main()
"""CPU longtrain contracts,not evidence of GPU execution or convergence."""
import ast,unittest
from pathlib import Path
from flux import longtrain_contract as c,longtrain_input as src,capacity_input,formal_input
ROOT=Path(__file__).resolve().parent

class LongtrainTests(unittest.TestCase):
    def scope(self):return {'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
        'gpu':'1x4090D-24G','num_envs':4096,'updates':20000,'outer_worker_seconds':3600,
        'worker_seconds':3300,'training_phase':'fullscale_4096_1h_segment',
        'code_commit':'a'*40,'billing_acceptance':'direct request','output_upload_probe_passed':True}
    def test_scope_accepts_one_segment(self):self.assertEqual(c.validate_scope(self.scope())['num_envs'],4096)
    def test_scope_rejects_resource_and_budget_drift(self):
        for k,v in [('num_envs',128),('updates',20),('trial_count',2),('automatic_retries',1),
                    ('outer_worker_seconds',7200),('worker_seconds',3421),('gpu_start_authorized',False),
                    ('output_upload_probe_passed',False)]:
            s=self.scope();s[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):c.validate_scope(s)
    def test_045_source_pinned(self):
        self.assertEqual(src.SOURCE_TASK,'TASK_20261025_045');self.assertEqual(src.SOURCE_ITERATION,2279)
        self.assertEqual(src.MAX_UPDATES,20000);self.assertEqual(src.ARCHIVE_BYTES,28676205)
    def test_older_contracts_unchanged(self):
        self.assertEqual(capacity_input.SOURCE_ITERATION,2260);self.assertEqual(capacity_input.MAX_UPDATES,20)
        self.assertEqual(formal_input.SOURCE_ITERATION,955);self.assertEqual(formal_input.MAX_UPDATES,10000)
    def test_reject_wrong_source(self):
        with self.assertRaises(ValueError):src.validate_control({'source_task':'TASK_20261025_044'})
    def test_teacher_target_and_save_recipe(self):
        t=c.teacher_source(ROOT);ast.parse(t)
        for token in ('num_envs=args.num_envs','args.updates != 20000','flux.longtrain_algo.LongtrainPPO','save_interval=500'):
            self.assertIn(token,t)
    def test_bootstrap_bounded_partial_native_reload(self):
        t=c.bootstrap_source(ROOT);ast.parse(t)
        for token in ("1 <= completed <= scope['updates']","native.get('native_load_passed') is True",
                      "resumed.get('passed') is True",'flux/longtrain_teacher.py',"'fullscale_segment': True",
                      "scope['worker_seconds'] - 120","scope['worker_seconds'] - 30"):
            self.assertIn(token,t)
        self.assertNotIn("completed == scope['updates']",t)
    def test_worker_deadlines_and_reserves(self):
        t=c.worker_source(ROOT);ast.parse(t)
        for token in ('signal.alarm(3480)','int(3598-(time.monotonic()-started))',
                      'remaining = min(3420, int(3420-(time.monotonic()-started)))',
                      "'outer_worker_seconds':3600",'worker_seconds=3600'):
            self.assertIn(token,t)
    def test_worker_uses4096_and_source045(self):
        t=c.worker_source(ROOT)
        for token in ('num_envs=4096',"'num_envs':4096",'updates = 20000',
                      "source_task='TASK_20261025_045'",'flux.longtrain_input',
                      'longtrain_worker_passed','longtrain_entrypoint_complete'):
            self.assertIn(token,t)
        for token in ('num_envs=128',"'num_envs':128",'capacity_worker_passed','updates = 20\n'):
            self.assertNotIn(token,t)
    def test_algo_preserves_native_guard_and_reports_resources(self):
        t=(ROOT/'longtrain_algo.py').read_text();ast.parse(t)
        for token in ('super().learn()','self.env.num_envs!=4096','num_learning_iterations!=20000',
                      'num_learning_epochs!=5','num_mini_batches!=4','resources.json',
                      'torch.cuda.mem_get_info','segment_after_native_reload'):
            self.assertIn(token,t)

if __name__=='__main__':unittest.main()
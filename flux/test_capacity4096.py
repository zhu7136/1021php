"""CPU contract checks; never evidence of4096GPU capacity."""
import ast,copy,json
from pathlib import Path
import unittest
from flux import capacity_contract as c,capacity_input as src,formal_input
ROOT=Path(__file__).resolve().parent

class CapacityTests(unittest.TestCase):
    def scope(self):return {'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
        'gpu':'1x4090D-24G','num_envs':4096,'updates':20,'outer_worker_seconds':1800,
        'worker_seconds':1600,'training_phase':'capacity_4096_validation','code_commit':'a'*40,
        'billing_acceptance':'direct user GPU request','output_upload_probe_passed':True}
    def test_scope_accepts_only_capacity(self):self.assertEqual(c.validate_scope(self.scope())['num_envs'],4096)
    def test_scope_rejects_old_scale(self):
        for key,value in [('num_envs',128),('updates',10000),('trial_count',2),('automatic_retries',1),('outer_worker_seconds',3600),('worker_seconds',1700),('output_upload_probe_passed',False)]:
            s=self.scope();s[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_scope(s)
    def test_source044pins_isolated(self):
        self.assertEqual(src.SOURCE_TASK,'TASK_20261025_044');self.assertEqual(src.SOURCE_ITERATION,2260)
        self.assertEqual(src.MAX_UPDATES,20);self.assertEqual(formal_input.SOURCE_ITERATION,955)
        self.assertEqual(formal_input.MAX_UPDATES,10000)
    def test_source_rejects033(self):
        with self.assertRaises(ValueError):src.validate_control({'source_task':'TASK_20261025_033'})
    def test_teacher_generated_consumes_4096_cli_and_20updates(self):
        text=c.teacher_source(ROOT);ast.parse(text)
        self.assertIn('num_envs=args.num_envs',text);self.assertIn('args.updates != 20',text)
        self.assertIn('flux.capacity_algo.CapacityPPO',text);self.assertIn('save_interval=5',text)
    def test_bootstrap_requires_twenty_and_capacity_report(self):
        text=c.bootstrap_source(ROOT);ast.parse(text)
        self.assertIn("completed == scope['updates']",text);self.assertIn("capacity.get('passed') is True",text)
        self.assertIn('flux/capacity_teacher.py',text);self.assertIn("'capacity_validation': True",text)
    def test_worker_no128_or10000_no_old_source(self):
        text=c.worker_source(ROOT);ast.parse(text)
        self.assertIn('num_envs=4096',text);self.assertIn("'num_envs':4096",text)
        self.assertIn('updates = 20',text);self.assertNotIn('10000',text)
        self.assertNotIn('num_envs=128',text);self.assertNotIn("'num_envs':128",text)
        self.assertIn("source_task='TASK_20261025_044'",text);self.assertIn('capacity_worker_passed',text)
    def test_runtime_memory_report_present(self):
        text=(ROOT/'capacity_algo.py').read_text();ast.parse(text)
        self.assertIn('torch.cuda.mem_get_info',text);self.assertIn('max_memory_allocated',text)
        self.assertIn('self._flux_completed_updates!=20',text);self.assertIn('self.env.num_envs!=4096',text)
    def test_fullscale_native_recipe_not_capacity_target(self):
        text=(ROOT/'fullscale_teacher.py').read_text();ast.parse(text)
        self.assertIn('num_envs=4096',text);self.assertIn('num_learning_iterations=20000',text)
        self.assertIn('save_interval=500',text);self.assertNotIn('CapacityPPO',text)
        self.assertIn('external_bounded_supervisor_verified',text)

if __name__=='__main__':unittest.main()
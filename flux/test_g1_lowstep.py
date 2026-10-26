import unittest,hashlib,json,tempfile,tarfile,io
from pathlib import Path
from flux import g1_lowstep_contract as c
from flux.g1_lowstep_input import fetch_data

class Contracts(unittest.TestCase):
    def scope(self):
        return dict(gpu_start_authorized=True,trial_count=1,automatic_retries=0,gpu='1x4090D-24G',
                    num_envs=4096,updates=20020,formal_seconds=1800,dataset='motions-low-step',fresh_training=True,
                    output_upload_probe_passed=True,code_commit='a'*40,billing_acceptance='approved',worker_seconds=3000)
    def test_valid(self):c.validate_scope(self.scope())
    def test_fail_closed(self):
        for k,v in [('num_envs',128),('automatic_retries',1),('fresh_training',False),('formal_seconds',3600),('dataset','motions-locomotion'),('trial_count',2),('output_upload_probe_passed',False),('worker_seconds',3301)]:
            with self.subTest(k=k),self.assertRaises(ValueError):c.validate_scope(self.scope()|{k:v})
    def test_generated(self):
        root=Path(__file__).parent
        for builder in (c.teacher_source,c.bootstrap_source,c.worker_source):
            source=builder(root);compile(source,'contract','exec')
        teacher=c.teacher_source(root);worker=c.worker_source(root)
        self.assertIn('flux.g1_lowstep_algo.G1LowstepPPO',teacher)
        self.assertIn('fetch_data',worker);self.assertNotIn("str(inputs/'motions-locomotion')",worker)
        self.assertIn('remaining = int(3060-',worker)
        self.assertIn("'formal_seconds':1800",worker)
        self.assertIn("signal.alarm(3180)",worker)
        self.assertIn("run_live(command,env,timeout,log_path,ROOT)",worker)
    def test_budget_gate(self):
        self.assertEqual(c.budget_times(1080,{}),(1050,1080))
        self.assertEqual(c.budget_times(1080,{'passed':False}),(1050,1080))
        self.assertEqual(c.budget_times(1080,{'passed':True,'formal_started_monotonic':500}),(2180,2300))
    def test_input_host_rejected(self):
        with self.assertRaises(ValueError):fetch_data({'url':'https://example.com/x','bytes':1,'sha256':'a'*64},'/workspace/not-created-g1-test')
    def test_input_extra_rejected(self):
        with self.assertRaises(ValueError):fetch_data({'url':'x','bytes':1,'sha256':'a'*64,'checkpoint':'x'},'/workspace/not-created-g1-test')

class NativeGateFixtures(unittest.TestCase):
    """CPU fixtures only; not evidence of real simulator/CUDA success."""
    def run_fixture(self,corruption=None,updates=21):
        import importlib.util,os,sys,types
        from unittest.mock import patch
        from flux.test_checkpoint_roundtrip import algorithm
        base=algorithm();native=base.__mro__[1]
        def fixture_learn(self):
            for i in range(updates):
                self.current_learning_iteration=i
                self._post_epoch_logging(i,{'loss':.1})
        native.learn=fixture_learn
        module=types.ModuleType('flux.teacher_algo');module.BoundedPPO=base
        module._require_same_state=base.learn.__globals__['_require_same_state']
        spec=importlib.util.spec_from_file_location('g1_gate_fixture',Path(__file__).with_name('g1_lowstep_algo.py'))
        loaded=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'flux.teacher_algo':module}):spec.loader.exec_module(loaded)
        out=Path(tempfile.mkdtemp(prefix='g1-gate-fixture-',dir='/workspace'))
        algo=loaded.G1LowstepPPO(out,corruption=corruption)
        algo.env=types.SimpleNamespace(num_envs=4096);algo.device='cpu'
        for k,v in dict(num_steps_per_env=24,num_learning_epochs=5,num_mini_batches=4,num_learning_iterations=20020,save_interval=500).items():setattr(algo.config,k,v)
        import torch
        with patch.dict(os.environ,{'PHP_FLUX_OUTPUT':str(out)}),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'mem_get_info',return_value=(1,2)),patch.object(torch.cuda,'max_memory_allocated',return_value=1):
            if corruption or updates<21:
                with self.assertRaises(ValueError):algo.learn()
            else:algo.learn()
        return out
    def test_gate_and_formal_fixture(self):
        out=self.run_fixture()
        self.assertTrue(json.loads((out/'capacity_gate.json').read_text())['passed'])
        self.assertEqual(json.loads((out/'formal_result.json').read_text())['formal_updates'],1)
    def test_corruption_blocks_formal(self):
        for item in ('actor','critic','actor_optimizer','iteration','learning_rate'):
            with self.subTest(item=item):
                out=self.run_fixture(corruption=item)
                self.assertFalse((out/'capacity_gate.json').exists())
                self.assertFalse((out/'formal_result.json').exists())
    def test_less_than_validation_blocks_formal(self):
        out=self.run_fixture(updates=19)
        self.assertFalse((out/'capacity_gate.json').exists())

if __name__=='__main__':unittest.main()
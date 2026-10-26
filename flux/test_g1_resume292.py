"""Actual CPU optimizer corruption fixtures for high-step292/60min."""
import unittest,json
from pathlib import Path
text=Path(__file__).with_name('test_g1_resume224.py').read_text()
for old,new in [('g1_resume224','g1_resume292'),('G1Resume224','G1Resume292'),
                ('g1_lowstep_algo','g1_resume292_gate'),('motions-low-step','motions-high-step'),
                ('--lowstep-input-b64','--highstep-input-b64'),('841','833'),('842','834'),
                ('16840','16680'),('863','855'),('861','853'),
                ("('formal_seconds',3600)","('formal_seconds',7200)"),
                ('formal_seconds=1800','formal_seconds=3600'),("'formal_seconds':1800","'formal_seconds':3600"),
                ('signal.alarm(3180)','signal.alarm(4980)')]:
    text=text.replace(old,new)
text=text.replace("if __name__=='__main__':unittest.main()",'')
exec(compile(text,__file__,'exec'),globals())

class HourBudget(unittest.TestCase):
    def test_total_learning_includes_gate(self):
        self.assertEqual(c.budget_times(1080,{}),(1050,1080))
        self.assertEqual(c.budget_times(1080,{'passed':True,'training_started_monotonic':500,'formal_started_monotonic':700}),(3980,4100))
    def test_missing_start_fails_closed(self):
        with self.assertRaises(KeyError):c.budget_times(1080,{'passed':True,'formal_started_monotonic':700})
    def test_generated_budgets(self):
        w=c.worker_source(ROOT);b=c.bootstrap_source(ROOT)
        for literal in ['signal.alarm(4980)','worker_seconds=5100','int(4860-','int(5098-',"'formal_seconds':3600"]:self.assertIn(literal,w)
        self.assertIn('from flux.g1_resume292_contract import budget_times',b)
        self.assertNotIn('motions-low-step',w)
        self.assertNotIn('g1_resume224',w)
    def test_scope_rejects_oversized_worker(self):
        s=ContinuationContracts().scope()
        c.validate_scope(s|{'worker_seconds':4800})
        for bad in [4981,True,179]:
            with self.assertRaises(ValueError):c.validate_scope(s|{'worker_seconds':bad})
    def test_exact_source_pin(self):
        self.assertEqual(inp.SOURCE_TASK,'TASK_20261026_292')
        self.assertEqual(inp.SOURCE_RUN,'00vxykeq')
        self.assertEqual(inp.SOURCE_ITERATION,833)
        self.assertEqual(inp.SOURCE_UPDATES,834)
        self.assertEqual(inp.CHECKPOINT_SHA256,'c254b82205e139ade4698a41f309849569712c069a4404d7003e53e08e1f756a')

if __name__=='__main__':unittest.main()
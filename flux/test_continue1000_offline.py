"""CPU checks for explicit offline selection; no GPU/start/network calls."""
import ast
from pathlib import Path
import unittest
from flux import continue1000_offline_contract as c
from flux.live_monitor import offline_environment
from flux import test_continue1000 as fixtures
ROOT=Path(__file__).resolve().parent

class OfflineTests(unittest.TestCase):
    def scope(self):
        return fixtures.ContinueTests().scope() | {'online_monitoring_required':False,'training_phase':'task048_continue1000_offline'}
    def test_scope(self):
        self.assertEqual(c.validate_scope(self.scope())['updates'],1000)
    def test_reject_scope_drift(self):
        for key,value in [('online_monitoring_required',True),('updates',2000),('worker_seconds',2521),('automatic_retries',1),('training_phase','task048_continue1000_online')]:
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_scope(self.scope()|{key:value})
    def test_no_key_required(self):
        self.assertEqual(offline_environment({})['WANDB_MODE'],'offline')
    def test_credentials_removed(self):
        keys=('WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY','GITHUB_TOKEN','github','key','WANDB_RUN_ID')
        env=offline_environment({k:'FAKE_TEST_ONLY' for k in keys})
        self.assertTrue(all(k not in env for k in keys))
    def test_teacher(self):
        t=c.teacher_source(ROOT);ast.parse(t)
        for token in ["mode='offline'","kw['mode'] = 'offline'",'flux.continue1000_offline_algo.OfflineContinue1000PPO','Continuation/completed_updates','native_offline_run_started','offline_environment']:
            self.assertIn(token,t)
        self.assertNotIn("mode='online'",t)
    def test_worker(self):
        t=c.worker_source(ROOT);ast.parse(t)
        for token in ["'online_monitoring_required':False",'run_live(command','signal.alarm(2580)',"source_task='TASK_20261025_048'",'updates = 1000','remaining = min(2520','flux/continue1000_offline_bootstrap.py','flux.continue1000_input',"WANDB_MODE='offline'"]:
            self.assertIn(token,t)
        self.assertNotIn('online_environment',t)
    def test_bootstrap(self):
        t=c.bootstrap_source(ROOT);ast.parse(t)
        self.assertIn('from flux.continue1000_offline_contract import validate_scope',t)
        self.assertIn("completed == scope['updates']",t)
        self.assertIn('flux/continue1000_offline_teacher.py',t)
    def test_algo(self):
        t=c.algo_source(ROOT);ast.parse(t)
        self.assertEqual(t.count("wandb.run.settings.mode != 'offline'"),2)
        self.assertIn('relative_checkpoint(completed)',t)
        self.assertIn('checkpoint_saved_locally',t)
        self.assertNotIn("policy='now'",t)

if __name__=='__main__':unittest.main()
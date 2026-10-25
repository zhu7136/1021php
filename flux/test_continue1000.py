"""CPU-only tests; no Flux submission, W&B run creation or GPU evidence."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from flux import continue1000_contract as c,continue1000_input as src,longtrain_input
from flux.live_monitor import online_environment,relative_checkpoint,run_live

ROOT=Path(__file__).resolve().parent

class ContinueTests(unittest.TestCase):
    def scope(self):
        return dict(gpu_start_authorized=True,trial_count=1,automatic_retries=0,
            gpu='1x4090D-24G',num_envs=4096,updates=1000,outer_worker_seconds=2700,
            worker_seconds=2500,training_phase='task048_continue1000_online',
            output_upload_probe_passed=True,online_monitoring_required=True,
            checkpoint_every_new_updates=500,code_commit='a'*40,billing_acceptance='test fixture only')
    def test_scope(self):
        self.assertEqual(c.validate_scope(self.scope())['updates'],1000)
    def test_reject_drift(self):
        for k,v in [('updates',20000),('checkpoint_every_new_updates',100),('online_monitoring_required',False),
                    ('num_envs',128),('trial_count',2),('automatic_retries',1),('worker_seconds',2521),
                    ('outer_worker_seconds',3600),('gpu_start_authorized',False)]:
            s=self.scope();s[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):c.validate_scope(s)
    def test_pinned_source(self):
        self.assertEqual((src.SOURCE_TASK,src.SOURCE_ITERATION,src.MAX_UPDATES),('TASK_20261025_048',3847,1000))
        self.assertEqual(longtrain_input.SOURCE_ITERATION,2279)
        with self.assertRaises(ValueError):src.validate_control({'source_task':'TASK_20261025_045'})
    def test_real_source_archive(self):
        archive=Path('/workspace/php-flux-smoke-20261025-060050/result-readback/php-smoke-result.zip')
        if not archive.exists():self.skipTest('Verified archive unavailable in this environment')
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            path=src.extract_checkpoint(archive,Path(d)/'input.pt')
            self.assertEqual(src.digest(path),src.CHECKPOINT_SHA256)
    def test_relative_save_cadence(self):
        self.assertEqual([x for x in range(1001) if relative_checkpoint(x)],[500,1000])
        self.assertEqual([3847+x-1 for x in (500,1000)],[4346,4846])
    def test_teacher_online_new_run(self):
        t=c.teacher_source(ROOT);ast.parse(t)
        for token in ['args.updates != 1000','flux.continue1000_algo.Continue1000PPO',
                      "mode='online'","resume='never'","kw.pop('id', None)",
                      'Continuation/completed_updates','checkpoint_every_new_updates=500']:
            self.assertIn(token,t)
        self.assertNotIn("mode='offline'",t)
    def test_bootstrap_exact_acceptance(self):
        t=c.bootstrap_source(ROOT);ast.parse(t)
        self.assertIn("completed == scope['updates']",t)
        self.assertIn('env = online_environment(env)',t)
        self.assertIn('flux/continue1000_teacher.py',t)
    def test_worker_tee_budget(self):
        t=c.worker_source(ROOT);ast.parse(t)
        for token in ['run_live(command','signal.alarm(2580)','int(2698-',
                      'remaining = min(2520',"source_task='TASK_20261025_048'",'updates = 1000',
                      'flux.continue1000_input','flux/continue1000_bootstrap.py']:
            self.assertIn(token,t)
        self.assertNotIn("WANDB_MODE='offline'",t)
    def test_missing_credentials_fail_closed(self):
        with self.assertRaises(ValueError):online_environment({})
    def test_online_env_removes_unrelated_secrets_and_old_runid(self):
        e=online_environment({'WANDB_API_KEY':'fake-only','FLUX_API_KEY':'fake-flux','WANDB_RUN_ID':'old','WANDB_MODE':'offline'})
        self.assertEqual(e['WANDB_MODE'],'online');self.assertEqual(e['WANDB_API_KEY'],'fake-only')
        self.assertNotIn('FLUX_API_KEY',e);self.assertNotIn('WANDB_RUN_ID',e)
    def run_fixture(self,code,timeout=3):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            lines=[];p=Path(d)/'live.log';env=dict(os.environ,WANDB_API_KEY='TEST_FAKE_SECRET')
            run_live([sys.executable,'-u','-c',code],env,timeout,p,cwd=d,
                     heartbeat_seconds=.05,sink=lambda text,**kw:lines.append(text))
            return lines,p.read_text()
    def test_tee_and_heartbeat(self):
        lines,disk=self.run_fixture("import time;print('progress 1/1000',flush=True);time.sleep(.2);print('done',flush=True)")
        self.assertIn('progress 1/1000',disk);self.assertIn('done',disk)
        self.assertTrue(any('training_heartbeat' in line for line in lines))
    def test_secret_url_redaction(self):
        lines,disk=self.run_fixture("print('TEST_FAKE_SECRET https://example.invalid/?token=FAKE')")
        self.assertNotIn('TEST_FAKE_SECRET',disk);self.assertNotIn('token=',disk)
        self.assertIn('[REDACTED]',disk)
    def test_redaction_across_chunk_boundary(self):
        _,disk=self.run_fixture("print('x'*8188+'TEST_FAKE_SECRET')")
        self.assertNotIn('TEST_FAKE_SECRET',disk);self.assertIn('[REDACTED]',disk)
    def test_oversized_line_suppressed(self):
        _,disk=self.run_fixture("print('x'*70000+'TEST_FAKE_SECRET')")
        self.assertIn('oversized console line suppressed',disk);self.assertNotIn('TEST_FAKE_SECRET',disk)
    def test_timeout(self):
        with self.assertRaises(TimeoutError):self.run_fixture('import time;time.sleep(10)',timeout=.2)
    def test_child_failure(self):
        with self.assertRaises(RuntimeError):self.run_fixture('raise SystemExit(2)')

if __name__=='__main__':unittest.main()
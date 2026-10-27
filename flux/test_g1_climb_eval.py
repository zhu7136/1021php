import unittest,os,ast
from pathlib import Path
from flux import g1_climb_eval_contract as c
class ClimbEvalTests(unittest.TestCase):
 def setUp(self):os.environ['PHP_G1_CLIMB_KIND']=os.environ.get('TEST_CLIMB_KIND','low')
 def protocol(self):
  g={'__file__':str(c.ROOT/'g1_climb_eval_protocol.py'),'__name__':'isolated'}
  exec(compile(c.protocol_source(),'test_protocol','exec'),g);return g
 def scope(self,p):
  return dict(gpu_start_authorized=True,trial_count=1,automatic_retries=0,gpu='1x4090D-24G',worker_seconds_max=600,training_updates=0,source_run=p['SOURCE_RUN'],checkpoint_sha256=p['CHECKPOINT_SHA256'],dataset='motions-'+c.kind()+'-climb-76',pairs=40,terrain_variants=64,repeats=3,seeds=[42,43,44],output_upload_probe_passed=True,code_commit='a'*40)
 def test_valid_scope(self):
  p=self.protocol();self.assertEqual(p['validate_scope'](self.scope(p))['worker_seconds_max'],600)
 def test_wrong_scope_rejected(self):
  p=self.protocol()
  for key,val in [('training_updates',1),('worker_seconds_max',601),('worker_seconds_max',True),('automatic_retries',1),('trial_count',2),('pairs',56),('source_run','9fe07g0l'),('dataset','motions-low-step')]:
   s=self.scope(p);s[key]=val
   with self.assertRaises(ValueError):p['validate_scope'](s)
 def test_summary_counts(self):
  p=self.protocol();rows=[dict(clip_index=i,seed=s,status='not_tested') for s in p['SEEDS'] for i in range(40)]
  rows[0]['status']='completed';rows[1]['status']='failed';rows[2]['status']='truncated'
  r=p['summarize'](rows);self.assertEqual(r['planned_trials'],120);self.assertEqual(r['not_tested'],117);self.assertEqual(r['completion_rate_resolved_only'],.5)
 def test_failure_and_timeout_not_success(self):
  p=self.protocol();self.assertEqual(p['classify'](True,True,True),'failed');self.assertEqual(p['classify'](False,True,True),'truncated')
 def test_teacher_static_safety(self):
  t=c.teacher_source();tree=ast.parse(t)
  self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('learn','backward') for n in ast.walk(tree)))
  for anchor in ['algo.learn=forbidden','algo.actor_optimizer.step=forbidden','algo.critic_optimizer.step=forbidden','_require_same_state(before,after','SOURCE_ITERATION','len(ends)!=2560','env.num_envs!=40','episode_seconds=max(12.0,max(lengths)/50.0+1.0)']:
   self.assertIn(anchor,t)
 def test_worker_deadlines(self):
  t=c.worker_source();ast.parse(t)
  for anchor in ['signal.alarm(540)','540-(time.monotonic()-started)','595-(time.monotonic()-started)','str(started+480)',"'climb-input-b64'"]:
   self.assertIn(anchor,t)
  self.assertNotIn('g1_eval283_teacher.py',t)
 def test_checkpoint_identity(self):
  p=self.protocol();self.assertEqual((p['SOURCE_RUN'],p['SOURCE_ITERATION'],p['CHECKPOINT_SHA256']),c.SOURCES[c.kind()][1:])
 def test_parallel_transport_is_eval_only_and_checkpoint_validated(self):
  t=c.worker_source();self.assertIn('from flux.g1_climb_eval_data import static_checks,fetch_data',t)
  t=c.input_source();ast.parse(t)
  self.assertIn('validate_control(control)',t);self.assertIn('from flux.parallel_download import download',t)
  self.assertIn("download(control['url'],path,CHECKPOINT_BYTES,CHECKPOINT_SHA256",t)
  data=(c.ROOT/'g1_climb_eval_data.py').read_text()
  self.assertIn("text.count(old)!=1",data)
  # Shared training source still uses the original sequential verified transport.
  self.assertIn('from flux.verified_download import download',(c.ROOT/'g1_climb_contract.py').read_text())
 def test_missing_kind_rejected(self):
  os.environ.pop('PHP_G1_CLIMB_KIND')
  with self.assertRaises(ValueError):c.kind()
if __name__=='__main__':unittest.main()
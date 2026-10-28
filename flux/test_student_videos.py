import ast,copy,unittest
from pathlib import Path
from flux.student_video_contract import *
class ContractTests(unittest.TestCase):
 def scope(self):return dict(gpu_start_authorized=True,user_approval_utc=APPROVAL,trial_count=1,automatic_retries=0,
  automatic_continuation=False,worker_seconds_max=600,gpu='1x4090D24GB',training_updates=0,
  source_task='TASK_20261027_216',source_checkpoint_sha256=SOURCE_SHA,student_only=True,num_envs=1,
  seconds_per_skill=10,terrain_variants=3,terrain_seed=42,skills=list(SKILLS),code_commit='a'*40)
 def test_approved(self):self.assertEqual(validate_scope(self.scope())['training_updates'],0)
 def test_reject_expanded_scope(self):
  for key,value in [('training_updates',1),('student_only',False),('worker_seconds_max',601),('trial_count',2),
    ('automatic_retries',1),('automatic_continuation',True),('source_task','TASK_20261027_269'),('num_envs',80),('seconds_per_skill',11)]:
   with self.subTest(key=key),self.assertRaises(ValueError):
    s=self.scope();s[key]=value;validate_scope(s)
 def test_boolean_not_int(self):
  with self.assertRaises(ValueError):s=self.scope();s['training_updates']=False;validate_scope(s)
 def test_clip_indices(self):
  self.assertEqual(sum(COUNTS),296);self.assertEqual(tuple(sum(COUNTS[:i]) for i in range(5)),SELECTED)
 def test_no_training_calls(self):
  tree=ast.parse((Path(__file__).parent/'student_video_inference.py').read_text())
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
    self.assertNotIn(n.func.attr,('learn','backward','load_teacher','teacher_act'))
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):self.assertNotIn(n.func.id,('train','PPO','FiveSkillDistillation'))
 def test_provenance_frozen_source(self):
  text=(Path(__file__).parent/'student_video_inference.py').read_text()
  self.assertIn('policy.load_state_dict(state,strict=True)',text)
  self.assertIn('policy.requires_grad_(False)',text);self.assertIn('policy.teacher_act=forbidden',text)
  self.assertIn('torch.equal(v.cpu(),state[k])',text);self.assertIn("digest(plan['student'])==SOURCE_SHA",text)
 def test_native_failure_not_bypassed(self):
  text=(Path(__file__).parent/'student_video_inference.py').read_text()
  self.assertIn("namespace['refresh_relative_targets'](cmd);original()",text)
  self.assertIn("native_failure_steps",text);self.assertIn('manual_stop=recorder.stop_recording',text)
 def test_worker_limits_and_sealing(self):
  text=(Path(__file__).parent/'run_student_videos.py').read_text()
  self.assertIn('598-(time.monotonic()-started)',text);self.assertIn("'.mp4'",text)
  self.assertIn("'soft_deadline':started+520",text);self.assertNotIn("'teachers-b64'",text)
if __name__=='__main__':unittest.main()
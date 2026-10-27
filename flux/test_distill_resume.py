"""CPU-only execution of native restore and the actual continuation update gate."""
import ast,copy,json,logging,os,unittest
from pathlib import Path
from unittest.mock import patch
from typing import cast,Any
import torch
from flux import distill_resume_contract as c
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/workspace/g1_distill_task216_result/student_final.pt')
HOLO=Path('/workspace/php_g1/thirdparty/holosoma/src/holosoma/holosoma/agents/distillation/distillation.py')
def function(path,name,classname=None):
 nodes=ast.parse(path.read_text()).body
 if classname:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==classname).body
 return next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
class Policy:
 def __init__(self):self.data={};self.teachers=self
 def load_training_state_dict(self,x):self.data=copy.deepcopy(x)
 def set_loaded_teachers(self,x):self.loaded_teachers=x
 def project_action_std(self):pass
 def eval(self):pass
 def parameters(self):return []
 def state_dict(self):return self.data
class Optimizer:
 def load_state_dict(self,x):self.data=copy.deepcopy(x)
 def state_dict(self):return self.data
class Stop(BaseException):pass
class Base:
 def __init__(self):self.policy=Policy();self.optimizer=Optimizer();self.device='cpu';self.current_learning_iteration=0;self.learning_rate=0
 def _restore_env_state(self,x):pass
 def _post_epoch_logging(self,it,loss_dict):self._completed+=1
def build():
 env={'torch':torch,'logger':logging.getLogger('cpu-resume'),'cast':cast,'Any':Any}
 n=function(HOLO,'load','Distillation');exec(compile(ast.Module(body=[n],type_ignores=[]),'native_load','exec'),env);Base.load=env['load']
 n=function(ROOT/'flux/teacher_algo.py','_require_same_state');exec(compile(ast.Module(body=[n],type_ignores=[]),'native_compare','exec'),env)
 env.update(FiveSkillDistillation=Base,StopAfterUpdate=Stop,cpu_state=lambda m:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},
  SOURCE_SHA=c.SOURCE_SHA,SOURCE_ITER=c.SOURCE_ITER,digest=c.digest,stop_required=c.stop_required,os=os,time=__import__('time'),Path=Path,json=json)
 n=next(n for n in ast.parse((ROOT/'flux/five_distill_resume_algo.py').read_text()).body if isinstance(n,ast.ClassDef))
 exec(compile(ast.Module(body=[n],type_ignores=[]),'actual_resume_class','exec'),env);return env['FiveSkillResume']
class Tests(unittest.TestCase):
 def test_source_bytes_and_metadata(self):
  self.assertEqual(SOURCE.stat().st_size,c.SOURCE_BYTES);self.assertEqual(c.digest(SOURCE),c.SOURCE_SHA)
  ck=torch.load(SOURCE,map_location='cpu',weights_only=True);self.assertEqual(ck['iter'],976);self.assertEqual(ck['teacher_loaded'],[True]*5)
 def test_actual_native_restore(self):
  a=build()();a.load(str(SOURCE));self.assertTrue(a._resume_loaded);self.assertEqual(a.current_learning_iteration,977)
  self.assertEqual(a.learning_rate,0.0003);self.assertEqual({float(v['step']) for v in a.optimizer.state_dict()['state'].values()},{15632.0})
 def test_optimizer_failure_rejected(self):
  a=build()()
  def bad(x):a.optimizer.data={'bad':1};raise ValueError('Fixture optimizer restore failed')
  a.optimizer.load_state_dict=bad
  with self.assertRaises(ValueError):a.load(str(SOURCE))
 def test_actual_update_gate(self):
  a=build()();a._completed=0
  with patch.dict(os.environ,PHP_DISTILL_SOFT_DEADLINE=str(__import__('time').monotonic()+100)):
   with self.assertRaises(Stop):
    for i in range(500):a._post_epoch_logging(i,{})
  self.assertEqual(a._completed,400)
 def test_actual_time_gate(self):
  a=build()();a._completed=0
  with patch.dict(os.environ,PHP_DISTILL_SOFT_DEADLINE='0'),self.assertRaises(Stop):a._post_epoch_logging(977,{})
  self.assertEqual(a._completed,1)
 def test_scope_rejects_old_1800s(self):
  s=json.loads(Path('/workspace/g1_distill_diagnostic_20261027_085037/state.json').read_text())['scope']
  with self.assertRaises(ValueError):c.validate_scope(s)
 def test_entrypoint_finite_config_and_checkpoint(self):
  s=(ROOT/'flux/five_distill_resume_student.py').read_text();self.assertIn("checkpoint=plan['student']",s);self.assertIn('num_learning_iterations=400',s);self.assertNotIn('100000',s)
 def test_total_worker_deadlines(self):
  s=(ROOT/'flux/run_five_distill_resume.py').read_text()
  for text in ['598-','signal.alarm(550)','595-','time.monotonic()-started>360']:self.assertIn(text,s)
 def test_no_credential_control_in_plan(self):
  s=(ROOT/'flux/run_five_distill_resume.py').read_text();nodes=ast.parse(s)
  writes=[ast.get_source_segment(s,n) for n in ast.walk(nodes) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text']
  self.assertTrue(all('controls' not in x and 'upload_url' not in x for x in writes))
 def test_compile(self):
  for p in (ROOT/'flux').glob('*resume*.py'):compile(p.read_text(),str(p),'exec')
if __name__=='__main__':unittest.main()
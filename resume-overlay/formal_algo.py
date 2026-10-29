"""UNVALIDATED native formal DAgger adapter; imported only on approved runner.

Not a smoke subclass: the smoke five-update guard remains untouched.
"""
import functools,json,os,time
from pathlib import Path
import torch
from holosoma.agents.distillation.distillation import Distillation
from holosoma.agents.modules.module_utils import setup_depth_student_teacher_module
from wbt_training.utils.student_teacher import routing_column_index
from .native_algo import StrictRoutedTeacher,cpu_state,require_equal
from .masked_training import ExplicitMaskTrainingMixin
from .smoke_config import check_live_observations
from .contract import load_verified
from .formal_core import FormalLoopMixin,FormalBudget,continuation_contract,contract_digest,load_continuation,save_continuation

class FormalSoftStop(TimeoutError):
 pass

class HuD04FormalDistillation(FormalLoopMixin,ExplicitMaskTrainingMixin,Distillation):
 def _rollout_step(self,observations):
  if self.update_budget.completed>0 and time.monotonic()>=self.update_budget.deadline-10:
   raise FormalSoftStop('Reserve final checkpoint and cleanup time')
  return super()._rollout_step(observations)
 def _resolve_obs_dims(self):
  super()._resolve_obs_dims();self.live_contract=check_live_observations(self.env.observation_manager)
  if (self.actor_obs_dim,self.teacher_obs_dim,self.num_act,self.depth_shape)!=(96,450,29,(58,87)):raise ValueError('Formal observation/action contract changed')
 def _build_policy(self):
  self.policy=setup_depth_student_teacher_module(num_actor_obs=self.actor_obs_dim,num_teacher_obs=self.teacher_obs_dim,
   num_actions=self.num_act,module_config=self.config.module,device=self.device,init_noise_std=self.config.init_noise_std,
   num_teachers=self.num_teachers,module_cls=functools.partial(StrictRoutedTeacher,routing_column=routing_column_index(self.env)))
 def load_teacher(self,paths):
  paths=[paths] if isinstance(paths,str) else paths
  checked=load_verified(paths);del checked
  return super().load_teacher(paths)
 def _formal_plan(self):return json.loads(Path(os.environ['HUD04_FORMAL_PLAN']).read_text())
 def load(self,path):
  if path is None:return None
  self.formal_next_iteration=load_continuation(path,policy=self.policy,optimizer=self.optimizer,contract=continuation_contract(self._formal_plan()))
  self.current_learning_iteration=self.formal_next_iteration
  self.learning_rate=self.optimizer.param_groups[0]['lr']
  self.policy.set_loaded_teachers([True]*5);self.policy.teachers.eval()
  for parameter in self.policy.teachers.parameters():parameter.requires_grad_(False)
  if 'HUD04_FORMAL_OUTPUT' in os.environ:
   from .formal_worker import sha,write_json
   write_json(Path(os.environ['HUD04_FORMAL_OUTPUT'])/'formal-resume-verification.json',
    {'checkpoint_sha256':sha(path),'start_completed_updates':self.formal_next_iteration,
     'strict_model_restore':True,'optimizer_restore_and_readback':True,'rng_restore':True,
     'trajectory_exact_resume':False,'_wb_agent':self._formal_plan()['_wb_agent']})
  return None
 def save(self,path=None,name='formal_last.pt'):
  path=path or str(Path(self.log_dir)/name)
  plan=self._formal_plan();count=getattr(self,'formal_next_iteration',self.current_learning_iteration)
  save_continuation(path,policy=self.policy,optimizer=self.optimizer,next_iteration=count,
   contract=continuation_contract(plan),provenance=plan['_wb_agent'])
 def learn(self,num_learning_iterations=None):
  plan=self._formal_plan();updates=plan['additional_updates']
  if num_learning_iterations is not None and num_learning_iterations!=updates:raise ValueError('Additional-update scope mismatch')
  if self.is_multi_gpu or self.env.num_envs!=plan['num_envs'] or self.num_teachers!=5:raise ValueError('Unapproved GPU/env scope')
  if hasattr(self,'update_budget'):raise RuntimeError('No automatic repeated learn call')
  self.update_budget=FormalBudget(updates,float(os.environ['HUD04_FORMAL_TRAIN_DEADLINE']))
  self.formal_next_iteration=self.current_learning_iteration
  self.policy.require_loaded_teachers()
  expected={id(p) for m in (self.policy.student,self.policy.depth_backbone) for p in m.parameters()}
  actual={id(p) for group in self.optimizer.param_groups for p in group['params']}
  if actual!=expected or any(p.requires_grad for p in self.policy.teachers.parameters()):raise ValueError('Optimizer/freeze contract')
  teacher_before=cpu_state(self.policy.teachers)
  student_before=cpu_state(self.policy.student);depth_before=cpu_state(self.policy.depth_backbone)
  stop_reason='max_updates'
  try:Distillation.learn(self,num_learning_iterations=updates)
  except FormalSoftStop:
   stop_reason='worker_soft_deadline'
  require_equal(teacher_before,cpu_state(self.policy.teachers))
  if any(p.grad is not None for p in self.policy.teachers.parameters()):raise RuntimeError('Teacher gradient leaked')
  if not 1<=self.update_budget.completed<=updates or (self.update_budget.completed!=updates and stop_reason!='worker_soft_deadline'):raise ValueError('Incomplete formal training')
  student_changed=any(not torch.equal(value,student_before[key]) for key,value in cpu_state(self.policy.student).items())
  depth_changed=any(not torch.equal(value,depth_before[key]) for key,value in cpu_state(self.policy.depth_backbone).items())
  if not student_changed or not depth_changed:raise ValueError('Student/depth parameters did not change')
  self.current_learning_iteration=self.formal_next_iteration
  checkpoint=Path(self.log_dir)/'formal_final.pt';self.save(str(checkpoint))
  # Training evidence is distinct from native process/lifecycle success. Parent
  # requires BOTH this record and true returncode=0 after interpreter teardown.
  if 'HUD04_FORMAL_OUTPUT' in os.environ:
   from .formal_worker import sha,write_json
   out=Path(os.environ['HUD04_FORMAL_OUTPUT']).resolve()
   if not checkpoint.resolve().is_relative_to(out):raise ValueError('Checkpoint outside approved output')
   saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
   if saved['completed_updates']!=self.formal_next_iteration:raise ValueError('Final checkpoint update count mismatch')
   require_equal(cpu_state(self.policy),saved['model_state_dict'])
   write_json(out/'formal-native-result.json',{'additional_updates_completed':self.update_budget.completed,
    'completed_updates':self.formal_next_iteration,'checkpoint':str(checkpoint.resolve().relative_to(out)),
    'checkpoint_sha256':sha(checkpoint),'continuation_contract_sha256':contract_digest(continuation_contract(plan)),
    '_wb_agent':plan['_wb_agent'],'teachers_frozen_verified':True,'checkpoint_model_readback_verified':True,
    'student_changed':student_changed,'depth_backbone_changed':depth_changed,
    'stop_reason':stop_reason,'requested_additional_updates':updates,
    'skills_evaluated':False,'native_process_success_not_yet_known':True})
 def _post_epoch_logging(self,it,loss_dict):
  super()._post_epoch_logging(it,loss_dict)
  record={'completed_updates':self.formal_next_iteration,'additional_updates_completed':self.update_budget.completed,
   'losses':loss_dict,'skills_evaluated':False,'monotonic_time':time.monotonic()}
  with (Path(self.log_dir)/'formal_progress.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
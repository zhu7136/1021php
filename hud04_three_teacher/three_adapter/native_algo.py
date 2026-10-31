"""UNVALIDATED native three-teacher pure DAgger; NO GPU execution approved.

Import only in the pinned worker runtime. Native checkpoint save/load is used;
no G1 preset, critic, PPO updates, ONNX export, automatic retry or resume.
"""
import copy,json,os,time
from pathlib import Path
import torch
from holosoma.agents.distillation.distillation import Distillation
from holosoma.agents.modules.module_utils import setup_depth_student_teacher_module
from .routing import StrictRoutedTeacher,routing_column_index
from .contract import BoundedUpdatesMixin,UpdateBudget,SKILLS,load_verified,routed_targets,checkpoint_contract,validate_student_checkpoint
from .masked_training import ExplicitMaskTrainingMixin,routing_validity
from .smoke_config import check_live_observations

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def require_equal(a,b):
 if isinstance(a,torch.Tensor):
  if not isinstance(b,torch.Tensor) or not torch.equal(a.detach().cpu(),b.detach().cpu()):raise RuntimeError('Checkpoint tensor mismatch')
 elif isinstance(a,dict):
  if set(a)!=set(b):raise RuntimeError('Checkpoint keys mismatch')
  for k,v in a.items():require_equal(v,b[k])
 elif isinstance(a,(tuple,list)):
  if type(a)!=type(b) or len(a)!=len(b):raise RuntimeError('Checkpoint sequence mismatch')
  for x,y in zip(a,b):require_equal(x,y)
 elif a!=b:raise RuntimeError('Checkpoint value mismatch')

class HuD04SmokeDistillation(BoundedUpdatesMixin,ExplicitMaskTrainingMixin,Distillation):
 def _checkpoint_metadata(self,iteration=None):
  metadata=super()._checkpoint_metadata(iteration=iteration)
  provenance={'thread_id':os.environ['HUD04_ARIA_THREAD_ID'],'turn_id':os.environ['HUD04_ARIA_TURN_ID']}
  if not all(provenance.values()):raise ValueError('Missing checkpoint provenance')
  metadata.update(_wb_agent=provenance,record_role='bounded_three_teacher_smoke_not_skill_evaluation',three_teacher_contract=checkpoint_contract())
  return metadata
 def _resolve_obs_dims(self):
  super()._resolve_obs_dims()
  self.live_contract=check_live_observations(self.env.observation_manager)
  if self.actor_obs_dim!=96 or self.teacher_obs_dim!=450 or self.depth_shape!=(58,87) or self.num_act!=29:
   raise ValueError('Native observation/action dimensions changed')
 def _build_policy(self):
  import functools
  self.policy=setup_depth_student_teacher_module(num_actor_obs=self.actor_obs_dim,num_teacher_obs=self.teacher_obs_dim,
   num_actions=self.num_act,module_config=self.config.module,device=self.device,init_noise_std=self.config.init_noise_std,
   num_teachers=self.num_teachers,module_cls=functools.partial(StrictRoutedTeacher,routing_column=routing_column_index(self.env)))
 def load_teacher(self,paths):
  if isinstance(paths,str):paths=[paths]
  # CPU authenticate all three identities, observation/controller semantics and
  # readiness BEFORE the native loader sends any weights to the policy device.
  verified=load_verified(paths);del verified
  return super().load_teacher(paths)
 def load(self,path):
  if path is not None:validate_student_checkpoint(torch.load(path,map_location='cpu',weights_only=True))
  return super().load(path)
 def save(self,path=None,name='last.ckpt'):
  if any(not torch.isfinite(t).all() for t in self.policy.state_dict().values()):raise FloatingPointError('Nonfinite checkpoint')
  return Distillation.save(self,path=path,name=name)
 def learn(self,num_learning_iterations=None):
  if os.environ.get('HUD04_THREE_GPU_EXECUTION_APPROVED')!='1':raise PermissionError('CPU preparation only; new GPU approval and bounded worker required')
  if hasattr(self,'update_budget'):raise RuntimeError('Smoke cannot be resumed or invoked twice')
  deadline=float(os.environ['HUD04_TRAIN_DEADLINE']);remaining=deadline-time.monotonic()
  if not 0<remaining<=600:raise ValueError('Invalid remaining worker budget')
  self.update_budget=UpdateBudget(seconds=remaining);self.update_budget.deadline=deadline
  if self.is_multi_gpu or self.env.num_envs!=6 or self.num_teachers!=3 or self.current_learning_iteration!=0:
   raise ValueError('Not a fresh single-GPU six-environment three-teacher smoke')
  if (self.config.num_learning_iterations,self.config.num_steps_per_env,self.config.num_learning_epochs,self.config.num_mini_batches,self.config.distillation_warmup_steps)!=(5,24,2,2,1):
   raise ValueError('Smoke scope changed')
  self.policy.require_loaded_teachers()
  allowed={id(p) for m in (self.policy.student,self.policy.depth_backbone) for p in m.parameters()}
  optimized={id(p) for g in self.optimizer.param_groups for p in g['params']}
  if optimized!=allowed or any(p.requires_grad for p in self.policy.teachers.parameters()):raise RuntimeError('Optimizer/freeze contract failed')
  teacher_before=cpu_state(self.policy.teachers);student_before=cpu_state(self.policy.student);depth_before=cpu_state(self.policy.depth_backbone)
  self._seen=[0]*3;self._valid=[0]*3
  original=self.policy.teacher_act
  @torch.no_grad()
  def measured(obs,teacher_idx=None):
   cmd=self.env.command_manager.get_state('motion_command');base=cmd.motion.motion_idxs[cmd.time_steps].long().flatten()
   route=obs[:,self.policy.routing_column];valid=routing_validity(obs,self.policy.routing_column)
   if ((base<0)|(base>2)).any() or not torch.equal(route[valid].long(),base[valid]):raise ValueError('Underlying expert route disagrees with observation')
   actions=original(obs,teacher_idx)
   for i in range(3):self._seen[i]+=int((base==i).sum());self._valid[i]+=int(((base==i)&valid).sum())
   return actions
  self.policy.teacher_act=measured
  try:super().learn(num_learning_iterations)
  finally:self.policy.teacher_act=original
  require_equal(teacher_before,cpu_state(self.policy.teachers))
  if any(p.grad is not None or p.requires_grad for p in self.policy.teachers.parameters()):raise RuntimeError('Teacher changed or received gradients')
  student_changed=any(not torch.equal(v,self.policy.student.state_dict()[k].cpu()) for k,v in student_before.items())
  depth_changed=any(not torch.equal(v,self.policy.depth_backbone.state_dict()[k].cpu()) for k,v in depth_before.items())
  if not student_changed or not depth_changed or not all(self._valid):raise RuntimeError('Missing student/depth learning or three-skill labels')
  final=Path(self.log_dir)/'student_smoke_final.pt';self.save(str(final))
  policy_state=cpu_state(self.policy);optimizer=copy.deepcopy(self.optimizer.state_dict());self.load(str(final))
  require_equal(policy_state,cpu_state(self.policy));require_equal(optimizer,self.optimizer.state_dict())
  import hashlib
  report={'pipeline_passed':True,'completed_updates':self.update_budget.completed,'teachers_frozen_verified':True,
   'student_changed':student_changed,'depth_backbone_changed':depth_changed,'native_reload_verified':True,
   'checkpoint':str(final.relative_to(Path(os.environ['HUD04_SMOKE_OUTPUT']))),'checkpoint_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
   'live_contract':self.live_contract,'skill_counts':self.skill_counts(),'skill_success_verified':False,
   'scope':'plumbing smoke only; no skill/convergence claim'}
  (Path(os.environ['HUD04_SMOKE_OUTPUT'])/'distillation-result.json').write_text(json.dumps(report,indent=2))
 def skill_counts(self):return {s:{'seen':self._seen[i],'valid':self._valid[i]} for i,s in enumerate(SKILLS)}
 def _post_epoch_logging(self,it,loss_dict):
  super()._post_epoch_logging(it,loss_dict)
  report={'iteration':int(it),'completed_updates':self.update_budget.completed,'losses':loss_dict,'skill_counts':self.skill_counts()}
  (Path(os.environ['HUD04_SMOKE_OUTPUT'])/'progress.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({'event':'hud04_student_smoke_progress',**report}),flush=True)
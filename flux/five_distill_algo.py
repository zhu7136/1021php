"""Masked pure DAgger; no PPO/critic training, frozen teachers, bounded saving."""
import copy,json,math,os,time
from pathlib import Path
import torch
from holosoma.agents.distillation.distillation import Distillation
from wbt_training.utils.student_teacher import PhpDistillation,routing_column_index
from flux.five_distill_contract import SKILLS,masked_loss
from flux.teacher_algo import _require_same_state

class StopAfterUpdate(BaseException):pass

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

class FiveSkillDistillation(PhpDistillation):
 def save(self,path=None,name='last.ckpt'):
  # Native checkpoint is required. ONNX export is not needed for this short pilot.
  for k,v in self.policy.state_dict().items():
   if torch.is_tensor(v) and not torch.isfinite(v).all():raise FloatingPointError('Nonfinite parameter '+k)
  return Distillation.save(self,path=path,name=name)

 def learn(self,num_learning_iterations=None):
  out=Path(os.environ['PHP_FLUX_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
  assert self.num_teachers==5 and self.env.num_envs==80 and torch.cuda.is_available()
  assert self.config.num_steps_per_env==24 and self.config.num_mini_batches==8 and self.config.num_learning_epochs==2
  self.policy.require_loaded_teachers()
  teacher_ids={id(p) for p in self.policy.teachers.parameters()}
  optimized={id(p) for g in self.optimizer.param_groups for p in g['params']}
  allowed={id(p) for m in (self.policy.student,self.policy.depth_backbone) for p in m.parameters()}
  assert not teacher_ids&optimized and optimized==allowed
  assert all(not p.requires_grad for p in self.policy.teachers.parameters())
  before_teachers=cpu_state(self.policy.teachers)
  before_student=cpu_state(self.policy.student);before_depth=cpu_state(self.policy.depth_backbone)
  self._completed=0;self._seen=torch.zeros(5,device=self.device);self._valid=torch.zeros_like(self._seen);self._residual=torch.zeros_like(self._seen)
  self.distill_loss_fn=masked_loss
  original=self.policy.teacher_act
  @torch.no_grad()
  def measured_teacher_act(obs,teacher_idx=None):
   actions=original(obs,teacher_idx)
   cmd=self.env.command_manager.get_state('motion_command')
   route=cmd.motion.motion_idxs[cmd.time_steps].long().flatten()
   if (route<0).any() or (route>=5).any():raise ValueError('Invalid underlying teacher route')
   valid=~torch.all(actions==0,dim=-1)
   residual=(self.policy.action_mean-actions).square().mean(-1)
   for i in range(5):
    mask=route==i;keep=mask&valid
    self._seen[i]+=mask.sum();self._valid[i]+=keep.sum();self._residual[i]+=(residual*keep).sum()
   return actions
  self.policy.teacher_act=measured_teacher_act
  print(json.dumps({'event':'five_distill_learning_started','teachers':5,'student_count':1,'num_envs':80,'teacher_frozen':True,'mode':'masked_pure_dagger'}),flush=True)
  try:super().learn(num_learning_iterations)
  except StopAfterUpdate:pass
  finally:self.policy.teacher_act=original
  assert self._completed>0
  _require_same_state(before_teachers,cpu_state(self.policy.teachers),'frozen_teachers')
  assert all(p.grad is None and not p.requires_grad for p in self.policy.teachers.parameters())
  student_changed=any(not torch.equal(v,cpu_state(self.policy.student)[k]) for k,v in before_student.items())
  depth_changed=any(not torch.equal(v,cpu_state(self.policy.depth_backbone)[k]) for k,v in before_depth.items())
  if not student_changed or not depth_changed:raise ValueError('Student or depth backbone did not learn')
  final=Path(self.log_dir)/'student_final.pt';self.save(str(final))
  expected=cpu_state(self.policy);expected_optimizer=copy.deepcopy(self.optimizer.state_dict())
  self.load(str(final))
  _require_same_state(expected,cpu_state(self.policy),'native_policy_reload')
  _require_same_state(expected_optimizer,self.optimizer.state_dict(),'native_optimizer_reload')
  report={'pipeline_passed':True,'completed_updates':self._completed,'teachers_frozen_verified':True,
   'student_changed':student_changed,'depth_backbone_changed':depth_changed,'native_reload_verified':True,
   'checkpoint':str(final.relative_to(out)),'skill_counts':self.skill_metrics(),'skill_success_verified':False,
   'scope':'short distillation pilot; not skill success or convergence evaluation'}
  (out/'distillation-result.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({'event':'five_distill_native_checkpoint_verified',**report}),flush=True)

 def skill_metrics(self):
  return {s:{'seen':int(self._seen[i]),'valid':int(self._valid[i]),
   'valid_fraction':float(self._valid[i]/self._seen[i].clamp_min(1)),
   'valid_action_mse':float(self._residual[i]/self._valid[i]) if self._valid[i]>0 else None} for i,s in enumerate(SKILLS)}

 def _post_epoch_logging(self,it,loss_dict):
  if not all(math.isfinite(float(v)) for v in loss_dict.values()):raise FloatingPointError('Nonfinite loss')
  super()._post_epoch_logging(it,loss_dict)
  self._completed+=1
  skills=self.skill_metrics()
  report={'completed_updates':self._completed,'iteration':int(it),'losses':loss_dict,'skill_counts':skills}
  out=Path(os.environ['PHP_FLUX_OUTPUT'])
  (out/'progress.json').write_text(json.dumps(report,indent=2))
  import wandb
  if wandb.run:
   wandb.log({f'distill/{s}/{k}':v for s,d in skills.items() for k,v in d.items() if v is not None})
  print(json.dumps({'event':'five_distill_progress',**report}),flush=True)
  if self._completed==2:
   # Same process/tiny-scale learning gate, not a second GPU job.
   if not all(d['valid']>0 for d in skills.values()):raise ValueError('Shakeout lacks labels for one or more skills')
   self.save(str(Path(self.log_dir)/'student_shakeout.pt'))
  if time.monotonic()>=float(os.environ['PHP_DISTILL_SOFT_DEADLINE']):raise StopAfterUpdate()

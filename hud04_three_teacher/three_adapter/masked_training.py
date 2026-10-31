"""Explicit teacher-route masking integrated with native DAgger minibatches.

No new storage field is necessary: native storage already preserves the
teacher observations, including the resolved routing column, through shuffle.
"""
import time
import torch
from .contract import explicit_masked_mse,BudgetExceeded,SKILLS

def routing_validity(teacher_observation,routing_column):
 if teacher_observation.ndim!=2 or teacher_observation.shape[1]!=450 or not 0<=routing_column<450:
  raise ValueError('Expected 449 privileged values plus one resolved routing column')
 route=teacher_observation[:,routing_column]
 if not torch.isfinite(route).all() or not torch.equal(route,route.round()) or ((route < -1)|(route>=len(SKILLS))).any():
  raise ValueError('Invalid teacher routing observation')
 return route>=0

class ExplicitMaskTrainingMixin:
 def _training_step(self):
  if self.is_multi_gpu:raise NotImplementedError('This authorized smoke is single-GPU only')
  accum=0.;updates=0;skipped=0;valid_rows=0;total_rows=0
  for batch in self.storage.mini_batch_generator(self.config.num_mini_batches,self.config.num_learning_epochs):
   if time.monotonic()>=self.update_budget.deadline:raise BudgetExceeded('Deadline before optimizer minibatch')
   valid=routing_validity(batch['teacher_obs'],self.policy.routing_column)
   total_rows+=len(valid);valid_rows+=int(valid.sum())
   self.optimizer.zero_grad(set_to_none=True)
   # Skip entirely invalid minibatches, including Adam momentum updates.
   if not valid.any():skipped+=1;continue
   prediction=self.policy.student(self.policy._student_input(batch['actor_obs'],batch['depth_obs']))
   loss=explicit_masked_mse(prediction,batch['teacher_actions'],valid)
   loss.backward()
   parameters=[p for group in self.optimizer.param_groups for p in group['params']]
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
    raise FloatingPointError('Nonfinite student/depth gradient')
   if any(p.grad is not None for p in self.policy.teachers.parameters()):raise RuntimeError('Teacher received gradients')
   torch.nn.utils.clip_grad_norm_(parameters,self.config.max_grad_norm,error_if_nonfinite=True)
   self.optimizer.step();accum+=float(loss.detach());updates+=1
  self.storage.clear()
  if updates==0:raise RuntimeError('No valid teacher labels in this training iteration')
  return {'behavior':accum/updates,'warmup_active':float(self._warmup_active),
   'valid_fraction':valid_rows/max(total_rows,1),'optimizer_minibatches':float(updates),
   'skipped_all_invalid_minibatches':float(skipped)}
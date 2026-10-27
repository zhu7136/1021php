"""Verified native policy/optimizer continuation, finite total-update and time bounds."""
import json,os,time
from pathlib import Path
import torch
from flux.five_distill_algo import FiveSkillDistillation,StopAfterUpdate,cpu_state
from flux.teacher_algo import _require_same_state
from flux.distill_resume_contract import SOURCE_SHA,SOURCE_ITER,digest,stop_required

class FiveSkillResume(FiveSkillDistillation):
 def load(self,path):
  initial=not getattr(self,'_resume_loaded',False)
  if path is None:raise ValueError('Continuation requires native checkpoint')
  if initial and digest(path)!=SOURCE_SHA:raise ValueError('Wrong source student')
  ck=torch.load(path,map_location='cpu',weights_only=True)
  if ck['num_teachers']!=5 or ck['teacher_loaded']!=[True]*5:raise ValueError('Unverified source teachers')
  result=super().load(path)
  _require_same_state(ck['model_state_dict'],cpu_state(self.policy),'resume_policy')
  _require_same_state(ck['optimizer_state_dict'],self.optimizer.state_dict(),'resume_optimizer')
  if self.learning_rate!=ck['learning_rate'] or self.current_learning_iteration!=ck['iter']:raise ValueError('Resume scalar mismatch')
  if initial:
   if ck['iter']!=SOURCE_ITER:raise ValueError('Wrong source iteration')
   self._source_state=ck
   self.current_learning_iteration=SOURCE_ITER+1
   self._resume_loaded=True
  return result

 def learn(self,num_learning_iterations=None):
  if not getattr(self,'_resume_loaded',False) or self.current_learning_iteration!=SOURCE_ITER+1:raise ValueError('Fresh start is forbidden')
  _require_same_state(self._source_state['model_state_dict'],cpu_state(self.policy),'source_before_first_update')
  _require_same_state(self._source_state['optimizer_state_dict'],self.optimizer.state_dict(),'optimizer_before_first_update')
  print(json.dumps({'event':'five_distill_resume_verified','source_iteration':SOURCE_ITER,'next_iteration':SOURCE_ITER+1,
   'source_sha256':SOURCE_SHA,'optimizer_loaded_verified':True,'learning_rate':self.learning_rate,'warmup_restarted':False}),flush=True)
  del self._source_state
  super().learn(num_learning_iterations)
  if not 0<self._completed<=400:raise ValueError('Total continuation updates exceeded bound')
  p=Path(os.environ['PHP_FLUX_OUTPUT'])/'distillation-result.json';r=json.loads(p.read_text())
  r.update(source_task='TASK_20261027_216',source_iteration=SOURCE_ITER,source_checkpoint_sha256=SOURCE_SHA,
   resume_policy_optimizer_verified=True,training_updates_max=400,worker_seconds_max=600,warmup_restarted=False,
   last_iteration=self.current_learning_iteration)
  p.write_text(json.dumps(r,indent=2))
  print(json.dumps({'event':'five_distill_resume_complete',**r}),flush=True)

 def _post_epoch_logging(self,it,loss_dict):
  super()._post_epoch_logging(it,loss_dict)
  if stop_required(self._completed,time.monotonic(),float(os.environ['PHP_DISTILL_SOFT_DEADLINE'])):raise StopAfterUpdate()
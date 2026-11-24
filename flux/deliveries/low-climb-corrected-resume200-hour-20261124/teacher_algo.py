"""Exact state resume, actual PPO wall-clock3600s; no200-update stop."""
import hashlib,json,math,os,time
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO,_StopAfterUpdate
from learning_algo import snapshot
from speed10_audit import install_native_audit
from resume_audit import verify_loaded_state,SOURCE_UPDATES
TRAIN_SECONDS=3600
class FullTeacherPPO(BoundedPPO):
 def load(self,path):
  result=super().load(path)
  if path is not None and Path(path).resolve()==Path(os.environ['PHP_RESUME_CHECKPOINT']).resolve():self._resume_evidence=verify_loaded_state(self,path,os.environ['PHP_FLUX_OUTPUT'])
  return result
 def _post_epoch_logging(self,it,loss_dict):
  out=Path(os.environ['PHP_FLUX_OUTPUT'])
  if time.monotonic()-self._started>=TRAIN_SECONDS:
   (out/'STOP_AFTER_UPDATE').touch(exist_ok=True);self._stop_reason='training_duration_reached'
  stop=False
  try:super()._post_epoch_logging(it,loss_dict)
  except _StopAfterUpdate:stop=True
  row=dict(update=self._flux_completed_updates,iteration=int(it),num_envs=int(self.env.num_envs),elapsed_seconds=time.monotonic()-self._started,transitions=self._flux_completed_updates*4096*24,source_completed_updates=SOURCE_UPDATES,cumulative_completed_updates=SOURCE_UPDATES+self._flux_completed_updates,**{'Loss/'+k:float(v) for k,v in loss_dict.items()})
  assert int(it)==SOURCE_UPDATES+self._flux_completed_updates-1
  if not all(math.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite update metric')
  with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  print(json.dumps(dict(event='formal_update',**row)),flush=True)
  if self._flux_completed_updates==1 or self._flux_completed_updates%100==0:self.save(str(Path(self.log_dir)/'model_latest.pt'))
  if self._flux_completed_updates==1 or self._flux_completed_updates%100==0 or stop:self._speed10_flush(self._flux_completed_updates)
  if stop:raise _StopAfterUpdate()
 def learn(self):
  from probe_audit import verify_native_reference
  verify_native_reference(self.env)
  out=Path(os.environ['PHP_FLUX_OUTPUT'])
  assert self.env.num_envs==4096 and self.config.num_steps_per_env==24
  assert self._resume_evidence['exact_training_state_loaded'] and self.current_learning_iteration==SOURCE_UPDATES-1
  self.current_learning_iteration=SOURCE_UPDATES
  self._speed10_flush=install_native_audit(self.env,out)
  before,old=snapshot(self)
  self._started=time.monotonic();self._stop_reason='not_yet_stopped'
  timer=dict(event='training_timer_started',target_seconds=TRAIN_SECONDS,source_completed_updates=SOURCE_UPDATES,maximum_additional_updates=100000,stop_at_update_boundary=True)
  (out/'training-timer.json').write_text(json.dumps(timer));print(json.dumps(timer),flush=True)
  capacity=dict(event='capacity_initialized',effective_num_envs=int(self.env.num_envs),samples_per_update=4096*24,gpu=torch.cuda.get_device_name(0),cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory)
  (out/'capacity.json').write_text(json.dumps(capacity));print(json.dumps(capacity),flush=True)
  result=super().learn()
  after,new=snapshot(self)
  record=dict(completed_updates=self._flux_completed_updates,actor_changed=any(not torch.equal(v,after['actor'][k]) for k,v in before['actor'].items()),critic_changed=any(not torch.equal(v,after['critic'][k]) for k,v in before['critic'].items()),actor_optimizer_steps_increased=new['actor_optimizer']>old['actor_optimizer'],critic_optimizer_steps_increased=new['critic_optimizer']>old['critic_optimizer'],parameters_finite=all(torch.isfinite(v).all().item() for a in after.values() for v in a.values()),teacher_converged=False,fresh_training=False,new_process_resume_verified=True,source_completed_updates=SOURCE_UPDATES,cumulative_completed_updates=SOURCE_UPDATES+self._flux_completed_updates,source_task='TASK_20261123_410',stop_reason=self._stop_reason,training_elapsed_seconds=time.monotonic()-self._started,training_target_seconds=TRAIN_SECONDS)
  checkpoint=Path(self.log_dir)/'model_smoke_final.pt';record.update(checkpoint_relative_path=str(checkpoint.relative_to(out)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
  (out/'learning_state.json').write_text(json.dumps(record,indent=2))
  assert 1<=record['completed_updates']<=100000
  assert all(record[k] for k in ['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite'])
  return result
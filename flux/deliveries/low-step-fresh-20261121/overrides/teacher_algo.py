"""Native PPO, full-skill inputs, finite-update evidence and bounded stop."""
import hashlib,json,math,os,time
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO,_StopAfterUpdate
from learning_algo import snapshot
class FullTeacherPPO(BoundedPPO):
 def _post_epoch_logging(self,it,loss_dict):
  stop=False
  try:super()._post_epoch_logging(it,loss_dict)
  except _StopAfterUpdate:stop=True
  out=Path(os.environ['PHP_FLUX_OUTPUT'])
  row={'update':self._flux_completed_updates,'iteration':int(it),'num_envs':int(self.env.num_envs),'elapsed_seconds':time.monotonic()-self._started,'transitions':self._flux_completed_updates*4096*24,**{'Loss/'+k:float(v) for k,v in loss_dict.items()}}
  if not all(math.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite update metric')
  with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  print(json.dumps({'event':'formal_update',**row}),flush=True)
  if self._flux_completed_updates==1 or self._flux_completed_updates%100==0:self.save(str(Path(self.log_dir)/'model_latest.pt'))
  if stop:raise _StopAfterUpdate()
 def learn(self):
  out=Path(os.environ['PHP_FLUX_OUTPUT']);self._started=time.monotonic()
  assert self.env.num_envs==4096 and self.config.num_steps_per_env==24
  before,old=snapshot(self)
  capacity={'event':'capacity_initialized','effective_num_envs':int(self.env.num_envs),'samples_per_update':4096*24,'gpu':torch.cuda.get_device_name(0),'cuda_total_bytes':torch.cuda.get_device_properties(0).total_memory}
  (out/'capacity.json').write_text(json.dumps(capacity));print(json.dumps(capacity),flush=True)
  result=super().learn()
  after,new=snapshot(self)
  record={'completed_updates':self._flux_completed_updates,'actor_changed':any(not torch.equal(v,after['actor'][k]) for k,v in before['actor'].items()),'critic_changed':any(not torch.equal(v,after['critic'][k]) for k,v in before['critic'].items()),'actor_optimizer_steps_increased':new['actor_optimizer']>old['actor_optimizer'],'critic_optimizer_steps_increased':new['critic_optimizer']>old['critic_optimizer'],'parameters_finite':all(torch.isfinite(v).all().item() for a in after.values() for v in a.values()),'teacher_converged':False,'fresh_training':True,'new_process_resume_verified':False}
  checkpoint=Path(self.log_dir)/'model_smoke_final.pt'
  record.update(checkpoint_relative_path=str(checkpoint.relative_to(out)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
  (out/'learning_state.json').write_text(json.dumps(record,indent=2))
  required=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite']
  if not 1<=record['completed_updates']<=2000 or not all(record[k] for k in required):raise ValueError('Native learning evidence failed')
  return result
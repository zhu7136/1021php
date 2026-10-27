"""Bounded real PPO continuation;20-update native checkpoint gate,1000 total ceiling."""
import json,os
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO,_require_same_state
from flux.resume_algo import snapshot,finite,require_loaded_state,require_updated
SOURCE_SHA='cb98b0762fbf3b478211bcb9131d53d0330c3459fa7dba04c7874205c1875399'
def digest(p):
 import hashlib
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def roundtrip(algo,path):
 algo.save(str(path));before=snapshot(algo);algo.load(str(path));_require_same_state(before,snapshot(algo),'native_roundtrip')
 if not path.is_file() or not path.stat().st_size:raise ValueError('empty native checkpoint')
 return dict(path=str(path.relative_to(Path(os.environ['PHP_FLUX_OUTPUT']))),bytes=path.stat().st_size,sha256=digest(path),native_load_passed=True)
class WalkTenPPO(BoundedPPO):
 def learn(self):
  source=Path(os.environ['PHP_FLUX_RESUME_CHECKPOINT']);out=Path(os.environ['PHP_FLUX_OUTPUT'])
  if digest(source)!=SOURCE_SHA or self.current_learning_iteration!=4846 or self.config.num_learning_iterations!=1000 or self.config.load_optimizer is not True or self.empirical_normalization is not False:raise ValueError('unapproved resume source or PPO scope')
  saved=torch.load(source,map_location='cpu',weights_only=True);self._walk_before=snapshot(self);finite(self._walk_before);require_loaded_state(self,saved,self._walk_before)
  for name,key in [('actor_learning_rate','actor_optimizer_state_dict'),('critic_learning_rate','critic_optimizer_state_dict')]:_require_same_state(saved[key]['param_groups'][0]['lr'],self._walk_before[name],name)
  (out/'resume-loaded.json').write_text(json.dumps(dict(source_sha256=SOURCE_SHA,source_iter=4846,native_full_state_matches=True,normalization_enabled=False)))
  self._walk_gate_passed=False
  result=super().learn();after=snapshot(self);finite(after);changes,steps=require_updated(self._walk_before,after)
  if not 1<=self._flux_completed_updates<=1000 or self.current_learning_iteration!=4846+self._flux_completed_updates-1 or digest(source)!=SOURCE_SHA:raise ValueError('update count/index/source mismatch')
  receipt=dict(completed_updates=self._flux_completed_updates,gate_passed=self._walk_gate_passed,actor_critic_changed=changes,optimizer_step_advancement=steps,source_unchanged=True,native_final_iteration=self.current_learning_iteration,teacher_converged=False,exact_environment_rng_resume_claimed=False)
  (out/'learning-result.json').write_text(json.dumps(receipt,indent=2));print('WALK_RESULT '+json.dumps(receipt),flush=True)
  return result
 def _post_epoch_logging(self,it,loss_dict):
  if self._flux_completed_updates>=1000:raise ValueError('total PPO update ceiling reached')
  super()._post_epoch_logging(it,loss_dict)
  out=Path(os.environ['PHP_FLUX_OUTPUT']);finite(snapshot(self))
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(completed_updates=self._flux_completed_updates,native_iteration=int(it),losses={k:float(v) for k,v in loss_dict.items()}))+'\n');f.flush()
  if self._flux_completed_updates==20:
   changes,steps=require_updated(self._walk_before,snapshot(self))
   if any(v['min_step_increment']!=400 or v['max_step_increment']!=400 for v in steps.values()):raise ValueError('20-update gate optimizer step count mismatch')
   gate=roundtrip(self,Path(self.log_dir)/'model_gate20.pt');self._walk_gate_passed=True
   (out/'gate20.json').write_text(json.dumps(dict(passed=True,completed_updates=20,parameters_changed=changes,optimizer_steps=steps,checkpoint=gate),indent=2))
   print('WALK_GATE '+json.dumps(dict(passed=True,completed_updates=20,checkpoint=gate)),flush=True)
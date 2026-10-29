"""Verified native resume and bounded, noise-free pre/post rollout diagnostics."""
import copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from fresh_algo import FullTeacherPPO as FreshPPO
from flux.teacher_algo import _require_same_state

def write(name,value):
 out=Path(os.environ['PHP_FLUX_OUTPUT']);(out/name).write_text(json.dumps(value,indent=2))

class FullTeacherPPO(FreshPPO):
 def load(self,path):
  result=super().load(path)
  if Path(path).name=='continuation-source.pt':
   expected=torch.load(path,map_location='cpu',weights_only=True)
   assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==os.environ['CONTINUATION_SOURCE_SHA']
   for name,key in [('actor','actor_model_state_dict'),('critic','critic_model_state_dict'),('actor_optimizer','actor_optimizer_state_dict'),('critic_optimizer','critic_optimizer_state_dict')]:
    _require_same_state(expected[key],getattr(self,name).state_dict(),name)
   for name in ['actor_obs_normalizer','critic_obs_normalizer']:
    state=expected.get(name+'_state_dict')
    if state is not None:_require_same_state(state,getattr(self,name).state_dict(),name)
   assert self.current_learning_iteration==expected['iter']
   assert self.actor_learning_rate==expected['actor_optimizer_state_dict']['param_groups'][0]['lr']
   assert self.critic_learning_rate==expected['critic_optimizer_state_dict']['param_groups'][0]['lr']
   self._source_iteration=expected['iter'];self._resume_verified=True
   self.current_learning_iteration=self._source_iteration+1
   write('new_process_resume.json',{'actor_critic_optimizers_exact':True,'normalization_checked_when_present':True,'source_iteration':self._source_iteration,'first_new_iteration':self.current_learning_iteration,'source_checkpoint_sha256':os.environ['CONTINUATION_SOURCE_SHA'],'native_new_process_load_verified':True,'scope':'model/optimizer continuation; source has no serialized simulator/RNG state, not bitwise trajectory continuation'})
  return result

 @torch.no_grad()
 def rollout(self,phase):
  started=time.monotonic();seed=20261028;steps=240
  rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
  env_state=copy.deepcopy(self._collect_env_state());evaluating=self.env.is_evaluating
  original_actor=copy.deepcopy(self.actor.state_dict())
  old_normalizers={n:copy.deepcopy(getattr(self,n).state_dict()) for n in ['actor_obs_normalizer','critic_obs_normalizer']}
  self._eval_mode();self.env.set_is_evaluating()
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  policy=self.get_inference_policy(device=self.device)
  total_reward=0.;resets=0;timeouts=0;finite=True;contact_sum=0.;contact_frames=0
  try:
   obs=self.env.reset_all()
   for _ in range(steps):
    actions=policy(obs)
    if not torch.isfinite(actions).all():raise FloatingPointError('Nonfinite deterministic action')
    obs,rewards,dones,info=self.env.step({'actions':actions})
    if not torch.isfinite(rewards).all():raise FloatingPointError('Nonfinite deterministic reward')
    total_reward+=float(rewards.sum());resets+=int(dones.sum())
    timeout=self.env.time_out_buf
    timeouts+=int((dones.bool() & timeout.bool()).sum())
   _require_same_state(original_actor,self.actor.state_dict(),'evaluation_must_not_change_actor')
   for n,v in old_normalizers.items():_require_same_state(v,getattr(self,n).state_dict(),'evaluation_'+n)
   report={'phase':phase,'seed':seed,'steps_per_env':steps,'num_envs':int(self.env.num_envs),'noise_free_actor_mean':True,'training_updates':0,'mean_reward_per_step':total_reward/(steps*self.env.num_envs),'reset_events':resets,'timeout_reset_events':timeouts,'non_timeout_reset_events':resets-timeouts,'all_actions_rewards_finite':finite,'policy_and_normalizers_unchanged':True,'elapsed_seconds':time.monotonic()-started,'scope':'short fixed-seed rollout diagnostic, not complete-clip success/fall certification; physics and randomization otherwise unchanged','video_recorded':False}
   write('evaluation-'+phase+'.json',report);print(json.dumps({'event':'deterministic_evaluation',**report}),flush=True)
   return report
  finally:
   self.env.is_evaluating=evaluating
   self._restore_env_state(env_state)
   random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3]);self._train_mode()

 def learn(self):
  assert getattr(self,'_resume_verified',False)
  assert not (Path(os.environ['PHP_FLUX_OUTPUT'])/'STOP_AFTER_UPDATE').exists()
  before=self.rollout('before')
  if (Path(os.environ['PHP_FLUX_OUTPUT'])/'STOP_AFTER_UPDATE').exists():raise TimeoutError('No continuation budget after evaluation')
  result=super().learn()
  after=self.rollout('after')
  p=Path(os.environ['PHP_FLUX_OUTPUT'])/'learning_state.json';record=json.loads(p.read_text())
  record.update(fresh_training=False,new_process_resume_verified=True,source_iteration=self._source_iteration,final_iteration=int(self.current_learning_iteration),fixed_seed_rollout_before_after=True,rollout_quality_certified=False)
  p.write_text(json.dumps(record,indent=2))
  return result
"""Exact actor/critic/optimizer/scalar resume verification; no random-init fallback."""
import hashlib,json
from pathlib import Path
import torch
from flux.teacher_algo import _require_same_state
SOURCE_SHA='9a0b61cdb689886c1171f77eb725ff85f4910573d7fa6376fdf4594415e45711'
SOURCE_UPDATES=661

def verify_loaded_state(algo,path,out):
 path=Path(path)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA, 'resume_checkpoint_identity_changed'
 d=torch.load(path,map_location='cpu',weights_only=False)
 assert d['iter']==SOURCE_UPDATES-1 and algo.current_learning_iteration==SOURCE_UPDATES-1
 assert algo.config.load_optimizer and d['experiment_config']['training']['num_envs']==4096
 names=d['experiment_config']['robot']['dof_names'];limits=d['experiment_config']['robot']['dof_vel_limit_list']
 effective=json.loads((Path(__file__).parent/'config.json').read_text())
 for group in ['robot','observation','action','reward','termination']:
  expected=json.dumps(d['experiment_config'][group],sort_keys=True).replace('/workspace/isaaclab/1021php/private-TASK_20261029_018/payload/bundle','$BUNDLE')
  assert expected==json.dumps(effective[group],sort_keys=True), 'source_contract_changed_'+group
 checked=[];normalizers=[]
 for name in ['actor','critic']:
  _require_same_state(d[name+'_model_state_dict'],getattr(algo,name).state_dict(),name);checked.append(name)
 for name in ['actor','critic']:
  key=name+'_obs_normalizer_state_dict'
  if d[key] is not None:
   assert algo.empirical_normalization
   _require_same_state(d[key],getattr(algo,name+'_obs_normalizer').state_dict(),key);normalizers.append(key)
  else:assert not algo.empirical_normalization
 steps={}
 for name in ['actor','critic']:
  key=name+'_optimizer_state_dict';opt=getattr(algo,name+'_optimizer')
  _require_same_state(d[key],opt.state_dict(),key);checked.append(name+'_optimizer')
  values=[float(v['step']) for v in opt.state_dict()['state'].values()]
  assert values and min(values)==max(values)==13220.
  assert getattr(algo,name+'_learning_rate')==d[key]['param_groups'][0]['lr']
  steps[name]=min(values)
 report=dict(event='resume_state_verified',source_task='TASK_20261029_018',source_checkpoint_sha256=SOURCE_SHA,source_completed_updates=SOURCE_UPDATES,source_iteration=SOURCE_UPDATES-1,next_iteration=SOURCE_UPDATES,checked_modules_and_optimizers=checked,checked_normalizers=normalizers,normalizers_disabled_in_source=not algo.empirical_normalization,optimizer_steps=steps,learning_rates_restored=True,exact_training_state_loaded=True,trajectory_exact_resume=False)
 (Path(out)/'resume-state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 return report
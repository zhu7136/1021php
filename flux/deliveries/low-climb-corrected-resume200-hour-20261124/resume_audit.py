"""Full actor/critic/optimizer/scalar identity check. Fail closed, no fresh fallback."""
import hashlib,json
from pathlib import Path
import torch
from flux.teacher_algo import _require_same_state
SOURCE_SHA='370df740c687140ec8b87de7ae70bb1f9d3f2b88ed8983a0ab9f38e8fb0bc09b'
SOURCE_UPDATES=200
def canonical(value):
 if isinstance(value,dict):return {k:canonical(v) for k,v in value.items() if not(k=='params' and (v is None or v=={}))}
 if isinstance(value,list):return [canonical(v) for v in value]
 return value
def verify_loaded_state(algo,path,out):
 path=Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA
 d=torch.load(path,map_location='cpu',weights_only=True)
 assert d['iter']==SOURCE_UPDATES-1 and algo.current_learning_iteration==SOURCE_UPDATES-1 and algo.config.load_optimizer
 effective=json.loads((Path(__file__).parent/'config.json').read_text())
 for group in ['robot','observation','action','reward','termination','command','randomization']:
  expected=json.dumps(canonical(d['experiment_config'][group]),sort_keys=True).replace('/workspace/corrected-prefix500-TASK_20261123_410/payload/bundle','$BUNDLE')
  assert expected==json.dumps(canonical(effective[group]),sort_keys=True),'source_contract_changed_'+group
 checked=[];steps={}
 for name in ['actor','critic']:
  _require_same_state(d[name+'_model_state_dict'],getattr(algo,name).state_dict(),name);checked.append(name)
  key=name+'_obs_normalizer_state_dict'
  if d[key] is not None:_require_same_state(d[key],getattr(algo,name+'_obs_normalizer').state_dict(),key)
  else:assert not algo.empirical_normalization
  key=name+'_optimizer_state_dict';opt=getattr(algo,name+'_optimizer');_require_same_state(d[key],opt.state_dict(),key);checked.append(name+'_optimizer')
  values=[float(v['step']) for v in opt.state_dict()['state'].values()];assert values and min(values)==max(values)==4000.
  assert getattr(algo,name+'_learning_rate')==d[key]['param_groups'][0]['lr'];steps[name]=min(values)
 report=dict(event='resume_state_verified',source_task='TASK_20261123_410',source_checkpoint_sha256=SOURCE_SHA,source_completed_updates=SOURCE_UPDATES,source_iteration=SOURCE_UPDATES-1,next_iteration=SOURCE_UPDATES,checked_modules_and_optimizers=checked,optimizer_steps=steps,learning_rates_restored=True,exact_training_state_loaded=True,trajectory_exact_resume=False)
 (Path(out)/'resume-state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return report
"""Fail closed on any native corrected-reference mismatch."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
def verify_native_reference(env):
 command=env.command_manager._state_terms['motion_command'];path=Path(command.motion_cfg.motion_file)
 manifest=json.loads((Path(__file__).parent/'training-reference-manifest.json').read_text());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==manifest['training_reference_sha256']
 motion=command.motion
 with np.load(path,allow_pickle=False) as data:
  expected={'_joint_pos':data['joint_pos'][:,7:],'_joint_vel':data['joint_vel'][:,6:],'_body_pos_w':data['body_pos_w'],'_body_quat_w':data['body_quat_w'][:,:,[1,2,3,0]],'_body_lin_vel_w':data['body_lin_vel_w'],'_body_ang_vel_w':data['body_ang_vel_w'],'motion_ends':data['motion_ends'],'motion_idxs':data['motion_idxs']}
  for name,values in expected.items():
   actual=getattr(motion,name).detach().cpu();wanted=torch.as_tensor(values,dtype=actual.dtype);assert actual.shape==wanted.shape and torch.equal(actual,wanted),'native_reference_tensor_mismatch:'+name
 assert motion.time_step_total==500 and int(motion.num_motions)==1
 report=dict(event='native_corrected_reference_verified',sha256=digest,frames=500,clips=1,all_loader_tensors_exact=True,fresh_training=False,historical_checkpoint_loaded=True,source_task='TASK_20261123_410')
 (Path(os.environ['PHP_FLUX_OUTPUT'])/'native-corrected-reference.json').write_text(json.dumps(report,indent=2))
"""CPU-testable Hu-D04 three-teacher contract; independent of five-skill state."""
import copy
import hashlib
import time
from pathlib import Path
import torch

SKILLS=('locomotion','low-step','high-step')
HASHES=(
 '75b3c06421cf0f9cd3abdb79c6591be33a79a7dc0f294bae683b40ff6acb86fe',
 '91c19b21fa2680fa40dd40d49f66f70aad9ee49e13ebc12e62c6151cb7ca566e',
 'd8aac2800e777c3a10bb20f8e30ff8d5c8fda83d2d2517cbb20aae716ac8cc07',
)
SCHEMA='hud04-three-teacher-dagger-v1'
ITERATIONS=(1361,1272,1373)
HEAD=('head_yaw_joint','head_pitch_joint')

def semantic_robot(robot):
 result=copy.deepcopy(robot)
 # Only task-local asset relocation is ignored. Controller gains, names,
 # collision settings, geometry filenames and all other values must agree.
 for key in ('asset_root','mesh_root','urdf_file','usd_file'):
  value=result['asset'][key]
  result['asset'][key]=Path(value).name
 return result

def load_verified(paths):
 if len(paths)!=len(SKILLS) or len(set(map(str,paths)))!=len(SKILLS): raise ValueError('Exactly three ordered distinct teachers are required')
 configs=[];checkpoints=[]
 for index,path in enumerate(paths):
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=HASHES[index]:
   raise ValueError('Teacher identity/order mismatch: '+SKILLS[index])
  ckpt=torch.load(path,map_location='cpu',weights_only=True)
  if ckpt.get('iter')!=ITERATIONS[index]:raise ValueError('Wrong teacher iteration')
  cfg=ckpt['experiment_config'];robot=cfg['robot'];algo=cfg['algo']['config']
  if len(robot['dof_names'])!=31 or robot['actions_dim']!=29 or not set(HEAD)<=set(robot['dof_names']):
   raise ValueError('Wrong physical/policy joint contract')
  actor=algo['module_dict']['actor']['layer_config']
  if actor['activation']!='ELU' or actor['hidden_dims']!=[512,256,128]:
   raise ValueError('Unsupported teacher MLP')
  if algo['empirical_normalization'] or ckpt['actor_obs_normalizer_state_dict'] is not None:
   raise ValueError('Unexpected observation normalization')
  for tensor in ckpt['actor_model_state_dict'].values():
   if not torch.isfinite(tensor).all(): raise ValueError('Nonfinite teacher')
  if configs:
   first=configs[0]
   if semantic_robot(robot)!=semantic_robot(first['robot']):raise ValueError('Robot/control mismatch')
   for key in ('observation','action'):
    if cfg[key]!=first[key]:raise ValueError(key+' semantic mismatch')
  configs.append(cfg);checkpoints.append(ckpt)
 return checkpoints

@torch.no_grad()
def routed_targets(policy,observations,routes):
 if observations.ndim!=2 or observations.shape[1]!=449 or not torch.isfinite(observations).all():
  raise ValueError('Expected finite 449-wide teacher observations')
 if routes.shape!=(observations.shape[0],) or routes.device!=observations.device:
  raise ValueError('Routing batch/device mismatch')
 if routes.dtype not in (torch.int32,torch.int64) or ((routes < -1)|(routes>=len(SKILLS))).any():
  raise ValueError('Routes must be integral -1 or 0..2')
 if len(policy.teachers)!=len(SKILLS):raise ValueError('Wrong teacher count')
 policy.require_loaded_teachers()
 valid=routes>=0
 # A valid all-zero teacher output is still valid. Never infer validity from
 # numerical action values; preserve the explicit routing mask.
 result=observations.new_zeros(observations.shape[0],29)
 for index,teacher in enumerate(policy.teachers):
  mask=routes==index
  if mask.any():result[mask]=teacher(policy.teacher_obs_normalizers[index](observations[mask]))
 if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite teacher target')
 return result,valid

def explicit_masked_mse(prediction,target,valid):
 if prediction.shape!=target.shape or prediction.ndim!=2 or valid.shape!=prediction.shape[:1] or valid.dtype!=torch.bool:
  raise ValueError('Loss tensor/mask contract mismatch')
 if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
  raise FloatingPointError('Nonfinite loss input')
 per_row=(prediction-target).square().mean(-1)
 return (per_row*valid).sum()/valid.sum().clamp_min(1)

def checkpoint_contract():
 return {'schema':SCHEMA,'skills':list(SKILLS),'teacher_sha256':list(HASHES),
  'num_teachers':3,'physical_joints':31,'actions':29,'actor_width':96,
  'teacher_width_with_route':450,'depth_shape':[58,87],'algorithm':'pure_dagger',
  'fresh_student':True,'trajectory_exact_resume':False}

def validate_student_checkpoint(data):
 if data.get('num_teachers')!=3 or data.get('three_teacher_contract')!=checkpoint_contract():
  raise ValueError('Not this three-teacher checkpoint; five-skill/old-teacher state is forbidden')
 if data.get('teacher_loaded')!=[True]*3:raise ValueError('Unloaded teachers in student checkpoint')
 for key in ('model_state_dict','optimizer_state_dict'):
  if key not in data:raise ValueError('Missing student checkpoint state')
 return data

class BudgetExceeded(RuntimeError):pass

class UpdateBudget:
 """Counts completed training iterations; minibatch optimizer steps are separate.

 The worker's parent-process deadline must additionally cover download, install,
 simulator initialization, checkpointing and teardown. This is not that watchdog.
 """
 def __init__(self,max_updates=5,seconds=600,clock=time.monotonic):
  if type(max_updates) is not int or not 1<=max_updates<=5 or not 0<seconds<=600:
   raise ValueError('Scope may not exceed five updates / 600 seconds')
  self.max_updates=max_updates;self.clock=clock;self.deadline=clock()+seconds
  self.completed=0;self.active=False
 def begin(self):
  if self.active or self.completed>=self.max_updates or self.clock()>=self.deadline:
   raise BudgetExceeded('No further training iteration authorized')
  self.active=True
 def complete(self):
  if not self.active:raise RuntimeError('No active training iteration')
  self.completed+=1;self.active=False

class BoundedUpdatesMixin:
 """Clamp the actual native loop AND reject entry into a sixth training step.

 Initialize ``update_budget`` before ``learn``. Intentionally no retry or resume.
 """
 def learn(self,num_learning_iterations=None):
  remaining=self.update_budget.max_updates-self.update_budget.completed
  if remaining<=0:raise BudgetExceeded('Update budget already exhausted')
  if num_learning_iterations is not None and (type(num_learning_iterations) is not int or num_learning_iterations<1):
   raise ValueError('Positive integer iteration count required')
  requested=remaining if num_learning_iterations is None else min(num_learning_iterations,remaining)
  return super().learn(num_learning_iterations=requested)
 def _rollout_step(self,observations):
  self.update_budget.begin()
  return super()._rollout_step(observations)
 def _training_step(self):
  if not self.update_budget.active or self.update_budget.clock()>=self.update_budget.deadline:
   raise BudgetExceeded('Training outside active/time budget')
  result=super()._training_step()
  self.update_budget.complete()
  return result
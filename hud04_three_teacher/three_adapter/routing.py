"""Three-teacher routing without importing the PPO algorithm or simulator."""
import torch
from holosoma.agents.modules.student_teacher_modules import DepthStudentTeacher
from .contract import routed_targets
from .masked_training import routing_validity

def routing_column_index(env):
 manager=env.observation_manager;group=manager.cfg.groups.get('teacher_obs')
 if group is None or not group.concatenate or group.history_length!=1 or 'which_motion' not in group.terms:raise ValueError('Flat single-frame named teacher routing required')
 start=0
 for name in sorted(group.terms):
  value=manager._compute_term('teacher_obs',name,group.terms[name])
  if value.ndim!=2 or not torch.isfinite(value).all():raise ValueError('Invalid live teacher term')
  if name=='which_motion':
   if value.shape[1]!=1:raise ValueError('Routing term must have exactly one column')
   return start
  start+=value.shape[1]
 raise ValueError('No routing term')

class StrictRoutedTeacher(DepthStudentTeacher):
 def __init__(self,*args,num_teacher_obs,routing_column,**kwargs):
  if num_teacher_obs!=450 or not 0<=routing_column<450 or kwargs.get('num_teachers')!=3:raise ValueError('Three-teacher routed policy shape mismatch')
  super().__init__(*args,num_teacher_obs=num_teacher_obs-1,**kwargs)
  self.routing_column=routing_column;self.num_teacher_obs_with_routing=num_teacher_obs
 @torch.no_grad()
 def teacher_act(self,teacher_obs,teacher_idx=None):
  routing_validity(teacher_obs,self.routing_column)
  col=self.routing_column;route=teacher_obs[:,col].long()
  privileged=torch.cat((teacher_obs[:,:col],teacher_obs[:,col+1:]),dim=-1)
  result,_=routed_targets(self,privileged,route)
  return result
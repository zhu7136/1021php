"""Hu-D04 student command and three-expert sampling contract, version 1.

Student input uses 3 desired controls [vx_heading m/s, vy_heading m/s,
omega_world_z rad/s]. Training labels are instantaneous reference velocities,
not recovered human intent or G1's unavailable 15-class labels. Deployment
must supply desired commands in the same robot-heading frame; it must NOT
substitute measured robot linear velocity. These weak labels need later
closed-loop evaluation and do not establish zero-shot deployment success.
"""
import torch
from .contract import SKILLS

STUDENT_TERM_WIDTHS={'actions':29,'base_ang_vel':3,'dof_pos':29,'dof_vel':29,
 'robot_anchor_projected_gravity':3,'velocity_command':3}
STUDENT_ACTOR_WIDTH=sum(STUDENT_TERM_WIDTHS.values())
DEPTH_LATENT_WIDTH=32
STUDENT_MLP_INPUT_WIDTH=STUDENT_ACTOR_WIDTH+DEPTH_LATENT_WIDTH

def reference_planar_command(reference_linear_world,reference_angular_world,robot_quat_xyzw):
 n=reference_linear_world.shape[0]
 if reference_linear_world.shape!=(n,3) or reference_angular_world.shape!=(n,3) or robot_quat_xyzw.shape!=(n,4):
  raise ValueError('Expected batched reference linear/angular velocities and robot IMU quaternion')
 tensors=(reference_linear_world,reference_angular_world,robot_quat_xyzw)
 if any(t.device!=tensors[0].device or t.dtype!=tensors[0].dtype for t in tensors):raise ValueError('Command device/dtype mismatch')
 if not tensors[0].is_floating_point() or any(not torch.isfinite(t).all() for t in tensors):raise ValueError('Nonfinite/nonfloating command input')
 if not torch.allclose(robot_quat_xyzw.norm(dim=-1),torch.ones(n,device=tensors[0].device,dtype=tensors[0].dtype),rtol=0,atol=1e-4):
  raise ValueError('Robot quaternion must be unit xyzw')
 x,y,z,w=robot_quat_xyzw.unbind(-1)
 forward_x=1-2*(y*y+z*z);forward_y=2*(x*y+w*z)
 heading_norm=(forward_x.square()+forward_y.square()).sqrt()
 if (heading_norm<1e-5).any():raise ValueError('Robot heading undefined near vertical root X axis')
 c=forward_x/heading_norm;s=forward_y/heading_norm
 vx,vy,_=reference_linear_world.unbind(-1)
 return torch.stack((c*vx+s*vy,-s*vx+c*vy,reference_angular_world[:,2]),dim=-1).detach()

@torch.no_grad()
def desired_velocity_command(env):
 cmd=env.command_manager.get_state('motion_command');t=cmd.time_steps
 raw=cmd._root_body_index_in_motion
 return reference_planar_command(cmd.motion._body_lin_vel_w[t,raw],
  cmd.motion._body_ang_vel_w[t,raw],env.simulator.robot_root_states[:,3:7])

class BalancedClipSampler:
 """Uniform expert -> uniform clip -> uniform safe phase, no clip-end frames.

 A round-robin expert cycle gives all three labels even for tiny smoke batches.
 A random permutation each cycle prevents fixed slot bias. Over a full cycle
 each expert appears exactly once regardless of source frames/clip counts.
 Separate sampler instances/generators are needed per distributed process.
 """
 def __init__(self,routes,ends,generator=None):
  if routes.ndim!=1 or ends.shape!=routes.shape or ends.dtype!=torch.bool or routes.dtype!=torch.int64:
   raise ValueError('Invalid routing/clip boundary arrays')
  if routes.numel()==0 or not ends[-1] or not torch.equal(routes.unique(sorted=True),torch.arange(len(SKILLS),device=routes.device)):
   raise ValueError('Exactly three nonempty expert routes required')
  self.device=routes.device;self.generator=generator
  self.end=ends.nonzero().flatten()+1;self.start=torch.cat((self.end.new_zeros(1),self.end[:-1]))
  if ((self.end-self.start)<2).any():raise ValueError('Every clip needs at least two frames')
  self.clip_routes=routes[self.start]
  reconstructed=torch.repeat_interleave(self.clip_routes,self.end-self.start)
  if not torch.equal(reconstructed,routes):raise ValueError('Teacher route changes inside a clip')
  self.pools=[torch.nonzero(self.clip_routes==i).flatten() for i in range(len(SKILLS))]
  if any(len(p)==0 for p in self.pools):raise ValueError('Missing teacher clip pool')
  self.pending=routes.new_empty(0)
 def sample(self,n,evaluation=False):
  if type(n) is not int or n<0:raise ValueError('Invalid sample count')
  pieces=[];remaining=n
  while remaining:
   if not self.pending.numel():self.pending=torch.randperm(len(SKILLS),device=self.device,generator=self.generator)
   size=min(remaining,self.pending.numel());pieces.append(self.pending[:size]);self.pending=self.pending[size:];remaining-=size
  selected=torch.cat(pieces) if pieces else self.start.new_empty(0)
  clip_ids=torch.empty_like(selected)
  for expert,pool in enumerate(self.pools):
   mask=selected==expert;size=int(mask.sum())
   if size:clip_ids[mask]=pool[torch.randint(len(pool),(size,),device=self.device,generator=self.generator)]
  start=self.start[clip_ids];end=self.end[clip_ids]
  phase=torch.zeros(n,device=self.device) if evaluation else torch.rand(n,device=self.device,generator=self.generator)
  time_steps=start+(phase*(end-start-1).float()).long()
  return clip_ids,time_steps,start,end,selected
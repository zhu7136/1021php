"""Fixed-clip, no-autoreset native bindings; no optimizer or learning code."""
import json
import torch
from hud04_php31_native import HuD04PHPMotionCommand
from wbt_training.hud04_holosoma_env import HuD04WholeBodyTrackingManager

class EpisodeStopped(RuntimeError):pass

class FixedMotion(HuD04PHPMotionCommand):
 def reset(self,env_ids):
  if getattr(self._env,'comparison_armed',False) and env_ids is not None and env_ids.numel():raise EpisodeStopped('unexpected_motion_reset_prevented')
  return super().reset(env_ids)

def values(x):return x.detach().cpu().tolist()

class FrozenEnvironment(HuD04WholeBodyTrackingManager):
 comparison_armed=False
 def reset_envs_idx(self,env_ids,target_states=None,target_buf=None):
  if self.comparison_armed and env_ids.numel():raise EpisodeStopped('native_tracking_termination')
  return super().reset_envs_idx(env_ids,target_states,target_buf)
 def snapshot(self,stage):
  m=self.command_manager.get_state('motion_command');sim=self.simulator
  term=list(self.action_manager._term_instances.values())
  assert len(term)==1
  a=term[0]
  err=torch.linalg.vector_norm(m.body_pos_relative_w-m.robot_body_pos_w,dim=-1)
  names=list(m.motion_cfg.body_names_to_track)
  row=dict(stage=stage,policy_step=int(self.episode_length_buf[0]),reference_frame=int(m.time_steps[0]),time_s=float(self.episode_length_buf[0])*float(self.dt),root_state=values(sim.robot_root_states[0]),joint_position=values(sim.dof_pos[0]),joint_velocity=values(sim.dof_vel[0]),reference_joint_position=values(m.joint_pos[0]),reference_joint_velocity=values(m.joint_vel[0]),body_names=names,body_position=values(m.robot_body_pos_w[0]),reference_body_position=values(m.body_pos_w[0]),aligned_reference_body_position=values(m.body_pos_relative_w[0]),body_error_m=values(err[0]),anchor_error_m=float(torch.linalg.vector_norm(m.ref_pos_w-m.robot_ref_pos_w,dim=-1)[0]),max_tracked_body_error_m=float(err.max()),action=values(self.action_manager.action[0]),position_target=values(a._applied_position_targets[0]),requested_position_target=values(a._requested_position_targets[0]),contact_body_names=list(sim._body_list),contact_force_w=values(sim.contact_forces[0]),reset=bool(self.reset_buf[0]),terminated=bool(self.termination_manager.terminated[0]),timeout=bool(self.time_out_buf[0]))
  self.comparison_stream.write(json.dumps(row,allow_nan=False)+'\n');self.comparison_stream.flush()
  return row
 def _check_termination(self):
  super()._check_termination()
  if self.comparison_armed:self.comparison_last=self.snapshot('after_physics_before_reset_or_reference_advance')
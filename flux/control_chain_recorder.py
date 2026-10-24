"""Read-only pre-reset control telemetry; no reward or dynamics recomputation."""
from __future__ import annotations
import json
import torch

class Recorder:
    def __init__(self,env,motion,sink):
        self.env,self.motion,self.sink=env,motion,sink
        self.phase=None;self.step=-1;self.pending=None;self.hooked=False
        self.names=list(env.robot_config.dof_names)
        self.limits=torch.as_tensor(env.robot_config.dof_effort_limit_list,device=env.device)
        self.last_phase=None;self.episode_ids=None
    def begin_step(self,phase,step,actor_obs,actor_mean,actions):
        assert self.phase is None and self.pending is None
        self.phase,self.step=phase,step
        if self.last_phase!=phase:
            self.episode_ids=torch.zeros(actions.shape[0],dtype=torch.long,device=actions.device)
            self.last_phase=phase
        self.input={k:v.detach().clone() for k,v in dict(actor_obs=actor_obs,actor_mean=actor_mean,actions=actions,reference_q=self.motion.joint_pos,q_before=self.env.simulator.dof_pos,episode_ids=self.episode_ids).items()}
        self.input['motion_frame_before']=self.motion.time_steps.detach().clone()
    def install(self):
        assert not self.hooked
        env=self.env;self.original_check=env._check_termination;self.original_reward=env._compute_reward
        def check():
            result=self.original_check()
            if self.phase is None:return result
            sim=env.simulator;data=sim._robot.data;ids=sim.dof_ids
            terms=[x for x in env.action_manager._term_instances.values() if hasattr(x,'_applied_position_targets')]
            assert len(terms)==1
            target=terms[0]._applied_position_targets.detach().clone()
            native=data.joint_pos_target[:,ids].detach().clone()
            assert torch.equal(native,target),'Native target readback differs from processed target'
            hard=sim.hard_dof_pos_limits
            assert hard.shape==(len(self.names),2)
            bnames=list(self.motion.motion_cfg.body_names_to_track)
            tracked=['left_ankle_roll_link','right_ankle_roll_link'];bi=[bnames.index(n) for n in tracked]
            values=dict(self.input)
            values.update(q_target=target,q_target_native_readback=native,q_after=sim.dof_pos,qdot_after=sim.dof_vel,
                target_outside_limits=(target<hard[:,0])|(target>hard[:,1]),target_error=target-sim.dof_pos,
                done=env.reset_buf.bool(),episode_age=env.episode_length_buf,
                tracked_ankle_error_m=torch.linalg.vector_norm(self.motion.body_pos_relative_w[:,bi]-self.motion.robot_body_pos_w[:,bi],dim=-1))
            for name in ['applied_torque','computed_torque','joint_stiffness','joint_damping','joint_effort_limits']:
                tensor=getattr(data,name,None)
                values[name]=None if tensor is None else tensor[:,ids]
            if values['applied_torque'] is not None:
                values['configured_effort_limit_fraction']=values['applied_torque'].abs()/self.limits
            sensor=sim.contact_sensor;contact_names=[str(n) for n in sensor.body_names]
            foot_ids=[contact_names.index(n) for n in tracked]
            values['ankle_contact_net_force_w']=sensor.data.net_forces_w[:,foot_ids]
            self.pending={k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in values.items()}
            return result
        def reward():
            if self.phase is None:return self.original_reward()
            assert self.pending is not None,'Reward ran without pre-reset capture'
            sums=env.reward_manager._episode_sums;before={k:v.detach().clone() for k,v in sums.items()}
            result=self.original_reward()
            contributions={k:(sums[k]-before[k]).detach().clone() for k in before}
            assert torch.allclose(sum(contributions.values()),env.rew_buf,atol=1e-6,rtol=1e-5),'Reward contribution mismatch'
            row=dict(phase=self.phase,control_step=self.step,capture_point='post_physics_pre_reset',joint_names=self.names,
                effort_limit_source='robot configuration; not independent measured force',
                torque_semantics='IsaacLab implicit-actuator buffers; NOT independent joint force measurements',
                values={k:v.cpu().tolist() if isinstance(v,torch.Tensor) else v for k,v in self.pending.items()},
                reward_contributions={k:v.cpu().tolist() for k,v in contributions.items()},reward_total=env.rew_buf.detach().cpu().tolist())
            self.sink.write(json.dumps(row,allow_nan=False)+'\n');self.sink.flush()
            self.episode_ids+=self.pending['done'].long()
            self.phase=None;self.pending=None
            return result
        env._check_termination=check;env._compute_reward=reward;self.hooked=True
    def uninstall(self):
        assert self.phase is None and self.pending is None
        if self.hooked:
            self.env._check_termination=self.original_check;self.env._compute_reward=self.original_reward;self.hooked=False
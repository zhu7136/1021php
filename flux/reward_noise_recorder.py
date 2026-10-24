"""Compact, pre-reset all-step rewards and action telemetry; never changes rewards."""
from __future__ import annotations
import json
import torch

class Recorder:
    def __init__(self,env,motion,sink):
        self.env,self.motion,self.sink=env,motion,sink
        self.phase=None;self.pending=None;self.hooked=False;self.last_phase=None
        self.names=list(env.robot_config.dof_names)

    def begin_step(self,phase,step,actor_obs,actor_mean,actions):
        assert self.phase is None and self.pending is None
        assert actions.shape==actor_mean.shape and actor_obs.shape[1]>=actions.shape[1]
        assert torch.isfinite(actions).all() and torch.isfinite(actor_mean).all()
        self.phase,self.step=phase,step
        if phase!=self.last_phase:
            self.episode_ids=torch.zeros(actions.shape[0],dtype=torch.long,device=actions.device)
            self.last_phase=phase
        self.input={k:v.detach().clone() for k,v in dict(
            actor_mean=actor_mean,actions=actions,previous_actions=actor_obs[:,:actions.shape[1]],
            reference_q=self.motion.joint_pos,q_before=self.env.simulator.dof_pos,
            motion_frame_before=self.motion.time_steps,episode_ids=self.episode_ids).items()}

    def install(self):
        assert not self.hooked
        env=self.env
        self.original_check,self.original_reward=env._check_termination,env._compute_reward
        def check():
            result=self.original_check()
            if self.phase is None:return result
            sim=env.simulator
            terms=[x for x in env.action_manager._term_instances.values() if hasattr(x,'_applied_position_targets')]
            assert len(terms)==1
            target=terms[0]._applied_position_targets
            native=sim._robot.data.joint_pos_target[:,sim.dof_ids]
            assert torch.equal(target,native),'Native target mismatch'
            values=dict(self.input)
            values.update(q_target=target,q_after=sim.dof_pos,done=env.reset_buf.bool(),
                          timeout=env.time_out_buf.bool(),episode_age=env.episode_length_buf)
            self.pending={k:v.detach().clone() for k,v in values.items()}
            return result
        def reward():
            if self.phase is None:return self.original_reward()
            assert self.pending is not None,'Missing pre-reset reward capture'
            sums=env.reward_manager._episode_sums
            before={k:v.detach().clone() for k,v in sums.items()}
            result=self.original_reward()
            contributions={k:(sums[k]-before[k]).detach().clone() for k in before}
            assert torch.allclose(sum(contributions.values()),env.rew_buf,atol=1e-6,rtol=1e-5),'Reward contribution mismatch'
            row=dict(phase=self.phase,control_step=self.step,capture_point='post_physics_pre_reset',
                     joint_names=self.names,
                     values={k:v.cpu().tolist() for k,v in self.pending.items()},
                     reward_contributions={k:v.cpu().tolist() for k,v in contributions.items()},
                     reward_total=env.rew_buf.detach().cpu().tolist())
            self.sink.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');self.sink.flush()
            self.episode_ids+=self.pending['done'].long()
            self.phase=None;self.pending=None
            return result
        env._check_termination,env._compute_reward=check,reward
        self.hooked=True

    def uninstall(self):
        assert self.phase is None and self.pending is None
        if self.hooked:
            self.env._check_termination,self.env._compute_reward=self.original_check,self.original_reward
            self.hooked=False
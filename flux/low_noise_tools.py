"""Initial exploration intervention and read-only native PPO diagnostics."""
import math
import torch

@torch.no_grad()
def set_initial_noise(actor,scale,*,completed_rollouts):
    if completed_rollouts!=0:raise ValueError('Intervention must precede fresh rollout')
    if scale not in (1.,.25):raise ValueError('Undeclared intervention')
    if getattr(actor,'_initial_noise_applied',False):raise ValueError('Intervention already applied')
    if getattr(actor,'distribution',None) is not None:raise ValueError('Freshly restored actor required')
    before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    old=actor.std.detach().clone()
    if old.shape!=(29,) or not torch.isfinite(old).all() or not (old>0).all():raise ValueError('Invalid std')
    if actor.min_noise_std or actor.min_mean_noise_std:raise ValueError('Unexpected std floor')
    actor.std.mul_(scale);actor.project_action_std()
    if not torch.equal(actor.std,old*scale):raise ValueError('Projection changed intervention')
    for k,v in actor.state_dict().items():
        if k!='std' and not torch.equal(v,before[k]):raise ValueError('Mean policy changed')
    actor._initial_noise_applied=True
    return dict(scale=scale,old_std_mean=float(old.mean()),new_std_mean=float(actor.std.mean()),
                new_std_min=float(actor.std.min()),new_std_max=float(actor.std.max()),
                optimizer_moments_preserved=True,mean_weights_unchanged=True,
                entropy_difference=29*math.log(scale))

@torch.no_grad()
def distribution_metrics(actor,minibatch,clip_param):
    n=minibatch['actions'].shape[0]
    mu,sigma=actor.action_mean,actor.action_std
    if mu.shape[0]!=n:raise ValueError('Unexpected augmentation')
    old_mu,old_sigma=minibatch['action_mean'],minibatch['action_sigma']
    if mu.shape!=old_mu.shape or sigma.shape!=old_sigma.shape:raise ValueError('Distribution mismatch')
    ratio=(actor.get_actions_log_prob(minibatch['actions']).reshape(-1)-minibatch['actions_log_prob'].reshape(-1)).exp()
    kl=(torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1).mean()
    row=dict(kl=float(kl),clip_fraction=float(((ratio-1).abs()>clip_param).float().mean()),
             entropy=float(actor.entropy.mean()),std_mean=float(sigma.mean()),std_min=float(sigma.min()),std_max=float(sigma.max()))
    if not all(math.isfinite(x) for x in row.values()):raise FloatingPointError('Nonfinite PPO telemetry')
    return row

def stop_reason(updates,elapsed):
    if updates>=200:return 'iteration_budget'
    if elapsed>=300:return 'time_budget'
    return None

def finite(tensor):
    if not torch.isfinite(tensor).all():raise FloatingPointError('Nonfinite training tensor')

class EnvironmentStats:
    """Aggregate actual per-step contributions and pre-reset term flags, no trajectory retention."""
    def __init__(self,env):
        self.env=env;self.active=False;self.mask=None;self.terms={};self.clear()
        # Capture actual native term return values ONCE; never recompute stateful termination terms.
        def wrapped(fn,name):
            def call(*args,**kwargs):
                value=fn(*args,**kwargs)
                if self.active:self.terms[name]=value.detach().clone()
                return value
            return call
        class Proxy:
            def __init__(proxy,target,name):proxy.target=target;proxy.call=wrapped(target,name)
            def __getattr__(proxy,key):return getattr(proxy.target,key)
            def __call__(proxy,*args,**kwargs):return proxy.call(*args,**kwargs)
        manager=env.termination_manager
        for k,fn in list(manager._term_funcs.items()):manager._term_funcs[k]=wrapped(fn,k)
        for k,instance in list(manager._term_instances.items()):manager._term_instances[k]=Proxy(instance,k)
        reward,check=env._compute_reward,env._check_termination
        def checked():
            if self.active:self.terms={}
            result=check()
            if self.active:
                for k,v in self.terms.items():
                    count=v.bool() if self.mask is None else v.bool()&self.mask
                    self.terminations[k]=self.terminations.get(k,0)+int(count.sum())
            return result
        def rewarded():
            if not self.active:return reward()
            before={k:v.clone() for k,v in env.reward_manager._episode_sums.items()}
            result=reward()
            finite(env.rew_buf)
            mask=torch.ones(env.num_envs,dtype=torch.bool,device=env.device) if self.mask is None else self.mask
            self.n+=int(mask.sum())
            for k,v in env.reward_manager._episode_sums.items():
                value=v-before[k];finite(value)
                self.rewards[k]=self.rewards.get(k,0.)+float(value[mask].sum())
            self.total+=float(env.rew_buf[mask].sum())
            return result
        env._compute_reward,env._check_termination=rewarded,checked
    def clear(self):
        self.n=0;self.total=0.;self.rewards={};self.terminations={}
    def summary(self):
        return dict(environment_steps=self.n,mean_reward=self.total/max(self.n,1),
                    reward_terms={k:v/max(self.n,1) for k,v in self.rewards.items()},termination_flags=self.terminations.copy())

@torch.no_grad()
def evaluate(algo,motion,stats,seed=1402):
    """First episode per env, right-censored at128steps; matched seed, uniform reference frames."""
    import dataclasses,random
    import numpy as np
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    old_cfg=motion.motion_cfg
    motion.motion_cfg=dataclasses.replace(old_cfg,use_adaptive_timesteps_sampler=False)
    env=algo.env;algo._eval_mode()
    old_evaluating=env.is_evaluating
    env.is_evaluating=False  # Native eval flag forces frame zero, NOT uniform training frames.
    stats.active=False
    try:
        env.reset_all()
        env.step({'actions':torch.zeros((32,29),device=algo.device)})
        obs=env.reset_all()
        alive=torch.ones(32,dtype=torch.bool,device=algo.device)
        survival=torch.full((32,),128,dtype=torch.long,device=algo.device)
        initial_q=env.simulator.dof_pos.detach().clone().cpu().tolist()
        initial_frames=motion.time_steps.detach().clone().cpu().tolist()
        stats.clear();stats.active=True
        for step in range(1,129):
            stats.mask=alive.clone()
            raw=torch.cat([obs[k] for k in algo.actor_obs_keys],dim=1)
            finite(raw)
            action=algo.actor.act_inference({'actor_obs':algo._normalize_actor_obs(raw,update=False)})
            finite(action)
            obs,reward,done,info=env.step({'actions':action})
            newly_done=alive&done.bool();survival[newly_done]=step;alive&=~done.bool()
        result=dict(seed=seed,horizon_steps=128,dt=float(env.dt),reset_sampler='uniform_training_frames',
                    first_episode_steps=survival.cpu().tolist(),survivors_at_horizon=int(alive.sum()),
                    restricted_mean_survival_seconds=float(survival.float().mean())*float(env.dt),
                    initial_joint_positions=initial_q,initial_motion_frames=initial_frames,**stats.summary())
        result['tracking_positive_mean_reward']=sum(v for k,v in result['reward_terms'].items() if env.reward_manager.cfg.terms[k].weight>0)
        return result
    finally:
        stats.active=False;stats.mask=None;motion.motion_cfg=old_cfg;env.is_evaluating=old_evaluating

class TrialStop(Exception):pass

def train(algo,stats,out,emit,clock=None):
    """Wrap native PPO; do not reimplement loss, rollout, Adam, LR schedule or save."""
    import json,time
    from pathlib import Path
    if clock is None:clock=time.monotonic
    out=Path(out);original_loss=algo._compute_ppo_loss;original_post=algo._post_epoch_logging
    updates=0;batch=0;start=clock()
    stats.clear();stats.mask=None;stats.active=True
    metrics_file=(out/'minibatches.jsonl').open('w')
    progress_file=(out/'iterations.jsonl').open('w')
    def loss(minibatch):
        nonlocal batch
        for value in minibatch.values():
            if torch.is_tensor(value):finite(value)
        result=original_loss(minibatch)
        for value in result.values():
            if torch.is_tensor(value):finite(value)
        row=distribution_metrics(algo.actor,minibatch,algo.config.clip_param)
        row.update(iteration=int(algo.current_learning_iteration),minibatch=batch,
                   actor_lr=algo.actor_optimizer.param_groups[0]['lr'],critic_lr=algo.critic_optimizer.param_groups[0]['lr'])
        metrics_file.write(json.dumps(row,allow_nan=False)+'\n');batch+=1
        return result
    def epoch(it,losses):
        nonlocal updates
        original_post(it,losses);updates+=1
        finite(algo.actor.std)
        if not (algo.actor.std>0).all():raise FloatingPointError('Nonpositive action std')
        row=dict(event='low_noise_training_progress',iteration=int(it),completed_updates=updates,
                 elapsed_seconds=clock()-start,losses={k:float(v) for k,v in losses.items()},
                 actor_lr=algo.actor_learning_rate,critic_lr=algo.critic_learning_rate,
                 std_mean=float(algo.actor.std.mean()),std_min=float(algo.actor.std.min()),std_max=float(algo.actor.std.max()),
                 **stats.summary())
        progress_file.write(json.dumps(row,allow_nan=False)+'\n');progress_file.flush();metrics_file.flush()
        emit(**row);stats.clear()
        reason=stop_reason(updates,clock()-start)
        if reason:raise TrialStop(reason)
    original_steps=[]
    for opt in (algo.actor_optimizer,algo.critic_optimizer):
        original_step=opt.step
        def checked_step(*args,_opt=opt,_step=original_step,**kwargs):
            grads=[p.grad.reshape(-1) for g in _opt.param_groups for p in g['params'] if p.grad is not None]
            if grads:finite(torch.cat(grads))
            result=_step(*args,**kwargs)
            finite(torch.cat([p.detach().reshape(-1) for g in _opt.param_groups for p in g['params']]))
            return result
        opt.step=checked_step;original_steps.append((opt,original_step))
    original_act=algo.actor.act
    def act(obs):
        for v in obs.values():finite(v)
        actions=original_act(obs);finite(actions);return actions
    algo.actor.act=act;algo._compute_ppo_loss=loss;algo._post_epoch_logging=epoch
    reason='iteration_budget'
    try:
        algo.learn()
    except TrialStop as exc:reason=str(exc)
    finally:
        algo.actor.act=original_act;algo._compute_ppo_loss=original_loss;algo._post_epoch_logging=original_post
        for opt,original_step in original_steps:opt.step=original_step
        stats.active=False;metrics_file.close();progress_file.close()
    if updates<1:raise RuntimeError('No completed PPO update')
    return dict(completed_updates=updates,minibatches=batch,stop_reason=reason,training_seconds=clock()-start,
                last_iteration=int(algo.current_learning_iteration),transitions=updates*32*24)
"""Read-only, pre-reset first-episode diagnostics and paired evaluation checks."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch

SEEDS=(1402,2402,3402)
INPUTS={'model_preflight.pt':'86e249d10da6a6c48ce0bb867660a3bc1faca5b4c9fd25fdeff305914631237b',
        'model_final.pt':'3dd4286a4b5e825f8e224f476cc5cc510a375c77a9ffabd168f518d946c680f1'}
BUNDLE_SHA='c30ecf9b2d7d8ab50e7864b79a97c56b356beea63f61106d746ee8047d93de69'
BUNDLE_BYTES=13513018

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare_pair(before,after,initial_before,initial_after):
    if before['seed']!=after['seed']:raise ValueError('Seed mismatch')
    if before['horizon_steps']!=128 or after['horizon_steps']!=128:raise ValueError('Horizon mismatch')
    if before['dt']!=after['dt']:raise ValueError('Timestep mismatch')
    if initial_before.keys()!=initial_after.keys():raise ValueError('Initial state keys differ')
    errors={}
    for name in initial_before:
        a=torch.as_tensor(initial_before[name]);b=torch.as_tensor(initial_after[name])
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Invalid initial state: '+name)
        error=float((a-b).abs().max()) if a.numel() else 0.
        errors[name]=error
        if not torch.equal(a,b):raise ValueError('Initial state is not exactly paired: '+name)
    for name in ['initial_motion_frames','initial_joint_positions']:
        if before[name]!=after[name]:raise ValueError('Reset mismatch: '+name)
    b=torch.tensor(before['first_episode_steps']);a=torch.tensor(after['first_episode_steps'])
    if b.shape!=(32,) or a.shape!=(32,):raise ValueError('Expected 32 environments')
    return dict(seed=before['seed'],initial_state_exact=True,initial_state_max_errors=errors,
        before_seconds=before['restricted_mean_survival_seconds'],
        after_seconds=after['restricted_mean_survival_seconds'],
        improved=int((a>b).sum()),unchanged=int((a==b).sum()),worse=int((a<b).sum()),
        before_survivors=before['survivors_at_horizon'],after_survivors=after['survivors_at_horizon'])

class FirstEpisodeCapture:
    """Runs after native termination checks but BEFORE auto-reset; never repeats term __call__."""
    def __init__(self,env,motion,stats,path):
        self.env=env;self.motion=motion;self.stats=stats;self.steps=0
        self.failures=[];self.seen=set();self.samples=0;self.target_violations=0
        self.initial=None;self.sink=Path(path).open('x',buffering=1)
        term=env.termination_manager._term_instances['bad_tracking']
        self.term=term
        self.names=list(term.bad_motion_body_pos_body_names)
        self.indices=term.bad_motion_body_pos_body_indexes
        # This diagnostic implements the exact native 3-D BadTracking formula, not Z-only variants.
        if type(getattr(term,'target',term)).__name__!='BadTracking':
            raise ValueError('Unsupported termination class; do not approximate its formula')
        original=env._check_termination
        def checked():
            result=original()
            if stats.active:self.observe()
            return result
        env._check_termination=checked
        reset=env.reset_all
        def reset_all(*args,**kwargs):
            obs=reset(*args,**kwargs)
            # evaluate() makes two reset_all calls with one warm-up step between.
            # Only the final reset snapshot is retained; warm-up tracking is not sampled.
            # Native RootStatesProxy is NOT a Tensor. Its public index path returns
            # the environment's xyzw state; backing IsaacSim storage uses wxyz.
            root_states=env.simulator.robot_root_states[:]
            if not torch.is_tensor(root_states):raise TypeError('Root-state indexing must return a Tensor')
            if root_states.shape!=(env.simulator.dof_pos.shape[0],13):
                raise ValueError('Root-state shape must be (num_envs, 13)')
            if not torch.isfinite(root_states).all():raise ValueError('Nonfinite root states')
            self.initial=dict(joint_positions=env.simulator.dof_pos.detach().cpu().tolist(),
                joint_velocities=env.simulator.dof_vel.detach().cpu().tolist(),
                root_states=root_states.detach().cpu().tolist(),
                motion_frames=motion.time_steps.detach().cpu().tolist(),
                dr_episode_count=env.dr_episode_count.detach().cpu().tolist(),
                **{'observation/'+k:v.detach().cpu().tolist() for k,v in obs.items()})
            return obs
        env.reset_all=reset_all

    @torch.no_grad()
    def observe(self):
        env,motion,term=self.env,self.motion,self.term
        self.steps+=1
        mask=self.stats.mask
        if mask is None or mask.shape!=(32,):raise ValueError('First-episode mask required')
        errors=torch.linalg.vector_norm(motion.body_pos_relative_w[:,self.indices]-motion.robot_body_pos_w[:,self.indices],dim=-1)
        root_error=torch.linalg.vector_norm(motion.ref_pos_w-motion.robot_ref_pos_w,dim=-1)
        ref_bad=term.bad_ref_pos(motion);ori_bad=term.bad_ref_ori(motion)
        body_bad=errors>term.bad_motion_body_pos_threshold
        if motion.motion.has_object:raise ValueError('Object-tracking diagnostics not declared')
        native=self.stats.terms['bad_tracking'].bool()
        if not torch.equal(native,ref_bad|ori_bad|body_bad.any(dim=-1)):
            raise ValueError('Diagnostic does not match native bad_tracking flags')
        terms=[t for t in env.action_manager._term_instances.values() if hasattr(t,'_requested_position_targets')]
        if len(terms)!=1:raise ValueError('Protected target term missing')
        action=terms[0];target=action._applied_position_targets;limits=env.simulator.hard_dof_pos_limits
        native_target=env.simulator._robot.data.joint_pos_target[:,env.simulator.dof_ids]
        if not torch.equal(target,native_target):raise ValueError('Applied target readback mismatch')
        outside=(target<limits[:,0])|(target>limits[:,1])
        if outside.any():raise ValueError('Applied targets escaped hard limits')
        active=mask.nonzero().flatten()
        for v in [errors[mask],root_error[mask],target[mask],action._requested_position_targets[mask]]:
            if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite evaluation telemetry')
        self.samples+=int(mask.sum())
        row=dict(step=self.steps,seconds=self.steps*float(env.dt),active_envs=active.cpu().tolist(),
            body_names=self.names,body_errors_m=errors[mask].cpu().tolist(),
            root_errors_m=root_error[mask].cpu().tolist(),
            root_orientation_failed=ori_bad[mask].cpu().tolist(),
            guard_fraction=float((action._requested_position_targets[mask]!=target[mask]).float().mean()) if mask.any() else 0.,
            applied_target_hard_limit_violations=int(outside[mask].sum()))
        self.sink.write(json.dumps(row,allow_nan=False)+'\n')
        failed=(mask&env.reset_buf.bool()).nonzero().flatten().cpu().tolist()
        for i in failed:
            if i in self.seen:raise ValueError('Repeated first-episode failure')
            self.seen.add(i)
            flags={k:bool(v[i]) for k,v in self.stats.terms.items()}
            self.failures.append(dict(env=i,step=self.steps,seconds=self.steps*float(env.dt),
                termination_flags=flags,bad_ref_pos=bool(ref_bad[i]),bad_ref_ori=bool(ori_bad[i]),
                failed_bodies=[name for j,name in enumerate(self.names) if bool(body_bad[i,j])],
                body_errors_m={name:float(errors[i,j]) for j,name in enumerate(self.names)},
                root_position_error_m=float(root_error[i]),
                requested_targets_rad=action._requested_position_targets[i].cpu().tolist(),
                applied_targets_rad=target[i].cpu().tolist(),
                joint_positions_rad=env.simulator.dof_pos[i].cpu().tolist()))

    def finish(self,evaluation):
        self.sink.close()
        if self.steps!=128 or self.initial is None:raise ValueError('Incomplete evaluation capture')
        if len(self.failures)+evaluation['survivors_at_horizon']!=32:raise ValueError('Failure coverage differs')
        for failure in self.failures:
            if evaluation['first_episode_steps'][failure['env']]!=failure['step']:
                raise ValueError('First failure timing mismatch')
        return dict(first_episode_only=True,warmup_and_post_failure_samples_excluded=True,
            first_episode_samples=self.samples,first_failures=self.failures,
            applied_target_hard_limit_violations=self.target_violations,
            body_threshold_m=float(self.term.bad_motion_body_pos_threshold),
            root_position_threshold_m=float(self.term.bad_ref_pos_threshold),
            root_orientation_threshold=float(self.term.bad_ref_ori_threshold))
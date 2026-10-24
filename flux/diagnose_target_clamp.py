"""Frozen paired intervention: only clamp policy position targets to hard limits."""
from pathlib import Path
import diagnose_short_episodes as base

MODES=('policy_mean','policy_mean_clamped','reference_pd')
NUM_ENVS=32
STEPS_PER_MODE=64
WORKER_SECONDS=600
MARGIN_RAD=1e-6

def transform(env,mode,actions):
    if mode!='policy_mean_clamped':return actions
    import torch
    scale=env.action_scales
    limits=env.simulator.hard_dof_pos_limits
    assert scale.shape==(actions.shape[1],) and limits.shape==(actions.shape[1],2)
    assert torch.isfinite(actions).all() and torch.isfinite(limits).all()
    assert torch.isfinite(scale).all() and (scale>0).all()
    assert ((limits[:,1]-limits[:,0])>2*MARGIN_RAD).all()
    control=env.robot_config.control
    source=torch.clamp(actions,-control.action_clip_value,control.action_clip_value) if control.clip_actions else actions
    target=env.default_dof_pos+source*scale
    bounded=torch.maximum(torch.minimum(target,limits[:,1]-MARGIN_RAD),limits[:,0]+MARGIN_RAD)
    corrected=(bounded-env.default_dof_pos)/scale
    # Preserve exactly the original action wherever no target correction is needed.
    corrected=torch.where(bounded==target,actions,corrected)
    raw=torch.clamp(corrected,-control.action_clip_value,control.action_clip_value) if control.clip_actions else corrected
    readback=env.default_dof_pos+raw*scale
    assert ((readback>=limits[:,0])&(readback<=limits[:,1])).all()
    return corrected

def configure():
    original=base.diagnose
    def diagnose(dataset):
        from control_chain_recorder import Recorder
        return original(dataset,recorder_factory=Recorder,action_transform=transform)
    base.MODES=MODES;base.NUM_ENVS=NUM_ENVS;base.STEPS_PER_MODE=STEPS_PER_MODE
    base.WORKER_SECONDS=WORKER_SECONDS
    base.OUT=base.ROOT/'outputs/target_clamp_diagnostic'
    base.ENTRYPOINT=Path(__file__).resolve()
    base.diagnose=diagnose

if __name__=='__main__':
    configure()
    base.main()
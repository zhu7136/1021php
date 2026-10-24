"""Frozen output-amplitude intervention, without target clamping or training."""
from pathlib import Path
import diagnose_short_episodes as base

MODES=('policy_mean','policy_mean_scaled025','default_pose_pd','reference_pd')
NUM_ENVS=32
STEPS_PER_MODE=64
WORKER_SECONDS=600

def transform(env,mode,actions):
    if mode not in ('policy_mean_scaled025','default_pose_pd'):return actions
    import torch
    assert torch.isfinite(actions).all()
    return actions*(.25 if mode=='policy_mean_scaled025' else 0.)

def configure():
    original=base.diagnose
    def diagnose(dataset):
        from control_chain_recorder import Recorder
        return original(dataset,recorder_factory=Recorder,action_transform=transform)
    base.MODES=MODES;base.NUM_ENVS=NUM_ENVS;base.STEPS_PER_MODE=STEPS_PER_MODE
    base.WORKER_SECONDS=WORKER_SECONDS
    base.OUT=base.ROOT/'outputs/action_scale_diagnostic'
    base.ENTRYPOINT=Path(__file__).resolve()
    base.diagnose=diagnose

if __name__=='__main__':
    configure()
    base.main()
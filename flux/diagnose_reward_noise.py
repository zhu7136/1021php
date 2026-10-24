"""One frozen mean-vs-sampled-action diagnostic, all-step reward capture, no training."""
from pathlib import Path
import diagnose_short_episodes as base

MODES=('policy_mean','policy_sample')
NUM_ENVS=32
STEPS_PER_MODE=64
WORKER_SECONDS=600

def configure():
    original=base.diagnose
    def diagnose(dataset):
        from reward_noise_recorder import Recorder
        return original(dataset,recorder_factory=Recorder,capture_steps=STEPS_PER_MODE)
    base.MODES=MODES;base.NUM_ENVS=NUM_ENVS;base.STEPS_PER_MODE=STEPS_PER_MODE
    base.WORKER_SECONDS=WORKER_SECONDS
    base.OUT=base.ROOT/'outputs/reward_noise_diagnostic'
    base.ENTRYPOINT=Path(__file__).resolve()
    base.diagnose=diagnose

if __name__=='__main__':
    configure()
    base.main()
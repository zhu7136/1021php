"""One authorized frozen diagnostic: 32env, 2x64 steps, 600s worker, no retry."""
from pathlib import Path
import diagnose_short_episodes as base

MODES=('policy_mean','reference_pd')
NUM_ENVS=32
STEPS_PER_MODE=64
WORKER_SECONDS=600
SHA=base.SHA

def configure():
    from control_chain_recorder import Recorder
    original=base.diagnose
    def diagnose(dataset):return original(dataset,recorder_factory=Recorder)
    base.MODES=MODES;base.NUM_ENVS=NUM_ENVS;base.STEPS_PER_MODE=STEPS_PER_MODE
    base.WORKER_SECONDS=WORKER_SECONDS
    base.OUT=base.ROOT/'outputs/control_chain_diagnostic'
    base.ENTRYPOINT=Path(__file__).resolve()
    base.diagnose=diagnose

if __name__=='__main__':
    configure()
    base.main()
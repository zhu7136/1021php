"""One GPU plumbing check: iter200 ->201,32env,no evaluation,1800s worker,no task retry."""
from pathlib import Path
import json
import train_protected_resume as resume

def summarize():
    result=json.loads((resume.OUT/'training/summary.json').read_text())
    if result['completed_updates']!=1 or result['final_checkpoint']['iteration']!=201:
        raise ValueError('One-update smoke must produce exactly iter201')
    if not result['restore']['actor_critic_optimizer_exact']:
        raise ValueError('Native source weights and optimizer not restored')
    result.update(purpose='checkpoint_delivery_smoke_only',model_evaluations=0,num_envs=32,
        policy_quality_evaluated=False,controller_checkpoint_download_validation_required=True)
    resume.base.save(resume.OUT/'summary.json',result)
    resume.emit('delivery_smoke_training_saved',completed_updates=1,iteration=201,model_evaluations=0,
        policy_quality_evaluated=False,sha256=result['final_checkpoint']['sha256'])

def configure():
    # Run identically in parent and runtime child; no torch/CUDA import before boundary activation.
    resume.OUT=resume.ROOT/'outputs/delivery_smoke'
    resume.MAX_UPDATES=1
    resume.TRAIN_SECONDS=300
    resume.WORKER_SECONDS=1800
    resume.ARMS=(('training',0.),)
    resume.SEEDS=()
    resume.summarize=summarize

def main():
    configure()
    # Re-enter this same bounded wrapper in the Isaac child, NEVER the 200-update entrypoint.
    resume.main(child_entrypoint=Path(__file__))

if __name__=='__main__':main()
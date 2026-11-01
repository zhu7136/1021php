"""New-process continuation; retain the audited native simulator lifecycle."""
from pathlib import Path

def main():
    source=Path(__file__).with_name('three_teacher_child.py').read_text()
    changes={
      "'turn-id','train-deadline'):":"'turn-id','train-deadline','resume-checkpoint'):",
      "cfg=ExperimentConfig(**raw)":"raw['algo']=DistillationPPOAlgoConfig(_target_='three_adapter.native_resume.HuD04ThreeTeacherResume',_recursive_=False,config=__import__('dataclasses').replace(raw['algo'].config,num_learning_iterations=1000))\n    raw['training'].update(num_envs=3000,checkpoint=a.resume_checkpoint,teacher_checkpoint=None,name='hud04-three-teacher-student-resume-3000env')\n    cfg=ExperimentConfig(**raw)",
      "cfg.algo.config.num_learning_iterations!=5 or cfg.training.checkpoint or cfg.training.multigpu":"cfg.algo.config.num_learning_iterations!=1000 or cfg.training.checkpoint!=a.resume_checkpoint or cfg.training.multigpu",
      "cfg.training.num_envs!=6":"cfg.training.num_envs!=3000",
      "Authorized six-env/five-update/fresh/single-GPU scope changed":"Authorized 3000-env/1000-update/resume/single-GPU scope changed",
      "HUD04_THREE_GPU_EXECUTION_APPROVED='1',":"HUD04_RESUME_APPROVED='1000_updates_7200_seconds_once',HUD04_THREE_GPU_EXECUTION_APPROVED='1',",
      "num_envs=6,max_updates=5,training_seconds_max=600":"num_envs=3000,max_updates=1000,training_seconds_max=6600",
      "fresh_student=True":"fresh_student=False,source_sha256='7828ba3698eb1607585559ff22f569cc94a3c6c44137473ef57df19e4fd52492',trajectory_exact_resume=False",
      "event='hud04_native_validation_started',num_envs=6,max_updates=5":"event='hud04_native_resume_started',num_envs=3000,max_updates=1000"
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('Unexpected audited child source anchor')
        source=source.replace(old,new,1)
    exec(compile(source,'audited_child_with_resume_config','exec'),{'__file__':__file__,'__name__':'__main__'})

if __name__=='__main__':main()
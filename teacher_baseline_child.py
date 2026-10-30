"""Same fixed high-step scene and lifecycle; only the actor is the frozen teacher."""
from dataclasses import replace
import offline_video_child as original
from obstacle_video_child import configuration as obstacle_configuration

def configuration(request):
    cfg=obstacle_configuration(request)
    if request['scope']['skills']!=['high-step'] or cfg.observation.groups['teacher_obs'].enable_noise:
        raise ValueError('Fixed high-step and noise-free privileged observations required')
    return replace(cfg,algo=replace(cfg.algo,_target_='teacher_baseline_algo.HuD04TeacherBaseline'))

if __name__=='__main__':
    original.configuration=configuration
    original.main()
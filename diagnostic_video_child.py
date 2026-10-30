"""Use the same fixed-clip, aligned-terrain child with added telemetry only."""
import offline_video_child as original
from obstacle_video_child import configuration as obstacle_configuration

def configuration(request):
    cfg=obstacle_configuration(request)
    from dataclasses import replace
    return replace(cfg,algo=replace(cfg.algo,_target_='diagnostic_video_algo.HuD04DiagnosticVideo'))

if __name__=='__main__':
    original.configuration=configuration
    original.main()

"""Single-episode child, invoked in the existing pinned IsaacSim worker.

Requires the existing hash-verified obstacle request/materialization path.
Not a standalone downloader or a GPU-task submission command.
"""
import hashlib
from dataclasses import replace
from pathlib import Path
import offline_video_child as lifecycle
from obstacle_video_child import configuration as obstacle_configuration


def configuration(request):
    if request.get('evaluation_role') not in ('student', 'teacher'):
        raise ValueError('Explicit evaluation_role required')
    if type(request.get('evaluation_seed')) is not int or not 0 <= request['evaluation_seed'] <= 2:
        raise ValueError('Bounded evaluation seed required')
    if hashlib.sha256(Path(request['checkpoint']).read_bytes()).hexdigest() != request['source_checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    cfg = obstacle_configuration(request)
    return replace(cfg,
        algo=replace(cfg.algo, _target_='hud04_single_episode_algo.HuD04SingleEpisode'),
        training=replace(cfg.training, seed=request['evaluation_seed']),
        logger=replace(cfg.logger, headless_recording=False,
                       video=replace(cfg.logger.video, enabled=False)))


if __name__ == '__main__':
    lifecycle.configuration = configuration
    lifecycle.main()
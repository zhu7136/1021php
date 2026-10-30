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
    # Keep all original reset/step/termination implementations and thresholds;
    # only schedule current-frame native pose-cache refresh before termination.
    command = replace(cfg.command, **{
        group: {name: replace(term, func='hud04_cache_fixed_command:HuD04CacheSynchronizedCommand')
                if name == 'motion_command' else term
                for name, term in getattr(cfg.command, group).items()}
        for group in ('setup_terms', 'reset_terms', 'step_terms')})
    prepared = replace(cfg,
        command=command,
        algo=replace(cfg.algo, _target_='hud04_single_episode_algo.HuD04SingleEpisode'),
        training=replace(cfg.training, seed=request['evaluation_seed']),
        logger=replace(cfg.logger, headless_recording=False,
                       video=replace(cfg.logger.video, enabled=False)))
    # No change without an explicit depth diagnostic request. Selecting this
    # option prepares a config; it grants no task/publication authorization.
    from hud04_depth_diagnostics import configure_depth_diagnostics
    return configure_depth_diagnostics(prepared, request)


if __name__ == '__main__':
    lifecycle.configuration = configuration
    lifecycle.main()
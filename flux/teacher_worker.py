"""Single-GPU teacher process using Flux's existing Python, without Conda setup."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--approval-file', type=Path, required=True)
    args = parser.parse_args()
    from flux.preflight import runtime_checks
    print(json.dumps({'event': 'runtime_preflight', 'result': runtime_checks(args.data)}), flush=True)
    # Import configuration, then let the upstream launcher initialize the simulator
    # before it imports the PPO implementation, matching PHP's training entry.
    from wbt_training.config_values.experiment import g1_wbt_terrain_ref_noscale
    from wbt_training.utils.isaacsim_clock import use_php_isaacsim
    from holosoma.utils.sim_utils import setup_simulator_imports
    from holosoma.train_agent import train
    cfg = g1_wbt_terrain_ref_noscale
    cfg = replace(cfg,
        training=replace(cfg.training, headless=True, num_envs=args.num_envs,
                         checkpoint=None, teacher_checkpoint=None, wandb_path=None,
                         registry_name=args.data.resolve().as_uri(),
                         project='HuD04_WBT', name='php-locomotion-teacher-smoke'),
        logger=replace(cfg.logger, mode='offline', base_dir=str(args.output),
                       project='HuD04_WBT', entity='zhul136-university-of-science-and-technology-of-china',
                       name='php-locomotion-teacher-smoke', video=replace(cfg.logger.video, enabled=False)),
        algo=replace(cfg.algo, _target_='flux.teacher_algo.BoundedPPO',
                     config=replace(cfg.algo.config, num_learning_iterations=args.updates, save_interval=5)))
    os.environ['PHP_FLUX_OUTPUT'] = str(args.output)
    os.environ['WANDB_MODE'] = 'offline'
    # The upstream wandb.init config gets the non-secret ARIA provenance.
    import wandb
    original_init = wandb.init
    approval = json.loads(args.approval_file.read_text())
    def init_with_provenance(*a, **kw):
        config = dict(kw.get('config') or {})
        if approval.get('aria_provenance'):
            config['_wb_agent'] = approval['aria_provenance']
        kw['config'] = config
        return original_init(*a, **kw)
    wandb.init = init_with_provenance
    cfg = use_php_isaacsim(cfg)
    sys.argv = [sys.argv[0]]
    setup_simulator_imports(cfg)
    train(cfg)


if __name__ == '__main__':
    main()
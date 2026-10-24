"""Bounded, opt-in Flux entrypoint for the recovered HU_D04 terrain teacher."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = '4a0bf04cde5930af5ed59da797568d32be972525'
SUBMODULE = ROOT / 'thirdparty/holosoma'
PACKAGE = SUBMODULE / 'src/holosoma/holosoma'


def inspect_sources() -> dict:
    manifest = json.loads((ROOT / 'flux/recovered-manifest.json').read_text())
    repair_path=ROOT/'flux/repair-manifest.json'
    repairs=json.loads(repair_path.read_text())['files'] if repair_path.is_file() else {}
    for rel, expected in manifest.items():
        if rel in repairs:
            if repairs[rel]['original_sha256'] != expected:
                raise ValueError(f'Repair provenance disagrees with original manifest: {rel}')
            expected=repairs[rel]['repaired_sha256']
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Recovered file missing or changed: {rel}')
    for rel,entry in repairs.items():
        path=ROOT/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['repaired_sha256']:
            raise ValueError(f'Repaired source missing or changed: {rel}')
    data = ROOT / 'flux/data/motions-low-step-hu-d04'
    motions = sorted(data.glob('*_motion.npz'))
    if len(motions) != 56:
        raise ValueError(f'Expected 56 motions; got {len(motions)}')
    for path in motions:
        if not path.with_name(path.name.replace('_motion.npz', '_terrain.npy')).is_file():
            raise ValueError(f'Missing terrain pair for {path.name}')
    if not (PACKAGE / 'train_agent.py').is_file():
        raise ValueError('Missing Holosoma submodule. Run git submodule update --init thirdparty/holosoma')
    # Verify the source pin without depending on .git metadata in uploaded jobs.
    expected = json.loads((ROOT / 'flux/holosoma-base.json').read_text())
    for rel, digest in expected.items():
        if hashlib.sha256((SUBMODULE / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Holosoma differs from expected revision {PIN}: {rel}')
    for path in [ROOT/'wbt_training/config_values/experiment.py', ROOT/'wbt_training/config_values/hu_d04/experiment.py']:
        ast.parse(path.read_text())
    return {'recovered_files': len(manifest), 'motion_terrain_pairs': len(motions), 'holosoma_revision': PIN,'repair_files':len(repairs),'original_dataset_training_allowed':False}


def require_candidate_dataset(args) -> Path:
    if args.dataset is None or not args.acknowledge_kinematic_candidate:
        raise ValueError('Training blocked: supply --dataset and --acknowledge-kinematic-candidate after simulator review. Original G1 data is not HU_D04 training data.')
    dataset=args.dataset.resolve()
    if dataset==(ROOT/'flux/data/motions-low-step-hu-d04').resolve():
        raise ValueError('Training on recovered G1 references is blocked; use a validated HU_D04 candidate bundle.')
    sys.path.insert(0,str(ROOT/'flux'))
    from validate_hud04 import validate
    report=validate(dataset)
    if not report['all_kinematic_gates_pass']:
        raise ValueError('Training blocked: all 56 motions must pass independent kinematic gates. This does not certify collisions or dynamics.')
    return dataset


def prepare_runtime() -> Path:
    runtime = ROOT / '.flux-runtime'
    destination = runtime / 'holosoma'
    shutil.copytree(PACKAGE, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT/'flux/holosoma_overlay', destination, dirs_exist_ok=True)
    return runtime


def runtime_child(args) -> None:
    deps = ROOT / '.flux-deps'
    if args.install_deps:
        # Fail closed: missing/corrupt local wheels never fall back to PyPI.
        from offline_dependencies import install
        install(ROOT)
    runtime = prepare_runtime()
    sys.path[:0] = [str(deps), str(ROOT), str(runtime)]
    os.chdir(ROOT)
    import numpy as np
    from wbt_training.utils.registry import validate_bundle
    dataset = require_candidate_dataset(args) if args.train else (args.dataset or ROOT / 'flux/data/motions-low-step-hu-d04').resolve()
    validate_bundle(dataset)
    from holosoma.config_values.robot import hu_d04
    for path in sorted(dataset.glob('*_motion.npz')):
        with np.load(path, allow_pickle=False) as motion:
            if motion['joint_names'].tolist() != hu_d04.dof_names or motion['body_names'].tolist() != hu_d04.body_names:
                raise ValueError(f'Robot/data ordering mismatch: {path.name}')
    if not args.train:
        sys.argv = ['wbt_training.train_agent', '--help']
        runpy.run_module('wbt_training.train_agent', run_name='__main__')
        return
    from startup_diagnostics import install as install_startup_diagnostics
    install_startup_diagnostics(ROOT)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; refusing CPU fallback')
    if args.logger == 'wandb':
        if not os.environ.get('WANDB_API_KEY'):
            raise RuntimeError('Inject WANDB_API_KEY securely in Flux before online logging')
        import wandb
        original_init = wandb.init
        def init_with_provenance(*pos, **kwargs):
            provenance = {k: os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'), ('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
            if provenance:
                kwargs['config'] = dict(kwargs.get('config') or {}, _wb_agent=provenance)
            return original_init(*pos, **kwargs)
        wandb.init = init_with_provenance
    sys.argv = [
        'wbt_training.train_agent', 'exp:hu_d04_wbt_terrain_ref_noscale', 'simulator:isaacsim',
        f'logger:{args.logger}', '--training.headless', 'True',
        '--training.registry-name', dataset.as_uri(), '--training.num-envs', str(args.num_envs),
        '--training.name', args.name, '--logger.video.enabled', 'False',
        '--algo.config.num-learning-iterations', str(args.iterations),
        '--algo.config.save-interval', str(min(args.iterations, 10)),
    ]
    if args.logger == 'wandb':
        sys.argv += ['--logger.project', 'HuD04_WBT', '--logger.entity', 'zhul136-university-of-science-and-technology-of-china']
    print(json.dumps({'event':'training_start','iterations':args.iterations,'num_envs':args.num_envs,'logger':args.logger}), flush=True)
    runpy.run_module('wbt_training.train_agent', run_name='__main__')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect', action='store_true', help='Verify source/data hashes only; no runtime or GPU')
    mode.add_argument('--check-config', action='store_true', help='Validate data and load training --help; no training')
    mode.add_argument('--train', action='store_true', help='Explicitly authorize this process to start training')
    parser.add_argument('--runtime-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--install-deps', action='store_true', help='Install missing Python runtime dependencies into .flux-deps')
    parser.add_argument('--dataset',type=Path,help='Separate HU_D04 candidate motion/terrain directory; never overwrite recovered sources')
    parser.add_argument('--acknowledge-kinematic-candidate',action='store_true',help='Acknowledge kinematic checks are not collision/dynamics validation; controlled simulation only')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--time-limit-seconds', type=int, default=900, help='Whole-process limit, including setup')
    parser.add_argument('--logger', choices=['disabled','wandb'], default='disabled')
    parser.add_argument('--name', default='hud04-flux-smoke')
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.time_limit_seconds) < 1:
        parser.error('iterations, num-envs and time-limit-seconds must be positive')
    print(json.dumps(inspect_sources()), flush=True)
    if args.inspect or not (args.check_config or args.train):
        return
    if args.train and not args.runtime_child:
        require_candidate_dataset(args)
    if args.runtime_child:
        runtime_child(args)
        return
    from runtime_preflight import resolve_isaac_launcher
    launcher = resolve_isaac_launcher()
    print(json.dumps({'event': 'isaac_launcher_selected', 'path': str(launcher)}), flush=True)
    child_env = os.environ.copy()
    child_env['PYTHONUNBUFFERED'] = '1'
    child_env['CUDA_VISIBLE_DEVICES'] = '0'
    child = subprocess.Popen([str(launcher), str(Path(__file__).resolve()), '--runtime-child', *sys.argv[1:]], env=child_env, start_new_session=True)
    try:
        from startup_diagnostics import wait_for_child
        raise SystemExit(wait_for_child(child, ROOT, args.time_limit_seconds))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise SystemExit('Configured time limit reached; this launcher terminated its own job process group.')


if __name__ == '__main__':
    main()
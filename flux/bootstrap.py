"""Flux gm-run entry: no GPU work without an explicit approval file.

Check only: python flux/bootstrap.py --check
If Flux clones as 1021php, use gm-run 1021php/flux/bootstrap.py.
Running also requires explicit --approval-file, --data and --output arguments.
Prepare input shards separately; see flux/offline-inputs/README_ZH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux.preflight import static_checks


def load_approval(path: Path) -> dict:
    scope = json.loads(path.read_text())
    if scope.get('gpu_start_authorized') is not True:
        raise ValueError('GPU start has not been authorized')
    if scope.get('trial_count') != 1 or scope.get('automatic_retries') != 0:
        raise ValueError('Only one attempt, no automatic retries, is supported')
    if scope.get('gpu') != '1x4090D-24G':
        raise ValueError('Unexpected resource authorization')
    for key in ('billing_currency', 'billing_unit', 'billing_acceptance', 'code_commit'):
        if not scope.get(key):
            raise ValueError(f'Unconfirmed approval field: {key}')
    for key, upper in [('worker_seconds', 1200), ('num_envs', 128), ('updates', 20)]:
        value = scope.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f'{key} outside prepared smoke scope')
    if scope['worker_seconds'] < 180:
        raise ValueError('At least 180 seconds are required for stop/save reserves')
    if scope.get('output_retrieval_verified') is not True:
        raise ValueError('Checkpoint retrieval path must be verified first')
    return scope


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--approval-file', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checks = static_checks()
    if args.check:
        print(json.dumps({'static_checks': checks, 'gpu_started': False}, indent=2))
        return 0
    if not args.approval_file or not args.data or not args.output:
        parser.error('--approval-file, --data and --output are required to run')
    scope = load_approval(args.approval_file)
    if checks['php_commit'] != scope['code_commit']:
        raise ValueError('Worker commit does not match approved commit')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Never reuse output from another attempt or silently resume a checkpoint.
    with (output / 'attempt.json').open('x') as stream:
        json.dump({'commit': scope['code_commit'], 'scope': scope}, stream, indent=2)
    env = os.environ.copy()
    # This initial recipe logs W&B offline; no general account key is needed.
    for name in ('WANDB_API_KEY', 'OPENAI_API_KEY', 'FLUX_API_KEY', 'GITHUB_TOKEN', 'github', 'key'):
        env.pop(name, None)
    env.update(WANDB_MODE='offline', WANDB_ENTITY='zhul136-university-of-science-and-technology-of-china',
               WANDB_PROJECT='HuD04_WBT', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    soft_deadline = started + scope['worker_seconds'] - 120
    hard_deadline = started + scope['worker_seconds'] - 30
    if time.monotonic() >= soft_deadline:
        raise TimeoutError('Preflight consumed initialization budget; training was not started')
    command = [sys.executable, str(ROOT / 'flux/teacher_worker.py'), '--data', str(args.data.resolve()),
               '--output', str(output), '--num-envs', str(scope['num_envs']), '--updates', str(scope['updates']),
               '--approval-file', str(args.approval_file.resolve())]
    timed_out = False
    child = None
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        while child.poll() is None:
            now = time.monotonic()
            if now >= soft_deadline:
                (output / 'STOP_AFTER_UPDATE').touch(exist_ok=True)
            if now >= hard_deadline:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                break
            time.sleep(1)
    finally:
        if child is not None:
            # Clean up the process group even if the main child exited first.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    progress_path = output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    checkpoints = [{'path': str(path.relative_to(output)), 'size': path.stat().st_size, 'sha256': sha256(path)}
                   for path in sorted(output.rglob('*.pt')) if path.is_file() and path.stat().st_size > 0]
    completed = progress.get('completed_updates', 0)
    passed = child.returncode == 0 and not timed_out and completed == scope['updates'] and bool(checkpoints)
    result = {'pipeline_completed': passed, 'requested_updates': scope['updates'],
              'completed_updates': completed, 'child_exit_code': child.returncode,
              'hard_timeout': timed_out, 'elapsed_seconds': round(time.monotonic()-started, 2),
              'checkpoints': checkpoints, 'checkpoint_reload_verified': False,
              'wandb_logging': 'offline; sync after verified retrieval',
              'acceptance_passed': False}
    (output / 'completion.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
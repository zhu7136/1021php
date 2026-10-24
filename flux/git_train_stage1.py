"""One authorized 3600s/one-GPU formal training stage, no restart/retry.

Remote worker receives only a signed one-file output PUT, never an API key.
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
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/formal_stage1'
WORKER_SECONDS = 3600
SOFT_SECONDS = 3400  # Reserve shutdown, packaging and one output upload.


def event(name, **fields):
    print(json.dumps(dict(event=name, **fields), sort_keys=True), flush=True)


def worker(started):
    import git_smoke_v8 as bootstrap
    from source_preflight import verify_sources
    from runtime_preflight import verify_runtime
    event('source_preflight_verified', **verify_sources(ROOT))
    event('runtime_preflight_verified', **verify_runtime(ROOT))
    from git_parts_v8 import assemble_parts
    archive = OUT / 'v8.zip'
    assemble_parts(ROOT / 'flux/data/hud04-v8-parts', archive,
                   bootstrap.ARCHIVE_BYTES, bootstrap.ARCHIVE_SHA256)
    dataset = OUT / 'candidates'
    bootstrap.extract_candidates(archive, dataset)
    from offline_dependencies import install
    install(ROOT)
    deps = ROOT / '.flux-deps'
    sys.path.insert(0, str(deps))
    os.environ['PYTHONPATH'] = str(deps) + os.pathsep + os.environ.get('PYTHONPATH', '')
    import train_hud04 as trainer
    runtime = trainer.prepare_runtime()
    sys.path[:0] = [str(ROOT), str(runtime)]
    from startup_diagnostics import install as diagnostics
    diagnostics(ROOT)
    from formal_training_hooks import install as hooks, ITERATION_CEILING, CHECKPOINT_SHA, digest
    checkpoint = ROOT / 'flux/replay-input/model_00001.pt'
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Mounted resume checkpoint mismatch')
    hooks(OUT, checkpoint, started + SOFT_SECONDS)
    event('formal_training_begin', worker_limit_seconds=WORKER_SECONDS,
        num_envs=32, motion_variants=32, iteration_safety_ceiling=ITERATION_CEILING,
        automatic_retries=0, checkpoint_input_sha256=CHECKPOINT_SHA)
    args = argparse.Namespace(train=True, dataset=dataset, acknowledge_kinematic_candidate=True,
        install_deps=False, logger='disabled', num_envs=32, motion_variants=32,
        iterations=ITERATION_CEILING, name='hud04-formal-stage1-1h')
    trainer.runtime_child(args)  # Same validated training path; no nested runtime process.


def seal_and_upload(url, headers):
    from formal_training_hooks import digest
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.username or parsed.password or
            not (parsed.hostname == 'storage.googleapis.com' or
                 str(parsed.hostname).endswith('.storage.googleapis.com'))):
        raise ValueError('Only scoped signed GCS output PUT accepted')
    if any(k.lower() != 'content-type' and not k.lower().startswith('x-goog-') for k in headers):
        raise ValueError('No API authorization may reach output upload')
    files = [p for p in OUT.glob('*.json*') if p.is_file()]
    summary_path = OUT / 'training_stage_completed.json'
    complete = summary_path.is_file()
    if complete:
        record = json.loads(summary_path.read_text())
        checkpoint = Path(record['final_checkpoint'])
        if not checkpoint.resolve().is_relative_to(ROOT.resolve()) or checkpoint.is_symlink():
            raise ValueError('Unsafe final checkpoint')
        if digest(checkpoint) != record['final_checkpoint_sha256']:
            raise ValueError('Final checkpoint digest mismatch')
        if checkpoint.stat().st_size != record['final_checkpoint_bytes'] or record['completed_updates'] < 1:
            raise ValueError('Empty training stage')
        files.append(checkpoint)
        config = checkpoint.parent / 'holosoma_config.yaml'
        if not config.is_file():
            raise ValueError('Training configuration not saved')
        files.append(config)
    bundle = ROOT / 'outputs/training_stage1_bundle.zip'
    if any(p.is_symlink() for p in files) or sum(p.stat().st_size for p in files) > 100 * 1024 * 1024:
        raise ValueError('Stage output exceeds bound')
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(files):
            name = path.name if path.parent == OUT else 'training/' + path.name
            target.write(path, arcname=name)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    request = urllib.request.Request(url, data=bundle.read_bytes(), headers=headers, method='PUT')
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        if response.status not in (200, 201):
            raise RuntimeError('Output upload rejected')
    event('formal_stage_bundle_uploaded', sha256=digest(bundle), bytes=bundle.stat().st_size,
        stage_complete=complete)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload-url')
    parser.add_argument('--upload-headers', default='{}')
    parser.add_argument('--runtime-child', action='store_true')
    parser.add_argument('--started', type=float)
    args = parser.parse_args()
    if args.runtime_child:
        if args.started is None:
            parser.error('Missing parent monotonic start')
        worker(args.started)
        return
    if not args.upload_url or OUT.exists():
        raise RuntimeError('Requires scoped upload and fresh stage directory; never restart')
    OUT.mkdir(parents=True)
    started = time.monotonic()
    child = None
    def deadline(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        raise TimeoutError('Approved 3600 second worker budget reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(WORKER_SECONDS)
    try:
        from runtime_preflight import resolve_isaac_launcher
        launcher = resolve_isaac_launcher()
        env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
        child = subprocess.Popen([str(launcher), str(Path(__file__).resolve()),
            '--runtime-child', '--started', str(started)], env=env, start_new_session=True)
        from startup_diagnostics import wait_for_child
        try:
            code = wait_for_child(child, ROOT, 3540)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            code = 124
        complete = seal_and_upload(args.upload_url, json.loads(args.upload_headers))
        if code != 0 or not complete:
            raise RuntimeError('Stage lacks successful child and sealed final checkpoint')
        event('formal_stage_entrypoint_complete', worker_limit_seconds=3600)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        signal.alarm(0)


if __name__ == '__main__':
    main()
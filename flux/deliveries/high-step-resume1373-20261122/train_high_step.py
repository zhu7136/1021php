"""Offline High Step full-state continuation; default is CPU preflight.

No provider credentials are accepted or needed. Upload delivery and GPU execution
remain unverified until Flux mounts this exact package and returns real evidence.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = 'd8aac2800e777c3a10bb20f8e30ff8d5c8fda83d2d2517cbb20aae716ac8cc07'
LIMIT = 1800


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), allow_nan=False), flush=True)


def check(root, rows):
    root = root.resolve()
    for row in rows:
        p = root / row['path']
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError('missing_or_escaping_input')
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('input_hash_mismatch')


def extract(p, target):
    with zipfile.ZipFile(p) as z:
        seen = set()
        total = 0
        for entry in z.infolist():
            q = PurePosixPath(entry.filename)
            total += entry.file_size
            if q.is_absolute() or '..' in q.parts or '\\' in entry.filename or entry.filename in seen:
                raise ValueError('unsafe_archive_path')
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('archive_symlink')
            if q.suffix.lower() in {'.pt', '.pth', '.onnx', '.ckpt', '.safetensors'}:
                raise ValueError('unexpected_inherited_weight')
            if total > 3 * 1024 ** 3:
                raise ValueError('archive_size_limit')
            seen.add(entry.filename)
        z.extractall(target)


def prepare(work, plan):
    check(ROOT, plan['files'])
    checkpoint = ROOT / 'source-checkpoint.pt'
    if sha(checkpoint) != SOURCE_SHA:
        raise ValueError('wrong_source_checkpoint')
    runtime = work / 'runtime'
    skill = work / 'skill'
    extract(ROOT / 'runtime.zip', runtime)
    extract(ROOT / 'high-step-input.zip', skill)
    payload = work / 'payload'
    payload.mkdir()
    for name in ['bundle', 'native-source', 'deps']:
        shutil.copytree(runtime / name, payload / name)
    for p in (skill / 'bundle').rglob('*'):
        if p.is_file():
            dest = payload / 'bundle' / p.relative_to(skill / 'bundle')
            if dest.exists():
                raise ValueError('immutable_bundle_collision')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.copytree(skill / 'teacher-code', payload / 'teacher-code')
    for name in ['config.json', 'teacher_child.py', 'teacher_algo.py', 'resume_audit.py', 'speed10_audit.py']:
        shutil.copyfile(ROOT / 'overrides' / name, payload / 'teacher-code' / name)
    shutil.copyfile(ROOT / 'overrides/hud04_robot.py', payload / 'bundle/overlay/wbt_training/config_values/hud04_robot.py')
    shutil.copyfile(ROOT / 'overrides/HU_D04_01.urdf', payload / 'bundle/asset/urdf/HU_D04_01.urdf')
    check(payload, plan['effective_overrides'])
    config = json.loads((payload / 'teacher-code/config.json').read_text())
    assert config['training']['num_envs'] == 4096
    assert config['training']['checkpoint'] == '/workspace/resume-input/model_smoke_final.pt'
    assert config['algo']['config']['load_optimizer'] is True
    assert config['algo']['config']['num_learning_iterations'] == 2000
    assert 'high-step' in config['training']['name']
    return payload, checkpoint


def seal(out, work):
    files = [p for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink()
             and (p.suffix in {'.json', '.jsonl', '.yaml', '.yml'}
                  or p.name in {'model_smoke_final.pt', 'model_latest.pt', 'model_previous.pt'})]
    if sum(p.stat().st_size for p in files) > 50 * 1024 ** 2:
        raise ValueError('result_archive_size_cap')
    manifest = {'files': [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p)) for p in files],
                'source_checkpoint_sha256': SOURCE_SHA, 'remote_retrieval_verified': False}
    (work / 'result-manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = work / 'high-step-result.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(work / 'result-manifest.json', 'result-manifest.json')
        for p in files:
            z.write(p, str(p.relative_to(out)))
    emit('result_archive_begin', bytes=archive.stat().st_size, sha256=sha(archive), chunk_bytes=12288)
    with archive.open('rb') as f:
        count = 0
        for b in iter(lambda: f.read(12288), b''):
            emit('result_archive_chunk', index=count, data=base64.b64encode(b).decode('ascii'))
            count += 1
    emit('result_archive_end', chunks=count, bytes=archive.stat().st_size, sha256=sha(archive), remote_retrieval_verified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--task-id')
    p.add_argument('--work', type=Path)
    a = p.parse_args()
    if a.execute and not re.fullmatch(r'TASK_\d{8}_\d+', a.task_id or ''):
        raise ValueError('task_identity_required')
    plan = json.loads((ROOT / 'PLAN.json').read_text())
    work = a.work or ROOT / ('work-' + (a.task_id or 'cpu-preflight'))
    work.mkdir(parents=True, exist_ok=False)
    out = work / 'output'
    out.mkdir()
    started = globals().get("OUTER_STARTED", time.monotonic())
    child = None
    report = dict(task_id=a.task_id, skill='high-step', source_iteration=1373,
                  source_checkpoint_sha256=SOURCE_SHA, fresh_training=False,
                  gpu_tasks_submitted=0, worker_seconds_max=LIMIT, update_cap=2000,
                  automatic_retries=0, num_envs=4096, completed_updates=0,
                  teacher_converged=False, worker_passed=False, native_lifecycle_passed=False)

    def deadline(sig, frame):
        raise TimeoutError('worker_budget_reached')

    if a.execute:
        signal.signal(signal.SIGALRM, deadline)
        signal.signal(signal.SIGTERM, deadline)
        signal.setitimer(signal.ITIMER_REAL, max(0.01, LIMIT - (time.monotonic() - started)))
    try:
        payload, checkpoint = prepare(work, plan)
        if not a.execute:
            emit('cpu_package_preflight_passed', gpu_started=False, native_gpu_resume_verified=False, **report)
            return 0
        vendor = next((Path(s) for s in ['/isaac-sim/python.sh', '/workspace/isaaclab/_isaac_sim/python.sh'] if Path(s).is_file()), None)
        if vendor is None:
            raise ValueError('vendor_interpreter_missing')
        # Use a small explicit allowlist; never forward account credentials.
        env = {k: os.environ[k] for k in ['PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'DISPLAY', 'ACCEPT_EULA', 'OMNI_KIT_ACCEPT_EULA'] if k in os.environ}
        env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
                   CUDA_VISIBLE_DEVICES='0', PHP_FLUX_TASK_ID=a.task_id, PHP_RESUME_CHECKPOINT=str(checkpoint))
        site = work / 'dependency-site'
        command = [str(vendor), '-m', 'pip', 'install', '--isolated', '--disable-pip-version-check', '--no-cache-dir',
                   '--no-index', '--only-binary=:all:', '--no-deps', '--require-hashes', '--find-links', str(payload / 'deps/wheels'),
                   '--target', str(site), '-r', str(payload / 'deps/overlay.lock')]
        with (out / 'install.log').open('xb') as log:
            rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
        if rc:
            raise ValueError('offline_install_failed')
        command = [str(vendor), str(payload / 'teacher-code/teacher_child.py'), '--payload', str(payload), '--output', str(out),
                   '--site', str(site), '--thread-id', plan['provenance']['thread_id'], '--turn-id', plan['provenance']['turn_id']]
        emit('simulator_start_requested', source_iteration=1373, gpu_training_verified=False)
        with (out / 'native.log').open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            position = 0
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= 1500:
                    (out / 'STOP_AFTER_UPDATE').touch(exist_ok=True)
                if elapsed >= 1670:
                    raise TimeoutError('native_child_deadline')
                progress = out / 'updates.jsonl'
                if progress.is_file():
                    with progress.open() as f:
                        f.seek(position)
                        for line in f:
                            if not line.endswith('\n'):
                                break
                            emit('training_progress', **json.loads(line))
                            position += len(line.encode())
                time.sleep(2)
        report['native_returncode'] = child.returncode
        learning_path = out / 'learning_state.json'
        if learning_path.is_file():
            learning = json.loads(learning_path.read_text())
            report['completed_updates'] = learning['completed_updates']
            cp = out / learning['checkpoint_relative_path']
            flags = ['actor_changed', 'critic_changed', 'actor_optimizer_steps_increased', 'critic_optimizer_steps_increased', 'parameters_finite', 'new_process_resume_verified']
            report['learning_evidence_passed'] = (1 <= report['completed_updates'] <= 2000 and all(learning.get(k) is True for k in flags)
                                                 and cp.is_file() and not cp.is_symlink() and cp.resolve().is_relative_to(out.resolve())
                                                 and sha(cp) == learning['checkpoint_sha256'])
        lifecycle_path = out / 'lifecycle.json'
        if lifecycle_path.is_file():
            life = json.loads(lifecycle_path.read_text())
            report['native_lifecycle_passed'] = (child.returncode == 0 and life.get('close_returned') is True and life.get('child_returned') is True)
        report['worker_passed'] = report.get('learning_evidence_passed') is True and report['native_lifecycle_passed']
    except BaseException as error:
        report['failure_type'] = type(error).__name__
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
        if a.execute:
            diagnostics = {}
            for name in ['native.log', 'install.log']:
                path = out / name
                if path.is_file():
                    with path.open('rb') as stream:
                        stream.seek(max(0, path.stat().st_size - 40000))
                        diagnostics[name] = stream.read().decode('utf8', errors='replace')
            (out / 'native_diagnostics.json').write_text(json.dumps(diagnostics))
            report['elapsed_seconds'] = round(time.monotonic() - started, 2)
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            seal(out, work)
            emit('worker_finished', **report)
            signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if report['worker_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
"""Generic 300-second private app diagnostic launcher; zero training/retries.

Only generic transport code is public. Scoped object tickets arrive privately.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile
from scoped_transport import deadline, digest, safe_unpack, transfer, validate_capability

SCOPE = {'worker_seconds_max': 300, 'native_cutoff_seconds': 240,
         'training_updates': 0, 'robot_environments': 0,
         'max_native_processes': 1, 'automatic_retries': 0, 'max_tasks': 1}


def unpack(archive, destination):
    safe_unpack(archive, destination)
    destination = Path(destination)
    manifest = json.loads((destination / 'payload-manifest.json').read_text())
    if manifest['scope'] != SCOPE:
        raise ValueError('Private diagnostic scope mismatch')
    expected = {'payload-manifest.json'}
    for row in manifest['files']:
        name = row['path']
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()) or name in expected:
            raise ValueError('Manifest path or duplicate')
        expected.add(name)
        if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError('Private code integrity failure')
    actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Unlisted private code')
    return manifest


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)


def main():
    # Entry time covers parsing, transfer, extraction, native app and delivery.
    started = time.monotonic()
    child = [None]
    lock = threading.Lock()
    def register(value):
        with lock:
            child[0] = value
    def hard_stop():
        time.sleep(max(0, started + 300 - time.monotonic()))
        with lock:
            process = child[0]
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os._exit(124)  # emergency FAILURE, never successful cleanup
    threading.Thread(target=hard_stop, daemon=True).start()
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError('Invalid private launcher arguments')
    p = Parser()
    for name in ('commit', 'input-cap', 'output-cap', 'input-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--input-bytes', required=True, type=int)
    a = p.parse_args()
    if not 0 < a.input_bytes < 1024 * 1024:
        raise ValueError('Minimal input archive only')
    caps = {k: json.loads(base64.urlsafe_b64decode(getattr(a, k + '_cap'))) for k in ('input', 'output')}
    for key, method in (('input', 'GET'), ('output', 'PUT')):
        validate_capability(caps[key], method)
    root = Path(__file__).resolve().parent
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=5).strip() != a.commit:
        raise ValueError('Generic code commit mismatch')
    work = root / 'private-probe'
    work.mkdir(exist_ok=False)
    out = work / 'output'
    out.mkdir()
    state = {'worker_passed': False, 'pipeline_passed': False, 'stage': 'input',
             'scope': SCOPE, 'code_commit': a.commit, 'input_sha256': a.input_sha256}
    try:
        emit('probe_stage', stage='input', worker_seconds_max=300, training_updates=0)
        transfer(caps['input'], work / 'input.zip', method='GET', size=a.input_bytes,
                 sha256=a.input_sha256, absolute_deadline=started + 60)
        with deadline(started + 65):
            unpack(work / 'input.zip', work / 'payload')
        state['stage'] = 'native_lifecycle'
        emit('probe_stage', stage='native_lifecycle', training_updates=0)
        payload = work / 'payload'
        sys.path.insert(0, str(payload))
        spec = importlib.util.spec_from_file_location('private_probe_worker', payload / 'probe_worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # No subprocess worker or installer: watchdog tracks the native process
        # group directly, including a vendor shell and its interpreter children.
        result = module.run(payload, out, started=started, on_spawn=register)
        state['worker_passed'] = result.get('native_lifecycle_passed') is True
        state['stage'] = 'complete' if state['worker_passed'] else 'native_lifecycle_failed'
    except Exception as exc:
        state['error_type'] = type(exc).__name__
        emit('probe_failure', stage=state['stage'], error_type=type(exc).__name__)
    state['elapsed_seconds'] = time.monotonic() - started
    (out / 'bootstrap-result.json').write_text(json.dumps(state, indent=2))
    with deadline(started + 270):
        files = []
        for path in sorted(out.rglob('*')):
            if path.is_symlink():
                raise ValueError('Result symlink')
            if path.is_file():
                files.append({'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
        if sum(r['bytes'] for r in files) > 32 * 1024 ** 2:
            raise ValueError('Result bound')
        (out / 'result-manifest.json').write_text(json.dumps({'files': files, 'scope': SCOPE}, sort_keys=True))
        archive = work / 'result.zip'
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(out))
    receipt = transfer(caps['output'], archive, method='PUT', size=archive.stat().st_size,
                       sha256=digest(archive), absolute_deadline=started + 295)
    emit('probe_result_uploaded', **receipt, worker_passed=state['worker_passed'])
    return 0 if state['worker_passed'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        emit('probe_bootstrap_failure', error_type=type(exc).__name__)
        sys.exit(1)
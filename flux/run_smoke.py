"""Exactly one authorized 128env/20update Flux smoke; no account credentials."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = None
CHILD = None


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def decode_control(value):
    """A whitespace-free CLI token survives gm-run's command forwarding."""
    decoded = base64.b64decode(value, validate=True)
    if len(decoded) > 8192:
        raise ValueError('Oversized control parameter')
    result = json.loads(decoded)
    if not isinstance(result, dict):
        raise ValueError('Control parameter must be an object')
    return result


def task_output(task_id):
    if not re.fullmatch(r'TASK_\d{8}_\d+', task_id):
        raise ValueError('Invalid task identifier')
    return ROOT / 'outputs' / ('php-smoke-' + task_id)


def upload(url, headers, path):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'storage.googleapis.com':
        raise ValueError('Only the approved scoped W&B object store upload is accepted')
    clean = dict(headers)
    if any(key.lower() in ('authorization', 'cookie') for key in clean):
        raise ValueError('No account-level upload credentials allowed')
    clean['Content-Length'] = str(path.stat().st_size)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=40)
    try:
        with path.open('rb') as stream:
            connection.request('PUT', parsed.path + '?' + parsed.query, body=stream, headers=clean)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise RuntimeError('Scoped object upload returned HTTP ' + str(response.status))
            response.read(1024)
    finally:
        connection.close()


def launcher():
    for path in (Path('/isaac-sim/python.sh'), Path('/workspace/isaaclab/_isaac_sim/python.sh')):
        if path.is_file() and os.access(path, os.X_OK) and (path.parent / 'kit/python/bin/python3').is_file():
            return path
    raise ValueError('Verified vendor Python launcher is missing; no system-Python fallback')


def stop_child():
    global CHILD
    if CHILD is not None:
        try:
            os.killpg(CHILD.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            CHILD.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        CHILD = None


def run(command, env, timeout, log_path):
    global CHILD
    # Child argv never includes scoped upload URLs or account credentials.
    # Persist stderr with stdout: the platform console can truncate scan noise.
    with log_path.open('xb') as log:
        CHILD = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            code = CHILD.wait(timeout=timeout)
            if code:
                raise RuntimeError('Child exited with code ' + str(code) + '; see ' + log_path.name)
        finally:
            stop_child()


def main():
    global OUT
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--upload-url', required=True)
    parser.add_argument('--probe-url', required=True)
    parser.add_argument('--upload-headers-b64', default='e30=')
    parser.add_argument('--aria-provenance-b64', default='e30=')
    args = parser.parse_args()
    # Parse control tokens BEFORE any setup/download subprocess or simulator.
    headers = decode_control(args.upload_headers_b64)
    provenance = decode_control(args.aria_provenance_b64)
    if not set(provenance).issubset({'thread_id','turn_id'}) or not all(isinstance(v,str) for v in provenance.values()):
        raise ValueError('Invalid provenance fields')
    OUT = task_output(args.task_id)
    # Claim once before any preparation; no automatic reuse or resume.
    OUT.mkdir(parents=True, exist_ok=False)
    def deadline(_signum, _frame):
        stop_child()
        raise TimeoutError('Bounded worker deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1140)  # Total worker cap1200s, reserve up to60s for sealing.
    success = False
    stage = 'source_preflight'
    emit('smoke_begin', num_envs=128, updates=20, worker_seconds=1200,
         automatic_retries=0, terrain_variants=64, onnx_export=False)
    try:
        actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != args.expected_commit:
            raise ValueError('Worker commit does not match authorized commit')
        sys.path.insert(0, str(ROOT))
        from flux.preflight import static_checks
        checks = static_checks(expected_commit=args.expected_commit)
        (OUT/'source-preflight.json').write_text(json.dumps(checks, indent=2))
        emit('source_preflight_passed', verification=checks['source_verification'],
             shallow_checkout=checks['shallow_checkout'])
        python = launcher()
        probe = OUT / 'worker-upload-probe.json'
        probe.write_text(json.dumps({'task_id': args.task_id, 'commit': actual, 'gpu_training_started': False}))
        upload(args.probe_url, headers, probe)
        emit('scoped_upload_probe_passed')
        stage = 'runtime_version_preflight'
        env = os.environ.copy()
        for name in ('WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY','GITHUB_TOKEN','github','key'):
            env.pop(name, None)
        env.update(PYTHONUNBUFFERED='1', WANDB_MODE='offline', CUDA_VISIBLE_DEVICES='0', PHP_SMOKE_OUTER_GROUP='1')
        # Use the actual image interpreter before any wheel download/install.
        # This stdlib-only probe never imports a native simulator or Torch.
        run([str(python), str(ROOT/'flux/runtime_versions.py')], env, 30, OUT/'runtime-versions.log')
        emit('runtime_version_preflight_passed')
        stage = 'pinned_upstream_wheels'
        private = OUT / 'upstream-wheels'
        private.mkdir()
        rows = json.loads((ROOT / 'flux/smoke-upstream-wheels.json').read_text())
        if len(rows) != 8:
            raise ValueError('Unexpected upstream wheel set')
        def download(row):
            if not row['url'].startswith('https://files.pythonhosted.org/packages/') or Path(row['file']).name != row['file']:
                raise ValueError('Unapproved upstream wheel destination')
            path = private / row['file']
            with urllib.request.urlopen(row['url'], timeout=45) as incoming, path.open('xb') as output:
                remaining = row['bytes']
                while remaining:
                    chunk = incoming.read(min(1024*1024, remaining))
                    if not chunk:
                        raise ValueError('Incomplete wheel download')
                    output.write(chunk)
                    remaining -= len(chunk)
                if incoming.read(1):
                    raise ValueError('Oversized wheel download')
            if digest(path) != row['sha256']:
                raise ValueError('Wheel hash mismatch')
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(download, rows))
        emit('upstream_wheels_verified', count=8)
        stage = 'reconstruct_and_install'
        inputs = OUT / 'inputs'
        run([str(python), str(ROOT/'flux/prepare_inputs.py'), '--output', str(inputs), '--install',
             '--python', str(python), '--withheld-wheel-dir', str(private)], env, 240, OUT/'prepare.log')
        env['PYTHONPATH'] = os.pathsep.join([str(inputs/'.flux-deps'), str(ROOT),
            str(ROOT/'thirdparty/holosoma/src/holosoma'), env.get('PYTHONPATH','')])
        emit('data_and_dependencies_installed')
        remaining = min(1050, int(1110-(time.monotonic()-started)))
        if remaining < 180:
            raise TimeoutError('Insufficient remaining budget for simulator and updates')
        scope = {'gpu_start_authorized': True, 'trial_count': 1, 'automatic_retries': 0,
                 'gpu': '1x4090D-24G', 'worker_seconds': remaining, 'num_envs':128,'updates':20,
                 'billing_unit_waived_by_user': True,
                 'billing_acceptance':'User authorized exactly one bounded smoke and waived currency/unit as a gate',
                 'code_commit': actual, 'output_upload_probe_passed': True,
                 'output_retrieval_verified': False, 'aria_provenance':provenance}
        approval = OUT / 'approval.json'
        approval.write_text(json.dumps(scope, indent=2))
        stage = 'simulation_and_updates'
        run([str(python), str(ROOT/'flux/bootstrap.py'), '--approval-file', str(approval),
             '--data', str(inputs/'motions-locomotion'), '--output', str(OUT/'training')], env, remaining, OUT/'bootstrap.log')
        completion = json.loads((OUT/'training/completion.json').read_text())
        reload_result = json.loads((OUT/'training/native_reload.json').read_text())
        if not completion['pipeline_completed'] or completion['completed_updates'] != 20 or not reload_result['native_load_passed']:
            raise ValueError('Smoke acceptance failed')
        success = True
        emit('smoke_worker_passed', completed_updates=20, native_reload=True)
    except BaseException as exc:
        # Do not render exceptions/arguments containing scoped signed URLs.
        (OUT/'failure.json').write_text(json.dumps({'stage':stage,'exception_type':type(exc).__name__,
            'elapsed_seconds':round(time.monotonic()-started,2),'automatic_retries':0,
            'frames':[{'file':Path(frame.filename).name,'line':frame.lineno,'function':frame.name}
                      for frame in traceback.extract_tb(exc.__traceback__)]}))
        emit('smoke_failed', stage=stage, exception_type=type(exc).__name__)
    finally:
        stop_child()
        signal.alarm(max(1, int(1198-(time.monotonic()-started))))
        manifest = []
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.pt','.yaml','.yml','.log') and 'inputs' not in path.relative_to(OUT).parts:
                manifest.append({'path':str(path.relative_to(OUT)),'bytes':path.stat().st_size,'sha256':digest(path)})
        (OUT/'result-manifest.json').write_text(json.dumps({'worker_passed':success,'files':manifest},indent=2))
        bundle = OUT / 'php-smoke-result.zip'
        with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.write(OUT/'result-manifest.json', 'result-manifest.json')
            for row in manifest:
                archive.write(OUT/row['path'], row['path'])
        try:
            upload(args.upload_url, headers, bundle)
            emit('smoke_result_uploaded', bytes=bundle.stat().st_size, sha256=digest(bundle))
        except BaseException as exc:
            success = False
            emit('smoke_result_upload_failed', exception_type=type(exc).__name__)
        signal.alarm(0)
    emit('smoke_entrypoint_complete', passed=success, elapsed_seconds=round(time.monotonic()-started,2))
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
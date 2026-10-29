"""Bounded native receipt collection; NOT a scheduler or full worker bootstrap.

Caller must authenticate code/inputs, start the whole-worker watchdog at entry,
and seal/upload/read back all results. A local native pass alone is not delivery
success or GPU resource release. Only our own explicitly created process group
can be killed on timeout.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from .lifecycle_probe import validate


def capture_process(command, *, log, deadline, env, cwd, on_spawn=None, on_tick=None):
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise TimeoutError('No native time remaining; no subprocess started')
    receipt = {'raw_returncode': None, 'timed_out': False, 'process_started': False,
               'native_lifecycle_passed': False}
    with Path(log).open('xb') as stream:
        try:
            child = subprocess.Popen(command, env=env, cwd=cwd, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            receipt['startup_error_type'] = type(exc).__name__
            return receipt
        receipt['process_started'] = True
        if on_spawn is not None:
            on_spawn(child)
        try:
            if on_tick is None:
                code = child.wait(timeout=max(.001, deadline - time.monotonic()))
            else:
                while True:
                    if on_tick is not None:
                        try:on_tick()
                        except Exception as exc:
                            receipt['telemetry_error_type']=type(exc).__name__
                            print(json.dumps({'event':'training_telemetry_failed','error_type':type(exc).__name__}),flush=True)
                            on_tick=None
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise subprocess.TimeoutExpired(command,0)
                    try:
                        code=child.wait(timeout=min(.5,remaining))
                        if on_tick is not None:
                            try:on_tick()
                            except Exception as exc:receipt['telemetry_error_type']=type(exc).__name__
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic()>=deadline:raise
        except subprocess.TimeoutExpired:
            receipt['timed_out'] = True
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                code = child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = None
                receipt['reap_timed_out'] = True
        receipt['raw_returncode'] = code
        if on_spawn is not None:
            on_spawn(None)
    return receipt


def assess(output, request, receipt):
    """Verify local lifecycle evidence; transport is a separate gate."""
    if receipt.get('process_started') is not True or receipt.get('timed_out') is not False or type(receipt.get('raw_returncode')) is not int or receipt['raw_returncode'] != 0:
        return False
    try:
        rows = [json.loads(line) for line in (Path(output) / 'probe-events.jsonl').read_text().splitlines()]
        expected = ['probe_started', 'app_create_started', 'app_created', 'fast_shutdown_verified']
        for _ in range(request['application_updates']):
            expected += ['update_started', 'update_returned']
        expected += ['before_close', 'close_returned', 'body_returned', 'main_returning']
        if [r['event'] for r in rows] != expected:
            return False
        first = rows[0]
        if first['backend'] != 'native_isaacsim' or first['task_id'] != request['task_id'] or first['_wb_agent'] != request['_wb_agent'] or first['settings'] != request['settings']:
            return False
        if type(first['training_updates']) is not int or first['training_updates'] != 0 or type(first['robot_environments']) is not int or first['robot_environments'] != 0:
            return False
        stamps = [r['monotonic_seconds'] for r in rows]
        if not all(type(t) in (int, float) and math.isfinite(t) for t in stamps) or any(b < a for a, b in zip(stamps, stamps[1:])):
            return False
        before = next(r for r in rows if r['event'] == 'before_close')
        if before['faulthandler_enabled'] is not True:
            return False
        for name in ('update_started', 'update_returned'):
            if [r['index'] for r in rows if r['event'] == name] != list(range(request['application_updates'])):
                return False
        source = json.loads((Path(output) / 'installed-close-source.json').read_text())['source']
        return isinstance(source, str) and bool(source.strip())
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def collect_approved(request, *, parent_output, vendor_python, env, cwd, worker_started, on_spawn=None):
    """Callable only: no execution-capable CLI and no submission side effects."""
    validate(request, require_approval=True)
    now = time.monotonic()
    if not math.isfinite(worker_started) or not 0 <= now - worker_started < 240:
        raise ValueError('Original worker-entry timestamp required; never restart budget')
    parent = Path(parent_output).resolve()
    child_output = Path(request['output']).resolve()
    if child_output != parent / 'native':
        raise ValueError('Isolated native output must be a child of parent output')
    if str(vendor_python) != '/isaac-sim/python.sh' or not Path(vendor_python).is_file():
        raise ValueError('Previously observed vendor launcher required')
    parent.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(request, sort_keys=True, allow_nan=False).encode()
    pin = hashlib.sha256(raw).hexdigest()
    path = parent / 'probe-request.json'
    path.write_bytes(raw)
    deadline = worker_started + 240
    # Fail closed against accidentally passing account keys or transfer URLs.
    allowed = {'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'DISPLAY', 'XDG_RUNTIME_DIR',
               'ACCEPT_EULA', 'OMNI_KIT_ACCEPT_EULA', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES',
               'ISAAC_PATH', 'ISAACSIM_PATH', 'ISAACLAB_PATH', 'EXP_PATH', 'CARB_APP_PATH', 'OMNI_KIT_ALLOW_ROOT'}
    child_env = {k: v for k, v in env.items() if k in allowed}
    child_env.update(PYTHONPATH=str(Path(cwd).resolve()), PYTHONUNBUFFERED='1',
                     CUDA_VISIBLE_DEVICES='0', WANDB_MODE='offline',
                     HUD04_PROBE_NATIVE_DEADLINE=str(deadline))
    receipt = capture_process([str(vendor_python), '-m', 'student_adapter.lifecycle_probe',
                               '--request', str(path), '--request-sha256', pin,
                               '--execute-approved-child'], log=parent / 'native.log',
                              deadline=deadline, env=child_env, cwd=cwd, on_spawn=on_spawn)
    receipt.update(request_sha256=pin, task_id=request['task_id'], _wb_agent=request['_wb_agent'],
                   worker_elapsed_seconds=time.monotonic() - worker_started,
                   native_lifecycle_passed=assess(child_output, request, receipt),
                   result_delivery_verified=False, pipeline_passed=False)
    target = parent / 'probe-parent-receipt.json'
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    if json.loads(target.read_text()) != receipt:
        raise RuntimeError('Local parent receipt readback failed')
    return receipt
"""Bounded tee/heartbeat and fail-closed online setup. No credential persistence."""
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time

ENTITY = 'zhul136-university-of-science-and-technology-of-china'
PROJECT = 'HuD04_WBT'

def offline_environment(env):
    """Explicit user-selected offline mode; no account credentials reach child."""
    result = dict(env)
    for name in ('WANDB_API_KEY', 'OPENAI_API_KEY', 'FLUX_API_KEY', 'GITHUB_TOKEN', 'github', 'key', 'WANDB_RUN_ID'):
        result.pop(name, None)
    result.update(WANDB_MODE='offline', WANDB_ENTITY=ENTITY, WANDB_PROJECT=PROJECT,
                  WANDB_RESUME='never', PYTHONUNBUFFERED='1')
    return result

def online_environment(env):
    result = dict(env)
    if not result.get('WANDB_API_KEY'):
        raise ValueError('Secure worker-side WANDB_API_KEY injection is required; no offline fallback')
    for name in ('OPENAI_API_KEY', 'FLUX_API_KEY', 'GITHUB_TOKEN', 'github', 'key'):
        result.pop(name, None)
    result.update(WANDB_MODE='online', WANDB_ENTITY=ENTITY, WANDB_PROJECT=PROJECT,
                  WANDB_RESUME='never', PYTHONUNBUFFERED='1')
    result.pop('WANDB_RUN_ID', None)
    return result

def redact(text, env):
    for name in ('WANDB_API_KEY', 'OPENAI_API_KEY', 'FLUX_API_KEY', 'GITHUB_TOKEN', 'github', 'key'):
        if env.get(name):
            text = text.replace(env[name], '[REDACTED]')
    return re.sub(r'https?://[^\s\"\x27<>]+', '[URL REDACTED]', text)

def run_live(command, env, timeout, log_path, cwd, heartbeat_seconds=20, sink=print):
    """Both disk and console are redacted. Heartbeats continue during quiet startup.

    stdout is read in bounded chunks; a bounded queue applies backpressure.
    Timeout/error paths clean up the entire child process group. Never log argv.
    """
    started = time.monotonic()
    events = queue.Queue(maxsize=32)
    with Path(log_path).open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        def reader():
            try:
                while True:
                    block = child.stdout.readline(8192)
                    if not block:
                        break
                    events.put(block)
            finally:
                events.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        last_heartbeat = started
        ended = False
        pending = b''
        suppress_line = False
        def emit(text):
            safe = redact(text, env)
            log.write(safe + '\n'); log.flush()
            sink(safe, flush=True)
        try:
            while not ended:
                now = time.monotonic()
                if now - started >= timeout:
                    raise TimeoutError('Live child exceeded bounded runtime')
                if now - last_heartbeat >= heartbeat_seconds:
                    emit(json.dumps({'event':'training_heartbeat', 'elapsed_seconds':round(now-started,1),
                                     'child_running':child.poll() is None, 'is_progress_evidence':False}))
                    last_heartbeat = now
                try:
                    block = events.get(timeout=min(.2, max(.001, timeout-(now-started))))
                except queue.Empty:
                    continue
                if block is None:
                    if pending and not suppress_line:
                        emit(pending.decode('utf-8',errors='replace'))
                    ended = True
                else:
                    pending += block
                    # Never redact independent chunks: a credential or URL can
                    # straddle the chunk boundary. Drop oversized whole lines.
                    if len(pending) > 65536:
                        suppress_line = True
                        pending = b''
                    if block.endswith(b'\n'):
                        if suppress_line:
                            emit('[oversized console line suppressed]')
                        else:
                            emit(pending.decode('utf-8',errors='replace').rstrip('\r\n'))
                        pending = b''
                        suppress_line = False
            code = child.wait(timeout=max(.01, timeout-(time.monotonic()-started)))
            if code:
                raise RuntimeError('Live child failed; inspect redacted log')
        finally:
            # Children may outlive the parent; always clean up this owned group.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.stdout.close()

def relative_checkpoint(completed_updates):
    return completed_updates > 0 and completed_updates % 500 == 0
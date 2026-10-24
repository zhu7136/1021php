"""Bounded diagnostics for this job only; no system/process discovery or retries."""
from __future__ import annotations

from contextlib import contextmanager
import faulthandler
import functools
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

INIT_LIMIT_SECONDS = 300
_phase = 'runtime_imports'
_started = time.monotonic()
_installed = False


def clean_line(text: str) -> str:
    # No environment dumps; redact credential assignments and URL queries in logs.
    text = re.sub(r'https?://[^\s"<>]+', '[URL_REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|token|password|authorization)\s*[:=]\s*)\S+',
                  r'\1[REDACTED]', text)
    return re.sub(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',
                  '[TOKEN_REDACTED]', text)


def event(name: str, **fields) -> None:
    payload = dict(event=name, phase=_phase, elapsed_seconds=round(time.monotonic()-_started, 3), **fields)
    print(json.dumps(payload, sort_keys=True), flush=True)


@contextmanager
def phase(name: str):
    global _phase
    previous = _phase
    _phase = name
    started = time.monotonic()
    event('diagnostic_phase_start')
    try:
        yield
    except BaseException as exc:
        event('diagnostic_phase_failed', exception_type=type(exc).__name__)
        raise
    else:
        event('diagnostic_phase_complete', duration_seconds=round(time.monotonic()-started, 3))
    finally:
        _phase = previous


def bounded_launcher(original, limit_seconds=INIT_LIMIT_SECONDS):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with phase('isaac_sim_initialization'):
            event('initialization_deadline_armed', limit_seconds=limit_seconds)
            # CPython's native watchdog works even if a C extension holds the GIL.
            # On expiry it dumps all Python stacks and exits THIS runtime process.
            faulthandler.dump_traceback_later(limit_seconds, repeat=False, file=sys.stderr, exit=True)
            try:
                return original(*args, **kwargs)
            finally:
                faulthandler.cancel_dump_traceback_later()
    return wrapped


def instrument_class(cls):
    if cls.__dict__.get('_hud04_diagnostic_wrapped', False):
        return cls
    for method_name in ('__init__', 'reset', 'reset_all', 'step', 'learn', 'save'):
        original = getattr(cls, method_name, None)
        if not callable(original):
            continue
        def make_wrapper(fn, method):
            seen = False
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                nonlocal seen
                if seen and method != 'save':
                    return fn(*args, **kwargs)
                seen = True
                with phase(cls.__name__ + '.' + method):
                    result = fn(*args, **kwargs)
                    if method == 'save':
                        path_arg = args[1] if len(args)>1 else kwargs.get('path')
                        if isinstance(path_arg, (str, Path)):
                            path = Path(path_arg)
                            event('checkpoint_save_returned', checkpoint_name=path.name,
                                  file_exists=path.is_file(), bytes=path.stat().st_size if path.is_file() else 0)
                    return result
            return wrapped
        setattr(cls, method_name, make_wrapper(original, method_name))
    cls._hud04_diagnostic_wrapped = True
    return cls


class LogMirror:
    """Bounded reads of current-job application logs, never arbitrary files."""
    def __init__(self, root: Path, since: float):
        self.root = Path(root)
        self.since = since
        self.offsets = {}

    def paths(self):
        patterns = (
            (Path('/isaac-sim/kit/logs/Kit/Isaac-Sim'), '*/kit_*.log'),
            (self.root/'logs/HuD04_WBT', '*/train.log'),
        )
        for directory, pattern in patterns:
            for path in sorted(directory.glob(pattern))[-3:]:
                try:
                    if path.is_file() and path.stat().st_mtime >= self.since:
                        yield path
                except OSError:
                    continue

    def poll(self):
        for path in self.paths():
            try:
                size = path.stat().st_size
                offset = self.offsets.get(path, 0)
                if size < offset:
                    offset = 0
                with path.open('rb') as stream:
                    stream.seek(offset)
                    data = stream.read(8192)
                    self.offsets[path] = stream.tell()
                if data:
                    event('application_log_tail', source=path.name, text=clean_line(data.decode('utf-8', 'replace')))
            except OSError as exc:
                event('application_log_read_failed', source=path.name, exception_type=type(exc).__name__)


def install(root: Path):
    """Install tracing without changing robot, PPO, renderer, or device settings."""
    global _installed
    if _installed:
        return
    _installed = True
    faulthandler.enable(file=sys.stderr, all_threads=True)
    event('startup_diagnostics_installed', init_limit_seconds=INIT_LIMIT_SECONDS)
    mirror = LogMirror(root, time.time()-5)
    stop = threading.Event()
    def heartbeat():
        beats = 0
        while not stop.wait(15):
            event('runtime_heartbeat')
            mirror.poll()
            beats += 1
            if beats % 4 == 0:
                event('python_stack_snapshot')
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    threading.Thread(target=heartbeat, daemon=True, name='hud04-diagnostics').start()
    # Import the same training modules the entrypoint already uses, not Isaac Sim.
    with phase('training_module_imports'):
        import holosoma.utils.sim_utils as sim_utils
        import holosoma.train_agent as train_agent
    sim_utils.setup_isaaclab_launcher = bounded_launcher(sim_utils.setup_isaaclab_launcher)
    get_class = train_agent.get_class
    def traced_get_class(target):
        return instrument_class(get_class(target))
    train_agent.get_class = traced_get_class
    os.environ['LOGURU_LEVEL'] = 'INFO'
    return stop


def wait_for_child(child, root: Path, timeout_seconds: float, heartbeat_seconds=15):
    """Parent heartbeat survives a runtime child blocked in native GPU code."""
    deadline = time.monotonic()+timeout_seconds
    mirror = LogMirror(root, time.time()-5)
    next_report = time.monotonic()
    while child.poll() is None:
        now = time.monotonic()
        if now >= deadline:
            import subprocess
            raise subprocess.TimeoutExpired('own-runtime-child', timeout_seconds)
        if now >= next_report:
            event('runtime_parent_heartbeat', child_still_running=True)
            mirror.poll()
            next_report = now+heartbeat_seconds
        time.sleep(min(1, max(0, deadline-time.monotonic())))
    mirror.poll()
    return child.returncode
"""Bounded Git-based Flux smoke bootstrap. No embedded credentials or data URLs.

Provide --data-file for a mounted immutable v8 repair ZIP or W&B export ZIP.
Alternatively provide --data-url with a short-lived HTTPS W&B artifact URL.
This is a simulation-only plumbing test, never hardware deployment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '99c8225859128d27ec94dcc9ea7176e1257257926ae19900be40bebe4208d4e4'
ARCHIVE_BYTES = 218231385
ARCHIVE_NAME = 'hud04-cpu-repair-v8-56-clips.zip'
PREFIX = 'kinematic-candidates-56/'
LIMIT_BYTES = 512 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 180
MAX_DOWNLOAD_TIMEOUT_SECONDS = 300


def event(name: str, **fields) -> None:
    """Structured diagnostics: callers must never pass URLs or credentials."""
    print(json.dumps({'event': name, **fields}, sort_keys=True), flush=True)


@contextmanager
def stage(name: str):
    started = time.monotonic()
    event('stage_start', stage=name)
    try:
        yield
    except BaseException as exc:
        event('stage_failed', stage=name, error_type=type(exc).__name__,
              elapsed_seconds=round(time.monotonic()-started, 3))
        raise
    else:
        event('stage_complete', stage=name,
              elapsed_seconds=round(time.monotonic()-started, 3))


class DownloadDeadlineExceeded(TimeoutError):
    pass


class ArchiveValidationError(ValueError):
    """Only fixed, URL-free messages may be preserved in an error."""
    pass


@contextmanager
def download_deadline(seconds: float):
    """Interrupt connect/read as well as slow trickle transfers on Linux.

    A socket timeout alone does not bound an entire download. This bootstrap
    calls downloads on the worker's main thread; never silently relax the limit.
    """
    if not 0 < seconds <= MAX_DOWNLOAD_TIMEOUT_SECONDS:
        raise ValueError('Download deadline must be positive and at most 300 seconds')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Bounded download requires the worker main thread')
    if not hasattr(signal, 'setitimer'):
        raise RuntimeError('Bounded download requires a POSIX interval timer')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('Refusing to replace an existing interval timer')
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise DownloadDeadlineExceeded('Download total time limit reached; URL omitted')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def check_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ''
    allowed = host == 'storage.googleapis.com' or host.endswith('.storage.googleapis.com')
    if (parsed.scheme != 'https' or not allowed or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError('Expected an HTTPS W&B artifact GCS download URL; URL omitted')


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_archive(url: str, destination: Path,
                  timeout_seconds: float = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    check_url(url)
    digest = hashlib.sha256()
    count = 0
    started = time.monotonic()
    last_report = started
    total = None
    opener = urllib.request.build_opener(SafeRedirect())
    event('download_start', timeout_seconds=timeout_seconds, max_bytes=LIMIT_BYTES)
    try:
        with download_deadline(timeout_seconds):
            with opener.open(url, timeout=min(45, timeout_seconds)) as response, destination.open('wb') as target:
                headers = getattr(response, 'headers', {})
                length = headers.get('Content-Length')
                if length is not None:
                    total = int(length)
                    if total < 0 or total > LIMIT_BYTES:
                        raise ArchiveValidationError('Invalid or oversized download Content-Length')
                event('download_connected', content_length_bytes=total)
                while chunk := response.read(64 * 1024):
                    if count + len(chunk) > LIMIT_BYTES:
                        raise ArchiveValidationError('Download exceeds configured archive limit')
                    target.write(chunk)
                    digest.update(chunk)
                    first_chunk = count == 0
                    count += len(chunk)
                    now = time.monotonic()
                    if first_chunk or now-last_report >= 5:
                        elapsed = now-started
                        event('download_progress', bytes_downloaded=count,
                              content_length_bytes=total, elapsed_seconds=round(elapsed, 3),
                              mib_per_second=round(count/(1024*1024*max(elapsed, 0.001)), 3))
                        last_report = now
                if total is not None and count != total:
                    raise ArchiveValidationError('Download size differs from Content-Length')
                if digest.hexdigest() != ARCHIVE_SHA256:
                    raise ArchiveValidationError('v8 archive SHA256 mismatch; refusing to train')
    except Exception as exc:
        event('download_failed', error_type=type(exc).__name__, bytes_downloaded=count,
              elapsed_seconds=round(time.monotonic()-started, 3),
              http_status=exc.code if isinstance(exc, urllib.error.HTTPError) else None)
        if isinstance(exc, ArchiveValidationError):
            raise ArchiveValidationError(str(exc)) from None
        raise RuntimeError('Artifact download failed ('+type(exc).__name__+'); URL omitted') from None
    event('download_complete', bytes_downloaded=count,
          elapsed_seconds=round(time.monotonic()-started, 3), sha256_verified=True)


def stage_local_archive(source: Path, destination: Path) -> None:
    """Read only the selected mount; never extract an untrusted wrapper tree.

    A W&B whole-version export may wrap the approved ZIP with other files.
    Accept exactly one expected member and require the original byte hash.
    Never fall back to the network when a local source fails validation.
    """
    if not source.is_absolute() or not source.is_file():
        raise ArchiveValidationError('Mounted archive must be an existing absolute file path')
    size = source.stat().st_size
    if not 0 < size <= LIMIT_BYTES:
        raise ArchiveValidationError('Mounted archive exceeds configured size limit or is empty')
    if source.resolve() == destination.resolve():
        raise ArchiveValidationError('Source and staged archive must differ')

    def copy_verified(reader) -> None:
        digest = hashlib.sha256()
        count = 0
        with destination.open('xb') as target:
            while chunk := reader.read(64 * 1024):
                count += len(chunk)
                if count > ARCHIVE_BYTES:
                    raise ArchiveValidationError('Mounted v8 payload exceeds expected size')
                target.write(chunk)
                digest.update(chunk)
        if count != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
            raise ArchiveValidationError('Mounted v8 archive size or SHA256 mismatch; refusing to train')

    if size == ARCHIVE_BYTES:
        with source.open('rb') as reader:
            copy_verified(reader)
        wrapper = False
    else:
        try:
            with zipfile.ZipFile(source) as bundle:
                members = [m for m in bundle.infolist()
                           if not m.is_dir() and m.filename.split('/')[-1] == ARCHIVE_NAME]
                if len(members) != 1:
                    raise ArchiveValidationError('Export must contain exactly one approved v8 ZIP')
                member = members[0]
                parts = member.filename.split('/')
                if (any(p in ('', '.', '..') for p in parts) or '\\' in member.filename
                        or member.file_size != ARCHIVE_BYTES
                        or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ArchiveValidationError('Unsafe or incorrectly sized export member')
                with bundle.open(member) as reader:
                    copy_verified(reader)
            wrapper = True
        except zipfile.BadZipFile:
            raise ArchiveValidationError('Mounted file is not a valid v8 archive or export ZIP') from None
    event('local_archive_verified', bytes_staged=ARCHIVE_BYTES, sha256_verified=True,
          export_wrapper=wrapper, network_download=False)


def extract_candidates(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if m.filename.startswith(PREFIX) and not m.is_dir()]
        if sum(m.file_size for m in members) > LIMIT_BYTES:
            raise ValueError('Expanded candidate data exceeds configured size limit')
        names = set()
        for member in members:
            relative = member.filename[len(PREFIX):]
            if not relative or '/' in relative or '\\' in relative or relative in ('.', '..'):
                raise ValueError('Unsafe archive path')
            if relative in names:
                raise ValueError('Duplicate archive entry')
            names.add(relative)
            with source.open(member) as src, (output/relative).open('xb') as dst:
                shutil.copyfileobj(src, dst)
    if len(list(output.glob('*_motion.npz'))) != 56:
        raise ValueError('Expected all56 v8 motions; refusing to train')
    import json
    report = json.loads((output/'validation.json').read_text())
    if report['files'] != 56 or not report['all_kinematic_gates_pass']:
        raise ValueError('Archive is not the approved all56 v8 candidate batch')
    for name, digest in report['sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe manifest path')
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Candidate manifest digest mismatch')


def worker(url: str | None, download_timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
           data_file: str | None = None) -> None:
    work = ROOT/'outputs'/'hud04-flux-git-v8-smoke'
    # A unique attempt directory preserves earlier data/checkpoints on restart.
    import tempfile
    work.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=work))
    archive = attempt/'v8.zip'
    if data_file:
        with stage('stage_local_archive'):
            stage_local_archive(Path(data_file), archive)
    else:
        with stage('download_archive'):
            fetch_archive(url, archive, download_timeout_seconds)
    dataset = attempt/'candidates'
    with stage('extract_and_verify_candidates'):
        extract_candidates(archive, dataset)
    print('v8 archive and all56 candidate hashes verified; installing pinned dependencies', flush=True)
    deps = ROOT/'.flux-deps'
    with stage('install_pinned_dependencies'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target',
            str(deps), '-r', str(ROOT/'flux/requirements.txt')], check=True)
    os.environ['PYTHONPATH'] = str(deps)+os.pathsep+os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, str(deps))
    sys.argv = [str(ROOT/'flux/train_hud04.py'), '--train', '--dataset', str(dataset),
        '--acknowledge-kinematic-candidate', '--num-envs', '32', '--iterations', '2',
        '--time-limit-seconds', '900', '--logger', 'disabled', '--name', 'hud04-flux-git-v8-smoke']
    import runpy
    with stage('training_entrypoint'):
        try:
            runpy.run_path(str(ROOT/'flux/train_hud04.py'), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    event('smoke_entrypoint_complete', num_envs=32, requested_iterations=2,
          note='Confirm PPO iteration logs and persisted checkpoint separately')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--data-url', default=os.environ.get('HUD04_V8_DATA_URL'))
    sources.add_argument('--data-file', default=os.environ.get('HUD04_V8_DATA_FILE'),
                         help='Absolute mounted path to the original v8 ZIP or W&B version export ZIP')
    parser.add_argument('--download-timeout-seconds', type=int, default=DOWNLOAD_TIMEOUT_SECONDS,
                        help='Total download deadline (1..300 seconds), within the unchanged 900-second worker limit')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.download_timeout_seconds <= MAX_DOWNLOAD_TIMEOUT_SECONDS:
        parser.error('Download timeout must be between 1 and 300 seconds')
    if bool(args.data_url) == bool(args.data_file):
        parser.error('Provide exactly one mounted data file or short-lived data URL')
    if args.data_url:
        check_url(args.data_url)
    if args.data_file and not Path(args.data_file).is_absolute():
        parser.error('Mounted data file path must be absolute')
    if args.worker:
        worker(args.data_url, args.download_timeout_seconds, args.data_file)
        return
    launcher = Path('/isaac-sim/python.sh')
    if not launcher.is_file():
        raise SystemExit('Expected the approved Flux IsaacSim image; no CPU/system-Python fallback')
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    env.pop('HUD04_V8_DATA_URL', None)
    env.pop('HUD04_V8_DATA_FILE', None)
    env['HUD04_V8_DATA_FILE' if args.data_file else 'HUD04_V8_DATA_URL'] = args.data_file or args.data_url
    child = subprocess.Popen([str(launcher), str(Path(__file__).resolve()), '--worker',
        '--download-timeout-seconds', str(args.download_timeout_seconds)],
        env=env, start_new_session=True)
    try:
        result = child.wait(timeout=900)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        event('worker_limit_or_interrupt', worker_timeout_seconds=900)
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise SystemExit('Smoke worker interrupted or reached900seconds; own worker group terminated')
    raise SystemExit(result)


if __name__ == '__main__':
    main()
"""Bounded Git-based Flux smoke bootstrap. No embedded credentials or data URLs.

Provide --data-url with a short-lived HTTPS download URL for the immutable v8
repair artifact ZIP, or inject HUD04_V8_DATA_URL through a supported runner.
This is a simulation-only plumbing test, never hardware deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '99c8225859128d27ec94dcc9ea7176e1257257926ae19900be40bebe4208d4e4'
PREFIX = 'kinematic-candidates-56/'
LIMIT_BYTES = 512 * 1024 * 1024


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


def fetch_archive(url: str, destination: Path) -> None:
    check_url(url)
    digest = hashlib.sha256()
    count = 0
    opener = urllib.request.build_opener(SafeRedirect())
    try:
        with opener.open(url, timeout=45) as response, destination.open('wb') as target:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > LIMIT_BYTES:
                    raise ValueError('Download exceeds configured archive limit')
                digest.update(chunk)
                target.write(chunk)
    except Exception as exc:
        raise RuntimeError('Artifact download failed ('+type(exc).__name__+'); URL omitted') from None
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError('v8 archive SHA256 mismatch; refusing to train')


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


def worker(url: str) -> None:
    work = ROOT/'outputs'/'hud04-flux-git-v8-smoke'
    # A unique attempt directory preserves earlier data/checkpoints on restart.
    import tempfile
    work.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=work))
    archive = attempt/'v8.zip'
    fetch_archive(url, archive)
    dataset = attempt/'candidates'
    extract_candidates(archive, dataset)
    print('v8 archive and all56 candidate hashes verified; installing pinned dependencies', flush=True)
    deps = ROOT/'.flux-deps'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target',
        str(deps), '-r', str(ROOT/'flux/requirements.txt')], check=True)
    os.environ['PYTHONPATH'] = str(deps)+os.pathsep+os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, str(deps))
    sys.argv = [str(ROOT/'flux/train_hud04.py'), '--train', '--dataset', str(dataset),
        '--acknowledge-kinematic-candidate', '--num-envs', '32', '--iterations', '2',
        '--time-limit-seconds', '900', '--logger', 'disabled', '--name', 'hud04-flux-git-v8-smoke']
    import runpy
    runpy.run_path(str(ROOT/'flux/train_hud04.py'), run_name='__main__')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-url', default=os.environ.get('HUD04_V8_DATA_URL'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.data_url:
        parser.error('A short-lived v8 artifact download URL is required; never put it in Git')
    check_url(args.data_url)
    if args.worker:
        worker(args.data_url)
        return
    launcher = Path('/isaac-sim/python.sh')
    if not launcher.is_file():
        raise SystemExit('Expected the approved Flux IsaacSim image; no CPU/system-Python fallback')
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0', HUD04_V8_DATA_URL=args.data_url)
    child = subprocess.Popen([str(launcher), str(Path(__file__).resolve()), '--worker'],
        env=env, start_new_session=True)
    try:
        result = child.wait(timeout=900)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
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
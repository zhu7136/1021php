"""CPU-only, fail-closed reconstruction of the public Flux input shards.

No network, simulator import, task submission, or training is performed.
Default: verify shards only. --output reconstructs into a NEW directory.
Some native wheels are intentionally NOT public; see delivery documentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Invalid relative path')
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe relative path')
    return Path(*pure.parts)


def regular(root: Path, name: str) -> Path:
    path = root / safe_relative(name)
    for candidate in [path, *path.parents]:
        if candidate == root.parent:
            break
        if candidate.is_symlink():
            raise ValueError('Symlinks are not accepted')
    if not path.is_file():
        raise ValueError(f'Missing regular file: {name}')
    return path


def verify_file(path: Path, record: dict) -> None:
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError(f'Size/hash mismatch: {path.name}')
    with path.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Git LFS pointer is not input data')


def verify_parts(source: Path, manifest: dict) -> None:
    if manifest.get('schema_version') != 1 or manifest.get('gpu_start_authorized') is not False:
        raise ValueError('Unexpected manifest or authorization state')
    seen = set()
    for archive in manifest['archives']:
        safe_relative(archive['name'])
        parts = archive['parts']
        if not parts or sum(p['bytes'] for p in parts) != archive['bytes']:
            raise ValueError('Invalid part count/size')
        for index, part in enumerate(parts):
            if part['index'] != index or part['path'] in seen or part['bytes'] > 48 * 1024 * 1024:
                raise ValueError('Invalid/duplicate part order or size')
            seen.add(part['path'])
            verify_file(regular(source, part['path']), part)


def safe_unpack(archive: Path, destination: Path, expected_files: list[dict]) -> None:
    expected = {r['path']: r for r in expected_files}
    if len(expected) != len(expected_files) or len(expected) > 1000:
        raise ValueError('Invalid extraction manifest')
    if sum(r['bytes'] for r in expected.values()) > 1024 ** 3:
        raise ValueError('Extraction bound exceeded')
    seen = set()
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            relative = safe_relative(member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected or non-regular archive member')
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError('Unexpected archive member size')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            for candidate in [target, *target.parents]:
                if candidate == destination.parent:
                    break
                if candidate.is_symlink():
                    raise ValueError('Refusing symlink extraction destination')
            incoming = source.extractfile(member)
            if incoming is None:
                raise ValueError('Missing archive member stream')
            with incoming, target.open('xb') as output:
                shutil.copyfileobj(incoming, output, 1024 * 1024)
            verify_file(target, record)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Archive members missing')


def reconstruct(source: Path, manifest: dict, output: Path) -> None:
    verify_parts(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    for record in manifest['archives']:
        archive = output / safe_relative(record['name'])
        with archive.open('xb') as target:
            for part in record['parts']:
                with regular(source, part['path']).open('rb') as stream:
                    shutil.copyfileobj(stream, target, 1024 * 1024)
        verify_file(archive, record)
        safe_unpack(archive, output, record['extracted_files'])


def install(output: Path, manifest: dict, python: Path, private_wheels: Path | None) -> None:
    if private_wheels is None:
        raise ValueError('Public wheels are incomplete. Supply the matching privately delivered --withheld-wheel-dir; no network fallback.')
    # Verify the complete withheld set before copying anything.
    for record in manifest['withheld_wheels']:
        verify_file(regular(private_wheels, record['file']), record)
    for record in manifest['withheld_wheels']:
        incoming = regular(private_wheels, record['file'])
        target = output / 'php-offline/wheelhouse' / record['file']
        with incoming.open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        verify_file(target, record)
    subprocess.run([str(python), '-c', 'import sys; assert sys.version_info[:2] == (3,11), "Python3.11 required"'], check=True)
    subprocess.run([str(python), '-m', 'pip', '--isolated', 'install', '--no-index',
                    '--no-deps', '--require-hashes', '--find-links', str(output / 'php-offline/wheelhouse'),
                    '--target', str(output / '.flux-deps'), '-r', str(output / 'php-offline/requirements-full.lock')], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'flux/offline-inputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--python', type=Path)
    parser.add_argument('--withheld-wheel-dir', type=Path)
    args = parser.parse_args()
    if args.source.is_symlink():
        raise ValueError('Source directory cannot be a symlink')
    source = args.source.resolve()
    manifest = json.loads(regular(source, 'manifest.json').read_text())
    if args.install and (args.output is None or args.python is None or args.withheld_wheel_dir is None):
        parser.error('--install requires --output, --python and --withheld-wheel-dir')
    if args.output:
        # Reject links before resolve, including existing ancestor directories.
        for item in [args.output, *args.output.parents]:
            if item.is_symlink():
                raise ValueError('Output cannot traverse a symlink')
        output = args.output.resolve()
        reconstruct(source, manifest, output)
        if args.install:
            install(output, manifest, args.python.resolve(), args.withheld_wheel_dir.resolve())
    else:
        verify_parts(source, manifest)
    print(json.dumps({'shards_verified': True, 'reconstructed': args.output is not None,
                      'offline_install_completed': args.install, 'gpu_started': False,
                      'public_wheels': manifest['public_wheel_count'],
                      'withheld_packages': [r['name'] for r in manifest['withheld_wheels']],
                      'worker_delivery_verified': False}))


if __name__ == '__main__':
    main()
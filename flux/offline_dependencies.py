"""Install a verified CPython 3.11 wheel bundle without any online fallback."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path, requirements: Path | None = None) -> dict:
    bundle=Path(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest.get('schema_version')!=1 or manifest.get('wheel_count')!=53:
        raise ValueError('Unexpected offline dependency manifest')
    if requirements is not None and sha256(requirements)!=manifest['original_requirements_sha256']:
        raise ValueError('Original requirements changed; rebuild the offline bundle')
    lock=bundle/'requirements.lock'
    if lock.is_symlink() or sha256(lock)!=manifest['lock_sha256']:
        raise ValueError('Offline lock digest mismatch')
    lines=lock.read_text().splitlines()
    pattern=r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+ --hash=sha256:[a-f0-9]{64}'
    if len(lines)!=53 or any(not re.fullmatch(pattern,line) for line in lines):
        raise ValueError('Unexpected offline lock contents')
    records=manifest['wheels']
    names=[record['filename'] for record in records]
    if len(records)!=53 or len(set(names))!=53:
        raise ValueError('Offline wheel count or uniqueness mismatch')
    for record in records:
        name=record['filename']
        if Path(name).name!=name or not name.endswith('.whl'):
            raise ValueError('Unsafe wheel filename')
        path=bundle/'wheelhouse'/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Offline wheel missing or symlinked: '+name)
        if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise ValueError('Offline wheel integrity mismatch: '+name)
    return manifest


def check_runtime() -> None:
    if sys.version_info[:2]!=(3,11) or platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Offline bundle requires Linux x86_64 CPython 3.11')
    if platform.python_implementation()!='CPython':
        raise RuntimeError('Offline bundle requires CPython')
    if importlib.metadata.version('torch').split('+')[0]!='2.7.0':
        raise RuntimeError('Expected image-provided Torch 2.7.0; refusing to replace Torch/CUDA')


def install(root: Path) -> None:
    root=Path(root)
    bundle=root/'flux/offline-deps'
    check_runtime()
    manifest=verify_bundle(bundle,root/'flux/requirements.txt')
    print(json.dumps({'event':'offline_dependencies_verified','wheel_count':53,
        'network_download':False,'torch_source':'existing_image'}),flush=True)
    subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check',
        'install','--no-index','--no-cache-dir','--only-binary=:all:','--no-deps',
        '--require-hashes','--find-links',str(bundle/'wheelhouse'),
        '--target',str(root/'.flux-deps'),'-r',str(bundle/'requirements.lock')],
        check=True,env=dict(os.environ,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1'))
    print(json.dumps({'event':'offline_dependencies_installed','wheel_count':manifest['wheel_count'],
        'network_download':False}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.verify_only:
        result=verify_bundle(root/'flux/offline-deps',root/'flux/requirements.txt')
        print(json.dumps({'integrity_verified':True,'wheel_count':result['wheel_count'],
            'runtime_abi_tested':False,'gpu_started':False}))
    else:
        install(root)
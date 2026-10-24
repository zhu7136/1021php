"""Fail-closed checks for PHP's Flux teacher; no simulator is started here."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PHP_COMMIT = 'c7c9ce3fc75a8754a7e403c58de3b6933254d01b'
HOLOSOMA_COMMIT = '4a0bf04cde5930af5ed59da797568d32be972525'


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def static_checks() -> dict:
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', PHP_COMMIT, 'HEAD'], check=True)
    tree = git('ls-tree', 'HEAD', 'thirdparty/holosoma')
    if HOLOSOMA_COMMIT not in tree:
        raise ValueError('Unexpected Holosoma submodule revision')
    preset = (ROOT / 'wbt_training/config_values/experiment.py').read_text()
    if 'g1_wbt_terrain_ref_noscale' not in preset:
        raise ValueError('Teacher preset is missing')
    component = json.loads((ROOT / 'release-assets.json').read_text())['components']['motions-locomotion']
    return {'php_commit': git('rev-parse', 'HEAD'), 'php_base': PHP_COMMIT,
            'holosoma': HOLOSOMA_COMMIT, 'dataset': 'motions-locomotion',
            'dataset_archive_sha256': component['archive']['sha256'],
            'declared_dataset_files': len(component['files']),
            'dataset_extracted_bytes': sum(f['size'] for f in component['files'])}


def runtime_checks(data: Path) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Training interpreter must be Python 3.11')
    versions = {}
    for package, expected in [('torch', '2.7.0'), ('isaacsim', '5.1.0'), ('isaaclab', '2.3.1')]:
        actual = metadata.version(package)
        if actual.split('+')[0] != expected:
            raise ValueError(f'{package}: expected {expected}, got {actual}')
        versions[package] = actual
    # Do not import IsaacSim/IsaacLab before the application launcher.
    for module in ('tyro', 'wandb', 'onnx', 'tensorboard', 'trimesh', 'scipy', 'holosoma'):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f'Missing module: {module}')
    submodule = ROOT / 'thirdparty/holosoma'
    actual = subprocess.check_output(['git', '-C', str(submodule), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != HOLOSOMA_COMMIT:
        raise ValueError('Worker Holosoma checkout does not match pin')
    import holosoma
    if not Path(holosoma.__file__).resolve().is_relative_to(submodule.resolve()):
        raise ValueError('holosoma import is not from the pinned checkout')
    asset = submodule / 'src/holosoma/holosoma/data/robots/g1/main_mesh_arm_collision_halfspherehand.urdf'
    if not asset.is_file() or asset.stat().st_size < 1000:
        raise ValueError('G1 robot asset is missing')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from download_assets import read_component, verify
    component = read_component(ROOT / 'release-assets.json', 'motions-locomotion')
    failures = verify(component, data)
    if failures:
        raise ValueError('; '.join(failures[:5]))
    from wbt_training.utils.registry import validate_bundle
    validate_bundle(data)
    return {'python': sys.version.split()[0], 'packages': versions, 'data_verified': True}


if __name__ == '__main__':
    print(json.dumps(static_checks(), indent=2))
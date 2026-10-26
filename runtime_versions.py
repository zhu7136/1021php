"""Read version evidence without importing IsaacSim, IsaacLab, Torch or CUDA.

IsaacLab v2.3.1 VERSION=2.3.1 but its extension/package version=0.48.0.
IsaacSim v5.1.0's official source VERSION is 5.1.0-rc.19. Only that
explicit build marker (not arbitrary prereleases) maps to the pinned release.
No environment override, network download or installation is performed.
"""
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import re
import sys
import tomllib


def _read(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 4096:
        raise ValueError(f'Missing or oversized runtime version evidence: {path}')
    return path.read_text(encoding='utf-8').strip()


def _distribution(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _origin(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)  # top-level names only; no package import
    if spec is None or spec.origin in (None, 'built-in', 'frozen'):
        return None
    return Path(spec.origin).resolve()


def _sim_release(raw: str) -> str:
    # Official tag: https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/VERSION
    if not re.fullmatch(r'5\.1\.0(?:-rc\.19)?(?:\+[A-Za-z0-9.-]+)?', raw):
        raise ValueError(f'IsaacSim expected release 5.1.0 (official rc.19 marker allowed), got {raw!r}')
    return '5.1.0'


def check_runtime_versions() -> dict:
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Training interpreter must be Python 3.11')
    torch = _distribution('torch')
    if torch is None or not re.fullmatch(r'2\.7\.0(?:\+[A-Za-z0-9.-]+)?', torch):
        raise ValueError(f'Torch expected 2.7.0, got {torch!r}')
    evidence = {'torch': {'distribution': torch}}

    sim_origin = _origin('isaacsim')
    executable = Path(sys.executable).resolve()
    roots = set()
    # Associate VERSION with the actual interpreter or resolved import, never
    # with cwd, a configurable arbitrary directory, or an unrelated ancestor.
    for anchor in (executable, sim_origin):
        if anchor is None:
            continue
        for root in anchor.parents:
            if ((root / 'python.sh').is_file() and (root / 'kit').is_dir()
                    and (anchor == sim_origin or executable.is_relative_to(root / 'kit/python'))):
                roots.add(root.resolve())
    if len(roots) > 1:
        raise ValueError('Conflicting standalone IsaacSim roots')
    sim_files = []
    for root in sorted(roots):
        raw = _read(root / 'VERSION')
        _sim_release(raw)
        if sim_origin is not None and not sim_origin.is_relative_to(root):
            raise ValueError('IsaacSim import shadows the active standalone installation')
        sim_files.append({'path': str(root / 'VERSION'), 'raw': raw})
    sim_dist = _distribution('isaacsim')
    if sim_dist is not None:
        _sim_release(sim_dist)
    if not sim_files and (sim_dist is None or sim_origin is None):
        raise ValueError('Cannot verify IsaacSim: need anchored standalone VERSION or importable pip distribution')
    evidence['isaacsim'] = {'release': '5.1.0', 'distribution': sim_dist,
                            'module_origin': str(sim_origin) if sim_origin else None,
                            'standalone_versions': sim_files}

    lab_origin = _origin('isaaclab')
    if lab_origin is None:
        raise ValueError('IsaacLab module cannot be resolved without import')
    # Editable install layout used by official IsaacLab and this Flux image.
    lab_root = lab_origin.parent.parent.parent.parent
    expected_origin = lab_root / 'source/isaaclab/isaaclab/__init__.py'
    if lab_origin != expected_origin or not (lab_root / 'isaaclab.sh').is_file():
        raise ValueError('Unrecognized IsaacLab source layout; release evidence unavailable')
    lab_release = _read(lab_root / 'VERSION')
    if lab_release != '2.3.1':
        raise ValueError(f'IsaacLab expected product release 2.3.1, got {lab_release!r}')
    extension_path = lab_root / 'source/isaaclab/config/extension.toml'
    extension = tomllib.loads(_read(extension_path)).get('package', {}).get('version')
    if extension != '0.48.0':
        raise ValueError(f'IsaacLab 2.3.1 expected extension 0.48.0, got {extension!r}')
    lab_dist = _distribution('isaaclab')
    if lab_dist is not None and lab_dist != extension:
        raise ValueError(f'IsaacLab package/extension version conflict: {lab_dist!r} vs {extension!r}')
    evidence['isaaclab'] = {'release': lab_release, 'distribution': lab_dist,
                            'extension': extension, 'module_origin': str(lab_origin),
                            'version_file': str(lab_root / 'VERSION'),
                            'extension_file': str(extension_path)}
    return {'packages': {'torch': torch, 'isaacsim': '5.1.0', 'isaaclab': lab_release},
            'runtime_version_evidence': evidence}


if __name__ == '__main__':
    import json
    print(json.dumps(check_runtime_versions(), indent=2), flush=True)
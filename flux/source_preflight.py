"""CPU/stdlib-only source checks; never fetch or repair sources implicitly."""
from __future__ import annotations
import configparser
import hashlib
import json
from pathlib import Path

SUBMODULE_URLS = {
    'GMR': 'https://github.com/limxdynamics/GMR.git',
    'humanoid-description': 'https://github.com/limxdynamics/humanoid-description.git',
    'thirdparty/holosoma': 'https://github.com/amazon-far/holosoma.git',
}


def verify_sources(root: Path) -> dict:
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(root / '.gitmodules'):
        raise ValueError('Missing .gitmodules; refusing dependency install')
    for path, url in SUBMODULE_URLS.items():
        section = f'submodule "{path}"'
        if config.get(section, 'path', fallback=None) != path or config.get(section, 'url', fallback=None) != url:
            raise ValueError(f'Missing or unexpected submodule mapping: {path}')
    source = root / 'thirdparty/holosoma'
    if not (source / 'src/holosoma/holosoma/train_agent.py').is_file():
        raise ValueError('Missing pinned Holosoma source; initialize submodules before running the smoke test')
    expected = json.loads((root / 'flux/holosoma-base.json').read_text())
    if not expected:
        raise ValueError('Empty Holosoma hash manifest')
    for rel, digest in expected.items():
        relative = Path(rel)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe Holosoma manifest path')
        path = source / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or changed pinned Holosoma source: {rel}')
    return {'submodule_mappings': len(SUBMODULE_URLS), 'holosoma_base_files_verified': len(expected)}
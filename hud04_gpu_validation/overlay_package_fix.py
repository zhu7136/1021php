"""Narrow, auditable repair for the pinned nested PHP overlay package.

Runs before importing wbt_training. Does not modify input archives, weights,
training configuration, the simulator, or site-packages.
"""
import hashlib
from pathlib import Path

ORIGINAL = b'"""Config presets for the G1 WBT training extension."""\n'
PATCHED = ORIGINAL + b'\nfrom pkgutil import extend_path\n\n__path__ = extend_path(__path__, __name__)\n'

def repair_overlay(bundle):
    bundle = Path(bundle).resolve(strict=True)
    relative = 'overlay/wbt_training/config_values/__init__.py'
    target = bundle / relative
    upstream = bundle / 'runtime/php/wbt_training/config_values/motion_command.py'
    for path in (target, upstream):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Pinned package member missing or linked')
        if any(p.is_symlink() for p in path.parents if p != bundle and bundle in p.parents):
            raise ValueError('Linked package parent rejected')
        if not path.resolve(strict=True).is_relative_to(bundle):
            raise ValueError('Package path escaped bundle')
    before = target.read_bytes()
    if before not in (ORIGINAL, PATCHED):
        raise ValueError('Unexpected overlay initializer; refusing broad rewrite')
    if before == ORIGINAL:
        target.write_bytes(PATCHED)
    after = target.read_bytes()
    if after != PATCHED:
        raise ValueError('Package initializer repair readback failed')
    return {
        'repair': 'extend_nested_config_values_package_path',
        'target': relative,
        'before_sha256': hashlib.sha256(before).hexdigest(),
        'after_sha256': hashlib.sha256(after).hexdigest(),
        'upstream_module_sha256': hashlib.sha256(upstream.read_bytes()).hexdigest(),
        'changed': before != after,
        'input_archives_modified': False,
        'native_training_verified': False,
    }
"""Static asset checks and vendor-launcher resolution; no simulator imports."""
from __future__ import annotations
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

LAUNCHER_CANDIDATES = (
    Path('/isaac-sim/python.sh'),
    Path('/workspace/isaaclab/_isaac_sim/python.sh'),
)


def resolve_isaac_launcher() -> Path:
    override = os.environ.get('HUD04_ISAAC_PYTHON_SH')
    candidates = (Path(override),) if override else LAUNCHER_CANDIDATES
    for path in candidates:
        # Require the vendor launcher plus its bundled interpreter. Never use
        # sys.executable or a PATH/system-Python fallback for simulator training.
        if (path.is_absolute() and path.name == 'python.sh' and path.is_file()
                and os.access(path, os.X_OK)
                and (path.parent/'kit/python/bin/python3').is_file()):
            return path
    raise RuntimeError('No verified IsaacSim python.sh launcher. Checked: '
        + ', '.join(map(str, candidates))
        + '. Set HUD04_ISAAC_PYTHON_SH to the absolute vendor python.sh; no system-Python fallback.')


def verify_assets(root: Path) -> dict:
    asset = root/'flux/holosoma_overlay/data/robots/hu_d04'
    urdf = asset/'HU_D04_01_train.urdf'
    if not urdf.is_file():
        raise ValueError('Missing HU_D04 IsaacSim training URDF')
    with urdf.open('rb') as source:
        if source.read(60).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('HU_D04 training URDF is only an LFS pointer')
    names = sorted({element.attrib['filename'] for element in ET.parse(urdf).findall('.//mesh')})
    if not names:
        raise ValueError('Training URDF has no mesh references')
    total = 0
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe HU_D04 mesh reference: {name}')
        path = asset/relative
        if not path.is_file():
            raise ValueError(f'Missing HU_D04 mesh: {name}')
        with path.open('rb') as source:
            header = source.read(84)
        if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'HU_D04 mesh is only an LFS pointer: {name}')
        size = path.stat().st_size
        if len(header) != 84 or size != 84 + 50*struct.unpack('<I', header[80:84])[0]:
            raise ValueError(f'Invalid binary STL size: {name}')
        total += size
    return {'isaacsim_urdf_meshes_verified': len(names), 'mesh_bytes': total}


def verify_runtime(root: Path) -> dict:
    assets = verify_assets(root)
    return {**assets, 'isaac_launcher': str(resolve_isaac_launcher()),
        'simulator_imported': False, 'gpu_initialized': False}
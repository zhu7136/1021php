"""CPU audit of the full official locomotion dataset and pinned G1 meshes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(data: Path) -> dict:
    from wbt_training.utils.registry import bundle_pairs, validate_bundle
    sys.path.insert(0, str(ROOT / 'scripts'))
    from download_assets import read_component, verify
    component = read_component(ROOT / 'release-assets.json', 'motions-locomotion')
    failures = verify(component, data)
    if failures:
        raise ValueError('; '.join(failures[:5]))
    validate_bundle(data)
    pairs = bundle_pairs(data)
    frames = 0
    motion_bytes = 0
    gpu_bytes = 0
    dtypes = set()
    fields = {}
    names = None
    fps_values = set()
    nonempty_terrains = 0
    for motion, terrain, _ in pairs:
        with np.load(motion, allow_pickle=False) as arrays:
            size = arrays['joint_pos'].shape[0]
            frames += size
            signature = (tuple(arrays['joint_names'].tolist()), tuple(arrays['body_names'].tolist()))
            if names is not None and names != signature:
                raise ValueError(f'Joint/body names differ: {motion.name}')
            names = signature
            fps_values.add(float(arrays['fps'].item()))
            if arrays['joint_pos'].shape[1] != len(signature[0]) + 7:
                raise ValueError('Unexpected root/joint position layout')
            if arrays['joint_vel'].shape[1] != len(signature[0]) + 6:
                raise ValueError('Unexpected root/joint velocity layout')
            for key in arrays.files:
                value = arrays[key]
                if value.dtype.kind in 'fiu' and not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite data: {motion.name}/{key}')
                if key not in ('fps', 'joint_names', 'body_names'):
                    motion_bytes += value.nbytes
                    fields.setdefault(key, {'shape_tail': list(value.shape[1:]), 'dtype': str(value.dtype), 'bytes': 0})['bytes'] += value.nbytes
                    dtypes.add(str(value.dtype))
            # Mirror MotionLib's float32 tensors, excluding the root DOFs.
            for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
                value = arrays[key]
                columns = value.size
                if key == 'joint_pos':
                    columns -= size * 7
                if key == 'joint_vel':
                    columns -= size * 6
                gpu_bytes += columns * 4
        if terrain is not None:
            nonempty_terrains += int(np.load(terrain, allow_pickle=False).size > 0)
    urdf = ROOT / 'thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/main_mesh_arm_collision_halfspherehand.urdf'
    mesh_files = set()
    for mesh in ET.parse(urdf).iter('mesh'):
        filename = mesh.attrib['filename']
        if '://' in filename:
            raise ValueError(f'Unresolved mesh URI: {filename}')
        path = urdf.parent / filename
        if not path.is_file() or path.stat().st_size < 100:
            raise ValueError(f'Missing mesh: {filename}')
        with path.open('rb') as stream:
            if stream.read(150).startswith(b'version https://git-lfs.github.com/spec/'):
                raise ValueError(f'LFS pointer instead of mesh: {filename}')
        mesh_files.add(filename)
    return {'hashes_verified': True, 'motion_pairs': len(pairs), 'frames': frames,
            'fps': sorted(fps_values), 'joints': len(names[0]), 'bodies': len(names[1]),
            'nonempty_terrain_files': nonempty_terrains, 'fields': fields,
            'raw_motion_array_bytes': motion_bytes,
            '64_variant_motion_array_GiB': round(motion_bytes * 64 / 2**30, 3),
            '64_variant_MotionLib_core_tensor_GiB_estimate': round(gpu_bytes * 64 / 2**30, 3),
            'memory_estimate_exclusions': ['temporary copies', 'index/mask tensors', 'simulator', 'terrain geometry', 'PPO storage and optimizer'],
            'urdf_unique_meshes_verified': len(mesh_files),
            'runtime_simulator_tested': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data.resolve()), indent=2))
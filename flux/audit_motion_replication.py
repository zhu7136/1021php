"""Header-only estimator for the HU_D04 combine -> terrain-variant path.

python flux/audit_motion_replication.py --motions-dir /path/to/v8 --variants 32
Does not rewrite or fully load source motion arrays. Reports NPY payload only.
"""
import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

META = {'fps', 'joint_names', 'body_names'}
KINEMATIC = {'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
             'body_lin_vel_w', 'body_ang_vel_w'}


def audit(root, variants=32):
    if variants < 1:
        raise ValueError('variants must be positive')
    files = sorted(Path(root).glob('*motion.npz'))
    if not files:
        raise ValueError('No *motion.npz files found')
    fields, frames = {}, 0
    for path in files:
        local = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.filename.endswith('.npy'):
                    continue
                key = member.filename[:-4]
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1,0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2,0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported NPY header version')
                if dtype.hasobject:
                    raise ValueError('Object data is unsupported')
                local[key] = (shape, dtype)
        T = local['joint_pos'][0][0]
        frames += T
        local.setdefault('motion_ends', ((T,), np.dtype('bool')))
        # Registry combining writes its own per-frame int64 motion_idxs.
        local['motion_idxs'] = ((T,), np.dtype('int64'))
        for key, (shape, dtype) in local.items():
            if key not in META and (not shape or shape[0] != T):
                raise ValueError('Unexpected non-frame field: '+key)
            item = fields.setdefault(key, {'dtype': str(dtype), 'combined_payload_bytes': 0,
                                          'example_shape': shape})
            if item['dtype'] != str(dtype):
                raise ValueError('Mixed field dtypes: '+key)
            if key in META and item['combined_payload_bytes']:
                continue
            item['combined_payload_bytes'] += math.prod(shape)*dtype.itemsize
    for key, item in fields.items():
        size = item['combined_payload_bytes'] * (1 if key in META else variants)
        old_itemsize = np.dtype(item['dtype']).itemsize
        # Existing code already casts body_pos_w to float32.
        legacy = size//old_itemsize*4 if key == 'body_pos_w' else size
        streamed = size//old_itemsize*4 if key in KINEMATIC else size
        item.update(legacy_payload_bytes=legacy, streamed_payload_bytes=streamed)
    return {'motion_files': len(files), 'source_frames': frames, 'variants': variants,
            'expanded_frames': frames*variants,
            'combined_payload_bytes': sum(i['combined_payload_bytes'] for i in fields.values()),
            'legacy_payload_bytes': sum(i['legacy_payload_bytes'] for i in fields.values()),
            'streamed_payload_bytes': sum(i['streamed_payload_bytes'] for i in fields.values()),
            'fields': fields,
            'note': 'Header-only estimate, excludes NPY/ZIP overhead; uses registry-added motion_ends/motion_idxs. Not a measured runtime memory peak.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions-dir', required=True, type=Path)
    parser.add_argument('--variants', default=32, type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.motions_dir, args.variants), indent=2))
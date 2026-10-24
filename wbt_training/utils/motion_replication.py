"""Bounded-memory motion replication, compatible with np.load and Holosoma WBT.

The source motion and each variant's frame order are preserved. Only the six
kinematic arrays consumed as float32 by the pinned Holosoma loader are narrowed.
This does NOT reduce the variant count or change obstacle sampling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

FLOAT32_FIELDS = frozenset({
    'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
    'body_lin_vel_w', 'body_ang_vel_w',
})
METADATA_FIELDS = frozenset({'fps', 'joint_names', 'body_names'})


def _numpy(value):
    # Accept CPU torch tensors without importing torch/Isaac into this utility.
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _event(**payload):
    print(json.dumps(payload, sort_keys=True), flush=True)


def stream_offset_motions_to_file(data, env_origins, per_frame_offsets_xy,
                                  file_path='motion.npz', *, chunk_bytes=8*1024*1024,
                                  narrow_float32=True):
    """Write a ZIP_STORED NPZ without an E-times-sized in-memory array.

    Memory is at most one source field, its optional float32 conversion and one
    write chunk; output storage still grows with E. Publish with os.replace only
    after a successful close. On failure a .partial scratch file is retained
    for inspection and the previous destination is left unchanged.
    """
    origins = np.asarray(_numpy(env_origins), dtype=np.float32)
    if origins.ndim != 2 or origins.shape[1] != 3 or len(origins) < 1:
        raise ValueError('env_origins must have shape (E, 3), E >= 1')
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    offsets = np.asarray(_numpy(per_frame_offsets_xy), dtype=np.float32)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError('per_frame_offsets_xy must have shape (T, 2)')
    T, E = len(offsets), len(origins)
    if T < 1:
        raise ValueError('At least one motion frame is required')
    if 'body_pos_w' not in data.keys():
        raise ValueError('body_pos_w is required')
    destination = Path(file_path)
    started = time.monotonic()
    _event(event='motion_replication_start', frames=T, variants=E,
           expanded_frames=T*E, chunk_bytes=chunk_bytes,
           narrow_float32=narrow_float32)
    fd, scratch = tempfile.mkstemp(prefix=destination.name+'.', suffix='.partial',
                                   dir=str(destination.parent))
    total = 0
    with os.fdopen(fd, 'w+b') as output, zipfile.ZipFile(
            output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        # Iterate keys, NOT NpzFile.items(): that materializes even skipped fields.
        for key in data.keys():
            if not isinstance(key, str) or '/' in key or '\\' in key or key in {'.', '..'}:
                raise ValueError('Motion keys must be simple names')
            arr = np.asarray(data[key])
            if arr.dtype.hasobject:
                raise ValueError('Object motion arrays are unsupported')
            field_start = time.monotonic()
            is_position = key == 'body_pos_w'
            repeat = arr.ndim >= 1 and arr.shape[0] == T and key not in METADATA_FIELDS
            if is_position:
                if arr.ndim != 3 or arr.shape[0] != T or arr.shape[2] != 3:
                    raise ValueError('body_pos_w must have shape (T, B, 3)')
                # Match legacy torch float32 arithmetic: cast, grid offset,
                # THEN per-variant origin. Do not fuse/reassociate these adds.
                arr = np.array(arr, dtype=np.float32, order='C', copy=True)
                arr[:, :, :2] += offsets[:, None, :]
            elif narrow_float32 and key in FLOAT32_FIELDS:
                if arr.dtype.kind != 'f':
                    raise ValueError('Kinematic arrays must be floating point')
                arr = arr.astype(np.float32, copy=False)
            shape = (E*T, *arr.shape[1:]) if repeat else arr.shape
            field_bytes = int(np.prod(shape, dtype=object)) * arr.dtype.itemsize
            with archive.open(key+'.npy', 'w', force_zip64=True) as member:
                if not repeat:
                    np.lib.format.write_array(member, arr, allow_pickle=False)
                else:
                    np.lib.format.write_array_header_2_0(member, {
                        'descr': np.lib.format.dtype_to_descr(arr.dtype),
                        'fortran_order': False, 'shape': shape})
                    row_bytes = max(1, int(np.prod(arr.shape[1:], dtype=object))*arr.dtype.itemsize)
                    rows = max(1, chunk_bytes // row_bytes)
                    for variant in range(E):
                        for begin in range(0, T, rows):
                            block = arr[begin:begin+rows]
                            if is_position:
                                block = block + origins[variant][None, None, :]
                            member.write(memoryview(np.ascontiguousarray(block)).cast('B'))
            total += field_bytes
            _event(event='motion_replication_field_complete', field=key,
                   output_shape=shape, output_dtype=str(arr.dtype),
                   payload_bytes=field_bytes,
                   seconds=round(time.monotonic()-field_start, 4))
            del arr
    os.replace(scratch, destination)
    result = dict(event='motion_replication_complete', frames=T, variants=E,
                  expanded_frames=T*E, payload_bytes=total,
                  disk_bytes=destination.stat().st_size,
                  seconds=round(time.monotonic()-started, 4))
    _event(**result)
    return result
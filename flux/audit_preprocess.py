"""CPU-only full-bundle check with one layout; never invokes a simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flux.audit_data import audit
    from wbt_training.utils.multi_motion_helpers import pull_paired_from_wandb_registry
    from wbt_training.utils.obstacle_helpers import add_onpath_obstacle_standalone

    source = audit(args.data.resolve())
    started = time.monotonic()
    counts, motion, obstacle, _ = pull_paired_from_wandb_registry(args.data.resolve().as_uri())
    spacing, result, entire, _ = add_onpath_obstacle_standalone(1, str(motion), str(obstacle), seed=42)
    with np.load(result, allow_pickle=False) as data:
        if data['joint_pos'].shape[0] != source['frames']:
            raise ValueError('Preprocessing changed frame count')
        if int(data['motion_ends'].sum()) != source['motion_pairs']:
            raise ValueError('Preprocessing lost motion boundaries')
        if not np.isfinite(data['body_pos_w']).all():
            raise ValueError('Nonfinite preprocessed body positions')
    report = {'source_audit': source, 'cpu_preprocess_passed': True,
              'registry_motion_counts': counts, 'audit_only_layout_variants': 1,
              'training_preset_layout_variants_unchanged': 64,
              'mesh_vertices': len(entire.vertices), 'mesh_faces': len(entire.faces),
              'layout_spacing': float(spacing), 'motion_output_bytes': Path(result).stat().st_size,
              'preprocess_seconds': round(time.monotonic() - started, 2),
              'simulator_started': False, 'gpu_training_started': False}
    (scratch / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
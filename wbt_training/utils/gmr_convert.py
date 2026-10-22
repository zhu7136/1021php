"""Convert GMR retargeting output for HU_D04 into PHP training bundles.

GMR's ``limx_oli_edu`` export is a BeyondMimic-layout NPZ with 31 actuated
joints and 42 bodies. PHP trains on the 29 joints / 32 bodies ordering that
:mod:`wbt_training.config_values.robot_config` declares for G1, so this module
drops the two head joints, maps the HU_D04 body names onto the G1 semantics,
and reorders both axes by name.

Measured GMR baseline (``scripts/xsens_bvh_to_robot.py --robot limx_oli_edu``);
the mapping tables below are derived from it and asserted at import time:

* joints (31): left leg 6, right leg 6, waist 3, head 2, left arm 7, right
  arm 7. The arm ends at ``wrist_roll_joint``, so each arm orders
  ``wrist_yaw -> wrist_pitch -> wrist_roll`` — the reverse of G1's
  ``wrist_roll -> wrist_pitch -> wrist_yaw``.
* bodies (42, world excluded): ``base_link`` (root), per-foot
  ``contact_foot_{heel,center,tip}_{L,R}``, and the HU_D04-only
  ``head_{yaw,pitch}_link``, ``*_hand_contact``, ``*_hand_manip``.

The converter never assumes input order: every axis is reordered through an
explicit name table and a missing or duplicated name is an error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from wbt_training.config_values.robot_config import (
    G1_29DOF_JOINT_NAMES,
    G1_32BODY_NAMES,
)

REQUIRED_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)

# GMR exports the two head joints; the trained policy has none.
HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")

# HU_D04 body name -> G1 semantic body name. Bodies absent from this table
# (head, hand contacts, heel/tip contact points) are dropped: the config keeps
# one contact point per foot, matching G1.
HU_D04_TO_G1_BODY = {
    "base_link": "pelvis",
    "waist_pitch_link": "torso_link",
    "contact_foot_center_L": "left_foot_contact_point",
    "contact_foot_center_R": "right_foot_contact_point",
}
# Every other G1 body shares its name with HU_D04 and maps to itself.

EXCLUDED_HU_D04_BODIES = frozenset(
    {
        "head_yaw_link",
        "head_pitch_link",
        "left_hand_contact",
        "left_hand_manip",
        "right_hand_contact",
        "right_hand_manip",
        "contact_foot_heel_L",
        "contact_foot_tip_L",
        "contact_foot_heel_R",
        "contact_foot_tip_R",
    }
)

GMR_HU_D04_JOINT_NAMES = tuple(G1_29DOF_JOINT_NAMES[:16]) + HEAD_JOINTS + tuple(
    G1_29DOF_JOINT_NAMES[16:]
)
"""GMR's joint order for ``limx_oli_edu`` (31 names, head inserted after waist)."""

GMR_HU_D04_BODY_NAMES = (
    "base_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "contact_foot_heel_L",
    "contact_foot_center_L",
    "contact_foot_tip_L",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "contact_foot_heel_R",
    "contact_foot_center_R",
    "contact_foot_tip_R",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "head_yaw_link",
    "head_pitch_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "left_hand_contact",
    "left_hand_manip",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
    "right_hand_contact",
    "right_hand_manip",
)
"""GMR's body order for ``limx_oli_edu`` (42 names, world body excluded)."""


def g1_body_for(hu_d04_body: str) -> str | None:
    """Return the G1 semantic name for a HU_D04 body, or ``None`` to drop it."""
    if hu_d04_body in EXCLUDED_HU_D04_BODIES:
        return None
    return HU_D04_TO_G1_BODY.get(hu_d04_body, hu_d04_body)


def _check_tables() -> None:
    if len(G1_29DOF_JOINT_NAMES) != 29 or len(set(G1_29DOF_JOINT_NAMES)) != 29:
        raise AssertionError("G1 joint table must hold 29 unique names")
    if len(G1_32BODY_NAMES) != 32 or len(set(G1_32BODY_NAMES)) != 32:
        raise AssertionError("G1 body table must hold 32 unique names")
    kept = [j for j in GMR_HU_D04_JOINT_NAMES if j not in HEAD_JOINTS]
    if set(kept) != set(G1_29DOF_JOINT_NAMES) or len(kept) != 29:
        raise AssertionError(
            "GMR HU_D04 joints minus head must equal G1's 29 controlled joints"
        )
    mapped = [g1_body_for(b) for b in GMR_HU_D04_BODY_NAMES if g1_body_for(b)]
    if len(mapped) != len(G1_32BODY_NAMES) or set(mapped) != set(G1_32BODY_NAMES):
        raise AssertionError(
            "HU_D04 -> G1 body map must cover the 32 G1 bodies exactly once"
        )
    if len(GMR_HU_D04_BODY_NAMES) != 42:
        raise AssertionError("GMR HU_D04 exports 42 bodies")


_check_tables()


def _load_npz(path: Path) -> dict:
    data = dict(np.load(path, allow_pickle=True))
    if missing := sorted(set(REQUIRED_FIELDS) - set(data)):
        raise ValueError(f"{path}: missing required GMR fields {missing}")
    return data


def _reorder_index(names, target, label: str) -> np.ndarray:
    if len(names) != len(set(names)):
        raise ValueError(f"{label}: duplicate names in input")
    position = {name: i for i, name in enumerate(names)}
    if missing := [name for name in target if name not in position]:
        raise ValueError(f"{label}: input is missing {missing}")
    return np.asarray([position[name] for name in target], dtype=np.int64)


def joint_reorder_index(joint_names) -> np.ndarray:
    """Indices selecting G1 order from the GMR (or G1) joint columns."""
    return _reorder_index(list(joint_names), G1_29DOF_JOINT_NAMES, "joints")


def body_reorder_index(body_names) -> np.ndarray:
    """Indices selecting G1 bodies from the GMR (or G1) body columns."""
    names = list(body_names)
    mapped: dict[str, int] = {}
    for index, name in enumerate(names):
        target = g1_body_for(name)
        if target is None:
            continue
        if target in mapped:
            raise ValueError(f"bodies: {target!r} produced twice from HU_D04 names")
        mapped[target] = index
    if missing := [name for name in G1_32BODY_NAMES if name not in mapped]:
        raise ValueError(f"bodies: input cannot supply {missing}")
    return np.asarray([mapped[name] for name in G1_32BODY_NAMES], dtype=np.int64)


def convert_motion(input_path: Path) -> dict:
    """Convert one GMR HU_D04 motion to PHP's 29-joint / 32-body NPZ layout."""
    data = _load_npz(input_path)
    joint_names = [str(n) for n in data["joint_names"]]
    body_names = [str(n) for n in data["body_names"]]

    joint_idx = joint_reorder_index(joint_names)
    body_idx = body_reorder_index(body_names)

    fps_raw = np.asarray(data["fps"]).reshape(-1)
    fps = int(round(float(fps_raw[0])))
    root_pos = data["body_pos_w"][:, body_idx[0], :]
    root_quat = data["body_quat_w"][:, body_idx[0], :]
    root_lin_vel = data["body_lin_vel_w"][:, body_idx[0], :]
    root_ang_vel = data["body_ang_vel_w"][:, body_idx[0], :]

    # .npy members are the native Holosoma layout used by motion_convert.py:
    # joint_pos = root pos + root quat (wxyz) + 29 dofs, joint_vel = root
    # linear + angular velocity + 29 dofs.
    joint_pos = np.concatenate(
        [root_pos, root_quat, data["joint_pos"][:, joint_idx]], axis=1
    ).astype(np.float32)
    joint_vel = np.concatenate(
        [root_lin_vel, root_ang_vel, data["joint_vel"][:, joint_idx]], axis=1
    ).astype(np.float32)

    converted = {
        "fps": np.array([fps], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_names": np.array(G1_29DOF_JOINT_NAMES),
        "body_names": np.array(G1_32BODY_NAMES),
    }
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        converted[key] = data[key][:, body_idx, :].astype(np.float32)
    for passthrough in ("vel_cmd", "motion_ends", "motion_idxs"):
        if passthrough in data:
            converted[passthrough] = data[passthrough]
    return converted


def terrain_report(terrain_path: Path) -> dict:
    """Summarise a box-array terrain NPY without modifying it."""
    boxes = np.load(terrain_path, allow_pickle=False)
    if not isinstance(boxes, np.ndarray):
        boxes.close()
        raise ValueError(f"{terrain_path}: terrain must be a box-array NPY file")
    if boxes.ndim != 3 or boxes.shape[2] != 10:
        raise ValueError(f"{terrain_path}: expected (boxes, variants, 10) data")
    # Column 2 is the box centre height in the PHP box schema; sizes are
    # columns 6:9.
    heights = boxes[:, :, 2]
    sizes = boxes[:, :, 6:9]
    return {
        "path": str(terrain_path),
        "shape": list(boxes.shape),
        "max_height_m": float(heights.max()),
        "min_height_m": float(heights.min()),
        "max_size_m": [float(x) for x in sizes.reshape(-1, 3).max(axis=0)],
    }


def convert_bundle(input_motion: Path, terrain: Path, out_dir: Path) -> dict:
    """Write ``<stem>_motion.npz`` plus a paired terrain NPY and a report."""
    converted = convert_motion(input_motion)

    if not terrain.is_file():
        raise FileNotFoundError(f"terrain file not found: {terrain}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_motion.name.split("_motion")[0].removesuffix(".npz")
    motion_out = out_dir / f"{stem}_motion.npz"
    terrain_out = out_dir / f"{stem}_terrain.npy"
    np.savez(motion_out, **converted)
    shutil.copyfile(terrain, terrain_out)

    report = {
        "input": str(input_motion),
        "output": str(motion_out),
        "frames": int(converted["joint_pos"].shape[0]),
        "fps": int(converted["fps"][0]),
        "joints": int(converted["joint_names"].size),
        "bodies": int(converted["body_names"].size),
        "terrain": terrain_report(terrain_out),
    }
    (out_dir / f"{stem}_conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="GMR HU_D04 motion NPZ")
    parser.add_argument("output", type=Path, help="converted motion NPZ")
    parser.add_argument(
        "--terrain",
        type=Path,
        default=None,
        help="terrain NPY to pair (writes <stem>_motion.npz + <stem>_terrain.npy)",
    )
    args = parser.parse_args()

    if args.terrain is None:
        converted = convert_motion(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **converted)
        print(f"Converted {args.input} -> {args.output}")
        return

    report = convert_bundle(args.input, args.terrain, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())

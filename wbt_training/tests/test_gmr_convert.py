"""Tests for the GMR HU_D04 -> PHP motion/terrain converter."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wbt_training.config_values.robot_config import (
    G1_29DOF_JOINT_NAMES,
    G1_32BODY_NAMES,
)
from wbt_training.utils import gmr_convert as gc
from wbt_training.utils.registry import validate_bundle


def _gmr_like(tmp_path, *, joint_names=None, body_names=None, frames=8):
    """Write a GMR-shaped NPZ with distinguishable per-joint ramps."""
    joint_names = list(joint_names or gc.GMR_HU_D04_JOINT_NAMES)
    body_names = list(body_names or gc.GMR_HU_D04_BODY_NAMES)
    nj, nb = len(joint_names), len(body_names)
    # Each joint column carries its own constant so a reorder bug is visible.
    joint_pos = np.tile(np.arange(nj, dtype=np.float64), (frames, 1))
    joint_vel = joint_pos * 2.0
    body_pos = np.zeros((frames, nb, 3))
    body_pos[:, :, 0] = np.arange(nb, dtype=np.float64)
    path = tmp_path / "src_motion.npz"
    np.savez(
        path,
        fps=np.array([30.0]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, nb, 1)),
        body_lin_vel_w=np.zeros((frames, nb, 3)),
        body_ang_vel_w=np.zeros((frames, nb, 3)),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
    )
    return path


def test_tables_cover_g1_exactly():
    assert len(G1_29DOF_JOINT_NAMES) == 29
    assert len(G1_32BODY_NAMES) == 32
    kept = [j for j in gc.GMR_HU_D04_JOINT_NAMES if j not in gc.HEAD_JOINTS]
    assert set(kept) == set(G1_29DOF_JOINT_NAMES)
    assert len(gc.GMR_HU_D04_BODY_NAMES) == 42


def test_head_joints_removed(tmp_path):
    out = gc.convert_motion(_gmr_like(tmp_path))
    assert out["joint_names"].size == 29
    assert not any(name.startswith("head_") for name in out["joint_names"])


def test_joint_data_follows_names(tmp_path):
    """Reordered columns still carry the data of the same physical joint."""
    src = _gmr_like(tmp_path)
    raw = dict(np.load(src, allow_pickle=True))
    out = gc.convert_motion(src)

    source_index = {n: i for i, n in enumerate(raw["joint_names"])}
    expected = [source_index[name] for name in G1_29DOF_JOINT_NAMES]
    # Root occupies 7 columns (pos 3 + quat 4) before the 29 dofs.
    np.testing.assert_allclose(out["joint_pos"][:, 7:], raw["joint_pos"][:, expected])
    np.testing.assert_allclose(out["joint_vel"][:, 6:], raw["joint_vel"][:, expected])
    pelvis = list(raw["body_names"]).index("base_link")
    np.testing.assert_allclose(
        out["joint_pos"][:, :3], raw["body_pos_w"][:, pelvis, :]
    )
    np.testing.assert_allclose(
        out["joint_pos"][:, 3:7], raw["body_quat_w"][:, pelvis, :]
    )


def test_crop_is_name_based(tmp_path):
    """A shuffled input still crops and reorders by name."""
    rng = np.random.default_rng(0)
    joints = list(gc.GMR_HU_D04_JOINT_NAMES)
    bodies = list(gc.GMR_HU_D04_BODY_NAMES)
    rng.shuffle(joints)
    rng.shuffle(bodies)
    src = _gmr_like(tmp_path, joint_names=joints, body_names=bodies)
    raw = dict(np.load(src, allow_pickle=True))
    out = gc.convert_motion(src)

    source_index = {n: i for i, n in enumerate(raw["joint_names"])}
    expected = [source_index[name] for name in G1_29DOF_JOINT_NAMES]
    np.testing.assert_allclose(out["joint_pos"][:, 7:], raw["joint_pos"][:, expected])
    assert list(out["body_names"]) == G1_32BODY_NAMES


def test_body_map_is_lossless(tmp_path):
    out = gc.convert_motion(_gmr_like(tmp_path))
    assert list(out["body_names"]) == G1_32BODY_NAMES
    assert out["body_pos_w"].shape[1] == 32


def test_missing_body_is_an_error(tmp_path):
    bodies = [b for b in gc.GMR_HU_D04_BODY_NAMES if b != "contact_foot_center_L"]
    src = _gmr_like(tmp_path, body_names=bodies)
    with pytest.raises(ValueError, match="contact_foot_center_L|left_foot_contact_point"):
        gc.convert_motion(src)


def test_missing_joint_is_an_error(tmp_path):
    joints = [j for j in gc.GMR_HU_D04_JOINT_NAMES if j != "left_knee_joint"]
    src = _gmr_like(tmp_path, joint_names=joints)
    with pytest.raises(ValueError, match="left_knee_joint"):
        gc.convert_motion(src)


def test_injected_bad_name_trips_assertion(tmp_path):
    """A wrong name in the export must fail, not silently shift columns."""
    joints = list(gc.GMR_HU_D04_JOINT_NAMES)
    joints[joints.index("left_knee_joint")] = "left_knee_jiont"
    src = _gmr_like(tmp_path, joint_names=joints)
    with pytest.raises(ValueError, match="left_knee_joint"):
        gc.convert_motion(src)


def test_missing_field_is_reported(tmp_path):
    src = _gmr_like(tmp_path)
    raw = dict(np.load(src, allow_pickle=True))
    del raw["body_quat_w"]
    np.savez(src, **raw)
    with pytest.raises(ValueError, match="body_quat_w"):
        gc.convert_motion(src)


def test_bundle_validates_and_pairs_terrain(tmp_path):
    src = _gmr_like(tmp_path)
    terrain = tmp_path / "terrain_src.npy"
    boxes = np.zeros((2, 10, 10))
    boxes[:, :, 2] = 0.381910115480423
    boxes[:, :, 6:9] = [0.70, 1.2, 0.76]
    np.save(terrain, boxes)
    original = terrain.read_bytes()

    out_dir = tmp_path / "bundle"
    report = gc.convert_bundle(src, terrain, out_dir)

    assert (out_dir / "src_motion.npz").is_file()
    assert (out_dir / "src_terrain.npy").is_file()
    # Terrain is reused untouched.
    assert (out_dir / "src_terrain.npy").read_bytes() == original
    validate_bundle(out_dir)

    assert report["joints"] == 29
    assert report["bodies"] == 32
    on_disk = json.loads((out_dir / "src_conversion_report.json").read_text())
    assert on_disk["terrain"]["max_height_m"] == pytest.approx(0.381910115480423)
    assert on_disk["terrain"]["max_size_m"] == pytest.approx([0.70, 1.2, 0.76])

"""HU_D04 D435i depth camera config (tasks 5.1-5.3).

Mirrors Holosoma's ``G1FlatRsD435iConfig`` for HU_D04:

- camera parameters (resolution, HFOV, range, crop) copied from ``RsD435iConfig``
  so the unchanged depth backbone (input ``[1, 58, 87]``) accepts the image;
- mount pose taken from the HU_D04 URDF's ``waist_camera_joint``
  (parent ``waist_pitch_link``), the torso-parity choice matching G1's
  ``torso_link`` mount;
- ``ray_cast_bodies`` maps HU_D04 link names to their meshes.

Note the HU_D04 serial MJCF uses ``body_waist_*`` actuator names but
``waist_*_link`` body names; the ray-cast map keys are BODY names.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import mujoco

from hud04_sim2sim.model_loader import DEFAULT_XML, load_model

PACKAGE_ROOT = Path(__file__).resolve().parent

HU_D04_MESHES_ROOT = str(
    PACKAGE_ROOT / "meshes" / "HU_D04_01"
)

# --- Camera intrinsics/geometry: copied from RsD435iConfig (G1) ---
WIDTH = 106
HEIGHT = 60
HORIZONTAL_FOV_DEG = 89.5
MAX_RANGE = 3.0
MIN_RANGE = 0.3
CROP_TOP = 2
CROP_LEFT = 4
CROP_RIGHT = 4
# Post-crop resolution the backbone expects.
CROPPED_WIDTH = WIDTH - CROP_LEFT - CROP_RIGHT   # 98? verify below
CROPPED_HEIGHT = HEIGHT - CROP_TOP               # 58

# --- Mount pose from HU_D04_01.urdf waist_camera_joint ---
CAMERA_PARENT_BODY = "waist_pitch_link"
CAMERA_OFFSET_XYZ = (0.092, 0.0175, 0.2751)      # m
CAMERA_OFFSET_RPY = (0.0, 2.1818, 0.0)           # rad

# Base-frame rotation applied to the sensor element (matches RsD435iConfig).
OFFSET_ROT_BASE = [-90.0, 0.0, -90.0]            # roll, pitch, yaw [deg]


def _mesh_name(body: str) -> str:
    return f"{body}.STL"


# HU_D04 ray-cast bodies: body name -> mesh file (both verified to exist).
RAY_CAST_BODIES: dict[str, str] = {
    "base_link": _mesh_name("base_link"),
    "left_hip_pitch_link": _mesh_name("left_hip_pitch_link"),
    "left_hip_roll_link": _mesh_name("left_hip_roll_link"),
    "left_hip_yaw_link": _mesh_name("left_hip_yaw_link"),
    "left_knee_link": _mesh_name("left_knee_link"),
    "left_ankle_pitch_link": _mesh_name("left_ankle_pitch_link"),
    "left_ankle_roll_link": _mesh_name("left_ankle_roll_link"),
    "right_hip_pitch_link": _mesh_name("right_hip_pitch_link"),
    "right_hip_roll_link": _mesh_name("right_hip_roll_link"),
    "right_hip_yaw_link": _mesh_name("right_hip_yaw_link"),
    "right_knee_link": _mesh_name("right_knee_link"),
    "right_ankle_pitch_link": _mesh_name("right_ankle_pitch_link"),
    "right_ankle_roll_link": _mesh_name("right_ankle_roll_link"),
    "waist_yaw_link": _mesh_name("waist_yaw_link"),
    "waist_roll_link": _mesh_name("waist_roll_link"),
    "waist_pitch_link": _mesh_name("waist_pitch_link"),
    "left_shoulder_pitch_link": _mesh_name("left_shoulder_pitch_link"),
    "left_shoulder_roll_link": _mesh_name("left_shoulder_roll_link"),
    "left_shoulder_yaw_link": _mesh_name("left_shoulder_yaw_link"),
    "left_elbow_link": _mesh_name("left_elbow_link"),
    "left_wrist_yaw_link": _mesh_name("left_wrist_yaw_link"),
    "left_wrist_pitch_link": _mesh_name("left_wrist_pitch_link"),
    "left_wrist_roll_link": _mesh_name("left_wrist_roll_link"),
    "right_shoulder_pitch_link": _mesh_name("right_shoulder_pitch_link"),
    "right_shoulder_roll_link": _mesh_name("right_shoulder_roll_link"),
    "right_shoulder_yaw_link": _mesh_name("right_shoulder_yaw_link"),
    "right_elbow_link": _mesh_name("right_elbow_link"),
    "right_wrist_yaw_link": _mesh_name("right_wrist_yaw_link"),
    "right_wrist_pitch_link": _mesh_name("right_wrist_pitch_link"),
    "right_wrist_roll_link": _mesh_name("right_wrist_roll_link"),
}


@dataclasses.dataclass
class Hud04DepthCameraConfig:
    num_sensors: int = 1
    sensor_type: str = "camera"
    width: int = WIDTH
    height: int = HEIGHT
    horizontal_fov_deg: float = HORIZONTAL_FOV_DEG
    max_range: float = MAX_RANGE
    min_range: float = MIN_RANGE
    crop_top: int = CROP_TOP
    crop_left: int = CROP_LEFT
    crop_right: int = CROP_RIGHT
    dynamic_meshes: bool = True
    randomize_placement: bool = False
    asset_meshes_root: str = HU_D04_MESHES_ROOT
    base_link_frame: str = CAMERA_PARENT_BODY
    offset_pos: tuple[float, float, float] = CAMERA_OFFSET_XYZ
    offset_rot: tuple[float, float, float] = CAMERA_OFFSET_RPY
    offset_rot_base: list[float] = dataclasses.field(
        default_factory=lambda: list(OFFSET_ROT_BASE)
    )
    ray_cast_bodies: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(RAY_CAST_BODIES)
    )
    add_offpath_obstacle: bool = False

    @property
    def cropped_width(self) -> int:
        return self.width - self.crop_left - self.crop_right

    @property
    def cropped_height(self) -> int:
        return self.height - self.crop_top


def validate_camera_config(
    config: Hud04DepthCameraConfig | None = None,
    xml_path=DEFAULT_XML,
    meshes_root: Path | None = None,
) -> dict:
    """Resolve the camera config against the model and mesh directory.

    Raises AssertionError on any mismatch. Returns a small report.
    """
    config = config or Hud04DepthCameraConfig()
    model = load_model(xml_path)
    mesh_root = Path(meshes_root or config.asset_meshes_root)

    # Parent body must exist in the model.
    parent_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, config.base_link_frame
    )
    assert parent_id >= 0, (
        f"camera parent body {config.base_link_frame!r} not in model"
    )

    missing_bodies: list[str] = []
    missing_meshes: list[str] = []
    for body, mesh in config.ray_cast_bodies.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
            missing_bodies.append(body)
        if not (mesh_root / mesh).is_file():
            missing_meshes.append(mesh)
    assert not missing_bodies, f"ray_cast_bodies not in model: {missing_bodies}"
    assert not missing_meshes, f"ray_cast meshes missing: {missing_meshes}"

    assert config.cropped_height == 58, config.cropped_height
    # The G1 pipeline crops left/right by 4 each from width 106 -> 98, then the
    # backbone downsamples to 87. Confirm the raw geometry is consistent.
    assert config.cropped_width == 98, config.cropped_width

    return {
        "parent": config.base_link_frame,
        "offset_pos": config.offset_pos,
        "offset_rot": config.offset_rot,
        "cropped": (config.cropped_height, config.cropped_width),
        "ray_cast_bodies": len(config.ray_cast_bodies),
        "meshes_root": str(mesh_root),
    }


def main() -> int:
    report = validate_camera_config()
    for k, v in report.items():
        print(f"[hud04-camera] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

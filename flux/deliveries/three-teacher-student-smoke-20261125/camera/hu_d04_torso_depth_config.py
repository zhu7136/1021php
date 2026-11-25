"""Hu-D04 torso depth camera; deploy next to its generated JSON contract.

Mount transform is official URDF CAD data. Optical convention is a nominal
simulation convention, NOT a measured camera-to-link calibration.
"""
import json
from pathlib import Path

from holosoma.sensors.warp.camera_config.d435i_depth_config import RsD435iConfig

_CONTRACT = json.loads(Path(__file__).with_name('hu_d04_torso_camera_contract.json').read_text())


class HuD04TorsoRsD435iConfig(RsD435iConfig):
    base_link_frame = {'cam_front_depth': 'waist_pitch_link'}
    offset = {'cam_front_depth': {
        'offset_pos': tuple(_CONTRACT['mount_xyz_m']),
        # CameraSensor converts these degrees to radians internally.
        'offset_rot': tuple(_CONTRACT['mount_rpy_deg']),
    }}
    # Right multiplication: R_parent_optical = R_parent_CAD @ R_CAD_optical.
    # Nominal: optical +Z -> CAD +Z, optical +X -> CAD -Y,
    # optical +Y -> CAD +X. This is NOT the G1 [-90, 0, -90] transform.
    offset_rot_base = [0.0, 0.0, -90.0]
    ray_cast_bodies = dict(_CONTRACT['ray_cast_bodies'])
    # Deterministic placement until native-view validation passes.
    randomize_placement = False
    add_offpath_obstacle = False
    offpath_obstacle_meshes_root = None
    offpath_obstacle_bodies = {}
    optical_frame_calibrated = False
    native_render_verified = False

    def __init__(self, asset_meshes_root=None):
        if asset_meshes_root is None:
            # Expected existing archive layout: bundle/runtime/... and bundle/asset/...
            candidates = [p / 'asset/meshes/HU_D04_01' for p in Path(__file__).resolve().parents]
            asset_meshes_root = next((p for p in candidates if p.is_dir()), None)
        if asset_meshes_root is None:
            raise FileNotFoundError('Hu-D04 meshes not found; install under bundle/runtime or provide asset_meshes_root.')
        root = Path(asset_meshes_root).resolve()
        missing = [f for f in self.ray_cast_bodies.values() if not (root / f).is_file()]
        if missing:
            raise FileNotFoundError(f'Missing Hu-D04 self-occlusion meshes: {missing}')
        self.asset_meshes_root = str(root)


CAMERA_SENSOR_CFG = 'holosoma.sensors.warp.camera_config.hu_d04_torso_depth_config.HuD04TorsoRsD435iConfig'


def apply_torso_depth_config(observation):
    """Return a dataclass copy with ONLY depth camera class path changed.

    Dimensions, latency, PPO/student terms, command labels and teachers remain
    unchanged. This helper does not fix missing vel_cmd or launch training.
    """
    from dataclasses import replace
    groups = dict(observation.groups)
    group = groups['depth_camera']
    terms = dict(group.terms)
    term = terms['depth_cam']
    params = dict(term.params)
    params['camera_sensor_cfg'] = CAMERA_SENSOR_CFG
    terms['depth_cam'] = replace(term, params=params)
    groups['depth_camera'] = replace(group, terms=terms)
    return replace(observation, groups=groups)
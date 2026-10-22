"""G1 motor calibration and joint/body orderings used by PHP.

``build_g1_implicit_actuators`` remains at this path for saved checkpoints.
The literal orderings keep the no-MuJoCo motion converter usable with NumPy
alone. Integration tests check them against the pinned Holosoma robot.
"""

from __future__ import annotations

import numpy as np

G1_32BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_foot_contact_point",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_foot_contact_point",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ]
)


# ---------------------------------------------------------------------------
# G1 implicit-actuator motor calibration
# Registered on ``robot.control.implicit_actuator_builder`` when a preset sets
# ``control_mode="implicit_position_target"``. Tuned so IsaacSim's built-in PD
# drive applies torque with the correct stiffness/damping/armature.
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ


def build_g1_implicit_actuators():
    # Imported lazily so this module is importable without IsaacLab.
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": _STIFFNESS_7520_14,
                ".*_hip_roll_joint": _STIFFNESS_7520_22,
                ".*_hip_yaw_joint": _STIFFNESS_7520_14,
                ".*_knee_joint": _STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": _DAMPING_7520_14,
                ".*_hip_roll_joint": _DAMPING_7520_22,
                ".*_hip_yaw_joint": _DAMPING_7520_14,
                ".*_knee_joint": _DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": _ARMATURE_7520_14,
                ".*_hip_roll_joint": _ARMATURE_7520_22,
                ".*_hip_yaw_joint": _ARMATURE_7520_14,
                ".*_knee_joint": _ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFFNESS_5020,
            damping=2.0 * _DAMPING_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFFNESS_5020,
            damping=2.0 * _DAMPING_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_STIFFNESS_7520_14,
            damping=_DAMPING_7520_14,
            armature=_ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _STIFFNESS_5020,
                ".*_shoulder_roll_joint": _STIFFNESS_5020,
                ".*_shoulder_yaw_joint": _STIFFNESS_5020,
                ".*_elbow_joint": _STIFFNESS_5020,
                ".*_wrist_roll_joint": _STIFFNESS_5020,
                ".*_wrist_pitch_joint": _STIFFNESS_4010,
                ".*_wrist_yaw_joint": _STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": _DAMPING_5020,
                ".*_shoulder_roll_joint": _DAMPING_5020,
                ".*_shoulder_yaw_joint": _DAMPING_5020,
                ".*_elbow_joint": _DAMPING_5020,
                ".*_wrist_roll_joint": _DAMPING_5020,
                ".*_wrist_pitch_joint": _DAMPING_4010,
                ".*_wrist_yaw_joint": _DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": _ARMATURE_5020,
                ".*_shoulder_roll_joint": _ARMATURE_5020,
                ".*_shoulder_yaw_joint": _ARMATURE_5020,
                ".*_elbow_joint": _ARMATURE_5020,
                ".*_wrist_roll_joint": _ARMATURE_5020,
                ".*_wrist_pitch_joint": _ARMATURE_4010,
                ".*_wrist_yaw_joint": _ARMATURE_4010,
            },
        ),
    }


# ---------------------------------------------------------------------------
# HU_D04 implicit-actuator motor calibration
#
# Joint limits, effort and velocity come from
# ``humanoid-description/HU_D04_description/urdf/HU_D04_01.urdf``. Stiffness and
# damping come from LimX's deployment calibration
# (``limx_rl_lab/deploy/sim2sim/configs/limx_walk_mimic_sdk.yaml``), not from
# G1's motor constants: HU_D04's leg Kp is 2-5.5x G1's (hip pitch 220 vs 40.2,
# knee 240 vs 99.1, hip roll 200 vs 99.1). Wrist joints cap at 30/1.0.
#
# Only the 29 controlled joints appear; ``head_yaw_joint``/``head_pitch_joint``
# stay in the URDF unactuated.
# ---------------------------------------------------------------------------
_HU_D04_EFFORT_LIMIT = {
    "hip_pitch": 140.0,
    "hip_roll": 140.0,
    "hip_yaw": 140.0,
    "knee": 140.0,
    "ankle_pitch": 42.0,
    "ankle_roll": 42.0,
    "waist_yaw": 42.0,
    "waist_roll": 42.0,
    "waist_pitch": 42.0,
    "shoulder_pitch": 42.0,
    "shoulder_roll": 42.0,
    "shoulder_yaw": 42.0,
    "elbow": 42.0,
    "wrist_roll": 19.0,
    "wrist_pitch": 19.0,
    "wrist_yaw": 19.0,
}
_HU_D04_VELOCITY_LIMIT = {
    "hip_pitch": 5.0,
    "hip_roll": 5.0,
    "hip_yaw": 5.0,
    "knee": 5.0,
    "ankle_pitch": 13.6,
    "ankle_roll": 13.6,
    "waist_yaw": 13.6,
    "waist_roll": 13.6,
    "waist_pitch": 13.6,
    "shoulder_pitch": 19.6,
    "shoulder_roll": 19.6,
    "shoulder_yaw": 19.6,
    "elbow": 19.6,
    "wrist_roll": 13.0,
    "wrist_pitch": 13.0,
    "wrist_yaw": 13.0,
}
_HU_D04_STIFFNESS = {
    "hip_pitch": 220.0,
    "hip_roll": 200.0,
    "hip_yaw": 200.0,
    "knee": 240.0,
    "ankle_pitch": 60.0,
    "ankle_roll": 60.0,
    "waist_yaw": 120.0,
    "waist_roll": 90.0,
    "waist_pitch": 90.0,
    "shoulder_pitch": 80.0,
    "shoulder_roll": 80.0,
    "shoulder_yaw": 70.0,
    "elbow": 70.0,
    "wrist_roll": 30.0,
    "wrist_pitch": 30.0,
    "wrist_yaw": 30.0,
}
_HU_D04_DAMPING = {
    "hip_pitch": 6.0,
    "hip_roll": 5.0,
    "hip_yaw": 5.0,
    "knee": 6.0,
    "ankle_pitch": 1.5,
    "ankle_roll": 1.5,
    "waist_yaw": 3.0,
    "waist_roll": 2.5,
    "waist_pitch": 2.5,
    "shoulder_pitch": 2.5,
    "shoulder_roll": 2.5,
    "shoulder_yaw": 2.0,
    "elbow": 2.0,
    "wrist_roll": 1.0,
    "wrist_pitch": 1.0,
    "wrist_yaw": 1.0,
}


def _hu_d04_group(exprs, table):
    return {f".*_{name}_joint": table[name] for name in exprs}


def build_hu_d04_implicit_actuators():
    """Build the per-group implicit actuator config for HU_D04's 29 joints.

    ``robot.control.implicit_actuator_builder`` points here when a preset
    selects ``control_mode="implicit_position_target"``, so IsaacSim's built-in
    PD drive uses HU_D04's own stiffness/damping/limits.
    """
    # Imported lazily so this module is importable without IsaacLab.
    from isaaclab.actuators import ImplicitActuatorCfg

    legs = ["hip_pitch", "hip_roll", "hip_yaw", "knee"]
    arms = [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[f".*_{name}_joint" for name in legs],
            effort_limit_sim=_hu_d04_group(legs, _HU_D04_EFFORT_LIMIT),
            velocity_limit_sim=_hu_d04_group(legs, _HU_D04_VELOCITY_LIMIT),
            stiffness=_hu_d04_group(legs, _HU_D04_STIFFNESS),
            damping=_hu_d04_group(legs, _HU_D04_DAMPING),
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": _HU_D04_EFFORT_LIMIT["ankle_pitch"],
                ".*_ankle_roll_joint": _HU_D04_EFFORT_LIMIT["ankle_roll"],
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": _HU_D04_VELOCITY_LIMIT["ankle_pitch"],
                ".*_ankle_roll_joint": _HU_D04_VELOCITY_LIMIT["ankle_roll"],
            },
            stiffness={
                ".*_ankle_pitch_joint": _HU_D04_STIFFNESS["ankle_pitch"],
                ".*_ankle_roll_joint": _HU_D04_STIFFNESS["ankle_roll"],
            },
            damping={
                ".*_ankle_pitch_joint": _HU_D04_DAMPING["ankle_pitch"],
                ".*_ankle_roll_joint": _HU_D04_DAMPING["ankle_roll"],
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim={
                "waist_roll_joint": _HU_D04_EFFORT_LIMIT["waist_roll"],
                "waist_pitch_joint": _HU_D04_EFFORT_LIMIT["waist_pitch"],
            },
            velocity_limit_sim={
                "waist_roll_joint": _HU_D04_VELOCITY_LIMIT["waist_roll"],
                "waist_pitch_joint": _HU_D04_VELOCITY_LIMIT["waist_pitch"],
            },
            stiffness={
                "waist_roll_joint": _HU_D04_STIFFNESS["waist_roll"],
                "waist_pitch_joint": _HU_D04_STIFFNESS["waist_pitch"],
            },
            damping={
                "waist_roll_joint": _HU_D04_DAMPING["waist_roll"],
                "waist_pitch_joint": _HU_D04_DAMPING["waist_pitch"],
            },
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=_HU_D04_EFFORT_LIMIT["waist_yaw"],
            velocity_limit_sim=_HU_D04_VELOCITY_LIMIT["waist_yaw"],
            stiffness=_HU_D04_STIFFNESS["waist_yaw"],
            damping=_HU_D04_DAMPING["waist_yaw"],
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[f".*_{name}_joint" for name in arms],
            effort_limit_sim=_hu_d04_group(arms, _HU_D04_EFFORT_LIMIT),
            velocity_limit_sim=_hu_d04_group(arms, _HU_D04_VELOCITY_LIMIT),
            stiffness=_hu_d04_group(arms, _HU_D04_STIFFNESS),
            damping=_hu_d04_group(arms, _HU_D04_DAMPING),
        ),
    }

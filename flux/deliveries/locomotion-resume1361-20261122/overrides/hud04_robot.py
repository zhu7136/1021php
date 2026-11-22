"""HuD04 PR-space calibration candidate, referenced to beifen1001 aca4fd0.

Not a runnable robot/experiment preset. Values reproduce that repository's
simulation calibration, not certified hardware ratings or AB motor gains.
"""
from __future__ import annotations

REFERENCE_COMMIT = "aca4fd023e0b14015ccda85da8bc71a00792b659"
HEAD_JOINT_NAMES = ("head_yaw_joint", "head_pitch_joint")
POLICY_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")
)
SDK_JOINT_NAMES = POLICY_JOINT_NAMES + HEAD_JOINT_NAMES
TRACKED_BODY_NAMES = (
    "base_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "waist_pitch_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_roll_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_roll_link",
)
ANCHOR_BODY_NAME = "waist_pitch_link"

# suffix: Kp, Kd, effort_limit_sim, velocity_limit_sim, armature
_CALIBRATION = {
    "hip_pitch": (220., 6., 140., 5., .14125),
    "hip_roll": (200., 5., 140., 5., .14125),
    "hip_yaw": (200., 5., 140., 5., .14125),
    "knee": (240., 6., 140., 5., .14125),
    "ankle_pitch": (60., 1.5, 42., 13.6, .1845504),
    "ankle_roll": (60., 1.5, 42., 13.6, .1845504),
    "waist_yaw": (120., 3., 42., 13.6, .1845504),
    "waist_roll": (90., 2.5, 42., 13.6, .1845504),
    "waist_pitch": (90., 2.5, 42., 13.6, .1845504),
    "shoulder_pitch": (80., 2.5, 42., 19.6, .0886706),
    "shoulder_roll": (80., 2.5, 42., 19.6, .0886706),
    "shoulder_yaw": (70., 2., 42., 19.6, .0886706),
    "elbow": (70., 2., 42., 19.6, .0886706),
    "wrist_yaw": (30., 1., 19., 13., .0153218),
    "wrist_pitch": (30., 1., 19., 13., .0153218),
    "wrist_roll": (30., 1., 19., 13., .0153218),
    "head_yaw": (20., .8, 19., 13., .0153218),
    "head_pitch": (20., .8, 19., 13., .0153218),
}


def calibration(name):
    if name not in SDK_JOINT_NAMES:
        raise ValueError(f"Unknown HuD04 PR joint: {name}")
    suffix = name.removesuffix("_joint")
    if suffix.startswith(("left_", "right_")):
        suffix = suffix.split("_", 1)[1]
    return _CALIBRATION[suffix]


def default_position(name):
    calibration(name)
    for suffix, value in (("hip_pitch_joint", -.25), ("knee_joint", .55), ("ankle_pitch_joint", -.30),
                          ("shoulder_pitch_joint", .10), ("elbow_joint", -.20)):
        if name.endswith(suffix):
            return value
    return {"left_shoulder_roll_joint": .10, "right_shoulder_roll_joint": -.10,
            "left_shoulder_yaw_joint": -.20, "right_shoulder_yaw_joint": .20}.get(name, 0.)


def policy_action_scales():
    """BeyondMimic reference scaling; not the constant-scale walking policy."""
    return tuple(.25 * calibration(name)[2] / calibration(name)[0] for name in POLICY_JOINT_NAMES)


def resolve_policy_indices(physical_names):
    names = tuple(physical_names)
    if len(names) != 31 or len(set(names)) != 31 or set(names) != set(SDK_JOINT_NAMES):
        raise ValueError("Expected exactly the 31 unique HuD04 PR joints; AB/passive joints are unsupported")
    return tuple(names.index(name) for name in POLICY_JOINT_NAMES)


def build_hud04_implicit_actuators():
    """Lazy IsaacLab construction; one named actuator per PR joint, including head."""
    from isaaclab.actuators import ImplicitActuatorCfg
    result = {}
    for name in SDK_JOINT_NAMES:
        kp, kd, effort, velocity, armature = calibration(name)
        result[name] = ImplicitActuatorCfg(joint_names_expr=[name], stiffness=kp, damping=kd,
            effort_limit_sim=effort, velocity_limit_sim=velocity, armature=armature)
    return result
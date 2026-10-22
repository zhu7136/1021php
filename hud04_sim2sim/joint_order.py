"""HU_D04 joint orderings used by the migration layer.

Two orderings matter:

1. ``HUD04_POLICY_JOINTS_31`` — the 31 actuated joints in the order LimX's
   own sim2sim config lists them (``limx_walk_mimic_sdk.yaml``). This is the
   order the LimX policy/observation pipeline uses.
2. ``HUD04_POLICY_JOINTS_29`` — the same list with the two head joints
   removed, matching the G1 29-DOF policy-facing space.

``G1_29DOF_JOINT_NAMES`` is imported from the PHP training config so the two
sides cannot drift apart silently.
"""

from __future__ import annotations

from wbt_training.config_values.robot_config import G1_29DOF_JOINT_NAMES

from hud04_sim2sim.head_lock import strip_head

# Order taken verbatim from limx_walk_mimic_sdk.yaml `policy_joints`.
HUD04_POLICY_JOINTS_31: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
)

HUD04_POLICY_JOINTS_29: tuple[str, ...] = tuple(strip_head(list(HUD04_POLICY_JOINTS_31)))

G1_JOINTS_29: tuple[str, ...] = tuple(G1_29DOF_JOINT_NAMES)


def validate() -> None:
    assert len(HUD04_POLICY_JOINTS_31) == 31, len(HUD04_POLICY_JOINTS_31)
    assert len(HUD04_POLICY_JOINTS_29) == 29, len(HUD04_POLICY_JOINTS_29)
    assert len(G1_JOINTS_29) == 29, len(G1_JOINTS_29)
    assert not any("head" in j for j in HUD04_POLICY_JOINTS_29)


def main() -> int:
    validate()
    print(f"[hud04] HU_D04 policy joints (31): {len(HUD04_POLICY_JOINTS_31)}")
    print(f"[hud04] HU_D04 policy joints (29): {len(HUD04_POLICY_JOINTS_29)}")
    print(f"[hud04] G1 joints (29): {len(G1_JOINTS_29)}")
    missing = set(G1_JOINTS_29) - set(HUD04_POLICY_JOINTS_29)
    extra = set(HUD04_POLICY_JOINTS_29) - set(G1_JOINTS_29)
    print(f"[hud04] joints in G1 but not HU_D04: {sorted(missing)}")
    print(f"[hud04] joints in HU_D04 but not G1: {sorted(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

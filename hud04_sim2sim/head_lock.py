"""Head degree-of-freedom locking for HU_D04.

HU_D04 has two head joints (`head_yaw_joint`, `head_pitch_joint`) on top of
the 29-DOF body used by the G1 policy. Locking them at their default angles
makes the HU_D04 policy-facing joint space exactly 29, matching G1.

This mirrors the `head_hold_targets` mechanism in LimX's
`deploy/sim2sim/gamepad_policy_controller.py`: the policy joint list excludes
the head; the controller publishes a fixed default target for the head every
control step.

Values are copied from `limx_walk_mimic_sdk.yaml`
(`default_angles`, `kps`, `kds` at indices 11 and 16).
"""

from __future__ import annotations

import dataclasses

# From limx_walk_mimic_sdk.yaml: policy_joints index 11 and 16.
HEAD_HOLD_TARGETS: dict[str, tuple[float, float, float]] = {
    "head_yaw_joint": (0.0, 20.0, 0.8),
    "head_pitch_joint": (0.0, 20.0, 0.8),
}

HEAD_JOINT_NAMES: tuple[str, ...] = tuple(HEAD_HOLD_TARGETS)


@dataclasses.dataclass(frozen=True)
class HeadHoldCommand:
    joint: str
    q: float
    kp: float
    kd: float


def head_hold_commands() -> list[HeadHoldCommand]:
    """Return the fixed head targets to publish each control step."""
    return [
        HeadHoldCommand(joint=name, q=q, kp=kp, kd=kd)
        for name, (q, kp, kd) in HEAD_HOLD_TARGETS.items()
    ]


def strip_head(joint_names: list[str]) -> list[str]:
    """Return `joint_names` with the head joints removed.

    Used to derive the 29-DOF policy joint list from a 31-joint list.
    """
    return [j for j in joint_names if j not in HEAD_HOLD_TARGETS]


def policy_joint_count(joint_names: list[str]) -> int:
    return len(strip_head(joint_names))


def main() -> int:
    names = list(HEAD_HOLD_TARGETS)
    print(f"[hud04] head joints: {names}")
    for cmd in head_hold_commands():
        print(f"  {cmd.joint:18s} q={cmd.q} kp={cmd.kp} kd={cmd.kd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Action scaling adapter: G1 student output -> HU_D04 PD targets.

Pipeline (mirrors Holosoma's ``BasePolicy`` post-processing):

    q_target = policy_action * action_scale + default_dof_angles

Verified facts about the G1 depth-student ONNX
(``~/.cache/php-parkour/student/student.onnx``):

- Output ``actions`` shape ``[1, 29]`` -> **normalized action**, not joint angles.
- ONNX metadata ``action_scale`` is all ``1.0`` (per-joint tensor).
- ONNX metadata ``default_joint_pos`` holds the G1 default angles.
- Holosoma inference: ``scaled = policy_action * action_scale`` then
  ``q_target = scaled + default_dof_angles``.

IMPORTANT: ``default_dof_angles`` here is the ``default_joint_pos`` from the
G1 student's own metadata (permuted to HU_D04 joint order), **not** the
HU_D04 robot's own default pose. The policy's action is a delta relative to
the defaults it was trained against; using the robot's own default pose as the
offset shifts every joint and makes the policy fall over. See
``configs/hud04_action_adapter.json`` ``default_source``.

The G1 action is first permuted into HU_D04 order via
``hud04_sim2sim.joint_mapping``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from hud04_sim2sim import joint_mapping
from hud04_sim2sim.joint_order import HUD04_POLICY_JOINTS_29

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "hud04_action_adapter.json"


@dataclasses.dataclass
class ActionAdapterConfig:
    action_scale: np.ndarray           # (29,) per-joint scale in HU_D04 order
    default_dof_angles: np.ndarray     # (29,) default angles in HU_D04 order
    joint_names: list[str]
    scale_source: str                  # provenance for the chosen scale
    source: str                        # e.g. "g1_student_onnx"


def _load_config(path: Path = DEFAULT_CONFIG) -> ActionAdapterConfig:
    data = json.loads(Path(path).read_text())
    return ActionAdapterConfig(
        action_scale=np.asarray(data["action_scale"], dtype=np.float32),
        default_dof_angles=np.asarray(data["default_dof_angles"], dtype=np.float32),
        joint_names=list(data["joint_names"]),
        scale_source=data["scale_source"],
        source=data["source"],
    )


class ActionAdapter:
    """Applies joint reorder + scale + default offset to produce q_target."""

    def __init__(self, config: ActionAdapterConfig | None = None,
                 mapping: tuple[int, ...] = joint_mapping.JOINT_IDS_MAP):
        self.config = config or _load_config()
        self.mapping = mapping
        self._validate()

    def _validate(self) -> None:
        n = len(HUD04_POLICY_JOINTS_29)
        if len(self.config.joint_names) != n:
            raise ValueError(
                f"joint_names length {len(self.config.joint_names)} != {n}"
            )
        if list(self.config.joint_names) != list(HUD04_POLICY_JOINTS_29):
            raise ValueError("config joint_names must match HUD04_POLICY_JOINTS_29 order")
        if self.config.action_scale.shape != (n,):
            raise ValueError(f"action_scale shape {self.config.action_scale.shape} != ({n},)")
        if self.config.default_dof_angles.shape != (n,):
            raise ValueError(
                f"default_dof_angles shape {self.config.default_dof_angles.shape} != ({n},)"
            )
        if len(self.mapping) != n:
            raise ValueError(f"mapping length {len(self.mapping)} != {n}")

    def to_hud04_q_target(self, g1_action: np.ndarray) -> np.ndarray:
        """Convert a G1-ordered action (shape (29,) or (1,29)) to HU_D04 q_target."""
        flat = np.asarray(g1_action, dtype=np.float32)
        squeeze = flat.ndim == 2
        if squeeze:
            flat = flat.reshape(-1)
        if flat.shape[0] != len(self.mapping):
            raise ValueError(f"expected {len(self.mapping)} action dims, got {flat.shape[0]}")

        reordered = joint_mapping.reorder(flat, self.mapping)
        reordered = np.asarray(reordered, dtype=np.float32)
        q_target = reordered * self.config.action_scale + self.config.default_dof_angles
        return q_target.reshape(1, -1) if squeeze else q_target

    def head_hold_command(self) -> dict[str, tuple[float, float, float]]:
        """Head (joint -> (q, kp, kd)) to publish every control step."""
        from hud04_sim2sim.head_lock import HEAD_HOLD_TARGETS
        return dict(HEAD_HOLD_TARGETS)


def main() -> int:
    adapter = ActionAdapter()
    print(f"[hud04] adapter source: {adapter.config.source}")
    print(f"[hud04] scale_source: {adapter.config.scale_source}")
    print(f"[hud04] action_scale (unique): {sorted(set(adapter.config.action_scale.tolist()))}")
    # identity action -> should yield default angles
    zeros = np.zeros(29, dtype=np.float32)
    q = adapter.to_hud04_q_target(zeros)
    print(f"[hud04] zero-action q_target == default: "
          f"{np.allclose(q, adapter.config.default_dof_angles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

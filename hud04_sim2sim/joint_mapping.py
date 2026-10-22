"""G1 -> HU_D04 joint mapping and action-vector reordering.

G1 and HU_D04 share the same 29 joint *names* (verified by
``hud04_sim2sim.joint_order``); only the *order* differs. So the mapping is a
pure permutation: build an index vector that takes a G1-ordered action vector
and produces an HU_D04-policy-ordered action vector.

``joint_ids_map`` convention (matching LimX deploy configs):
    hud04_action[hud04_index] = g1_action[ joint_ids_map[hud04_index] ]
"""

from __future__ import annotations

from hud04_sim2sim.joint_order import G1_JOINTS_29, HUD04_POLICY_JOINTS_29


class JointMappingError(ValueError):
    """Raised when the G1->HU_D04 joint mapping cannot be built or applied."""


def build_joint_ids_map(
    source_joints: tuple[str, ...] = G1_JOINTS_29,
    target_joints: tuple[str, ...] = HUD04_POLICY_JOINTS_29,
) -> tuple[int, ...]:
    """Return `joint_ids_map` mapping target index -> source index.

    Raises JointMappingError if any target joint is absent from, or any
    source joint has no target, or lengths differ.
    """
    if len(source_joints) != len(target_joints):
        raise JointMappingError(
            f"joint count mismatch: source={len(source_joints)} target={len(target_joints)}"
        )
    source_index = {name: i for i, name in enumerate(source_joints)}
    if len(source_index) != len(source_joints):
        raise JointMappingError("source joint names contain duplicates")
    if len(set(target_joints)) != len(target_joints):
        raise JointMappingError("target joint names contain duplicates")

    missing = [j for j in target_joints if j not in source_index]
    if missing:
        raise JointMappingError(f"target joints absent from source: {missing}")

    mapping = tuple(source_index[j] for j in target_joints)
    if sorted(mapping) != list(range(len(target_joints))):
        raise JointMappingError(
            "mapping is not a permutation; source and target joint sets differ"
        )
    return mapping


JOINT_IDS_MAP: tuple[int, ...] = build_joint_ids_map()


def reorder(source_action, mapping: tuple[int, ...] = JOINT_IDS_MAP):
    """Reorder a G1-ordered action vector into HU_D04 policy order.

    Accepts a sequence or a 1-D numpy array and returns the same type.
    Validates the mapping at call time (length + index range).
    """
    import numpy as np  # local import keeps module import-light

    if len(mapping) != len(set(mapping)):
        raise JointMappingError("joint_ids_map has duplicate indices")
    n = len(mapping)
    if max(mapping) >= n or min(mapping) < 0:
        raise JointMappingError(
            f"joint_ids_map index out of range [0,{n}): min={min(mapping)} max={max(mapping)}"
        )

    if isinstance(source_action, np.ndarray):
        flat = source_action
        if flat.ndim == 1:
            if flat.shape[0] < n:
                raise JointMappingError(
                    f"source action has {flat.shape[0]} dims, need >= {n}"
                )
            return flat[list(mapping)]
        if flat.ndim == 2:
            if flat.shape[1] < n:
                raise JointMappingError(
                    f"source action has {flat.shape[1]} dims, need >= {n}"
                )
            return flat[:, list(mapping)]
        raise JointMappingError(f"unsupported action ndim: {flat.ndim}")

    seq = list(source_action)
    if len(seq) < n:
        raise JointMappingError(f"source action has {len(seq)} dims, need >= {n}")
    return [seq[i] for i in mapping]


def describe() -> str:
    lines = ["joint_ids_map (HU_D04 target index -> G1 source index):"]
    for i, src in enumerate(JOINT_IDS_MAP):
        lines.append(f"  {i:2d} {HUD04_POLICY_JOINTS_29[i]:28s} <- G1[{src:2d}] {G1_JOINTS_29[src]}")
    return "\n".join(lines)


def main() -> int:
    print(describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

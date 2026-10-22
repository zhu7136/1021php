"""HU_D04 model integration for the PHP MuJoCo sim2sim flow.

Loads the serial-ankle variant of the HU_D04 MJCF (motors on the serial
`ankle_pitch` / `ankle_roll` hinge pair, no achilles A/B actuators) and
exposes the actuator/joint tables the rest of the migration layer needs.

Usage::

    python -m hud04_sim2sim.model_loader --check
    python -m hud04_sim2sim.model_loader --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import mujoco

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_XML = PACKAGE_ROOT / "assets" / "HU_D04_01_serial.xml"
DEFAULT_ACTUATOR_CONFIG = PACKAGE_ROOT / "configs" / "hud04_actuators.json"

# Serial ankle hinge joints that the policy drives (HU_D04_01_gmr variant).
SERIAL_ANKLE_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
# Parallel achilles actuator names that MUST NOT appear in the serial variant.
PARALLEL_ACHILLES_ACTUATORS = (
    "ankle_A_left",
    "ankle_B_left",
    "ankle_A_right",
    "ankle_B_right",
)


@dataclasses.dataclass(frozen=True)
class ActuatorInfo:
    index: int
    actuator: str
    joint: str
    joint_range: tuple[float, float]
    ctrl_range: tuple[float, float]
    gear: float


@dataclasses.dataclass
class Hud04ModelInfo:
    xml_path: str
    nq: int
    nv: int
    nu: int
    nbody: int
    free_joint_count: int
    actuators: list[ActuatorInfo]
    serial_ankles_present: bool
    achilles_actuators_present: bool


def load_model(xml_path: Path | str = DEFAULT_XML) -> mujoco.MjModel:
    """Load the HU_D04 serial-ankle MJCF into MuJoCo."""
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"HU_D04 MJCF not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _name(model: mujoco.MjModel, objtype: int, idx: int) -> str:
    return mujoco.mj_id2name(model, objtype, idx) or ""


def describe_model(model: mujoco.MjModel, xml_path: Path | str = DEFAULT_XML) -> Hud04ModelInfo:
    """Collect the model facts needed by the migration layer."""
    free_joints = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]

    actuators: list[ActuatorInfo] = []
    actuator_names: list[str] = []
    for i in range(model.nu):
        aname = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = int(model.actuator_trnid[i, 0])
        jname = _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.jnt_range[jid]
        clo, chi = model.actuator_ctrlrange[i]
        actuators.append(
            ActuatorInfo(
                index=i,
                actuator=aname,
                joint=jname,
                joint_range=(float(lo), float(hi)),
                ctrl_range=(float(clo), float(chi)),
                gear=float(model.actuator_gear[i, 0]),
            )
        )
        actuator_names.append(aname)

    joint_names = {
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
    }
    serial_ankles_present = all(j in joint_names for j in SERIAL_ANKLE_JOINTS)
    achilles_actuators_present = any(
        a in actuator_names for a in PARALLEL_ACHILLES_ACTUATORS
    )

    return Hud04ModelInfo(
        xml_path=str(xml_path),
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        nbody=model.nbody,
        free_joint_count=len(free_joints),
        actuators=actuators,
        serial_ankles_present=serial_ankles_present,
        achilles_actuators_present=achilles_actuators_present,
    )


def check_model(xml_path: Path | str = DEFAULT_XML) -> Hud04ModelInfo:
    """Load and assert the invariants the migration layer depends on.

    Raises AssertionError with a specific message when an invariant fails.
    """
    model = load_model(xml_path)
    info = describe_model(model, xml_path)

    assert info.free_joint_count == 1, (
        f"expected exactly 1 free joint (floating base), got {info.free_joint_count}"
    )
    assert info.serial_ankles_present, (
        "serial ankle joints missing; need the serial-ankle MJCF variant "
        f"(expected all of {SERIAL_ANKLE_JOINTS})"
    )
    assert not info.achilles_actuators_present, (
        "parallel achilles actuators present; this is the parallel-ankle model, "
        "not the serial variant: " + ", ".join(PARALLEL_ACHILLES_ACTUATORS)
    )
    # Every actuator must map to a HINGE joint.
    for act in info.actuators:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, act.joint)
        assert jid >= 0, f"actuator {act.actuator!r} has no joint"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, (
            f"actuator {act.actuator!r} drives non-hinge joint {act.joint!r}"
        )
    return info


def _to_json(info: Hud04ModelInfo) -> str:
    return json.dumps(dataclasses.asdict(info), indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="HU_D04 MJCF path")
    parser.add_argument("--check", action="store_true", help="assert migration invariants")
    parser.add_argument("--json", action="store_true", help="emit model info as JSON")
    args = parser.parse_args(argv)

    if args.check:
        info = check_model(args.xml)
        print(f"[hud04] model OK: nq={info.nq} nv={info.nv} nu={info.nu} "
              f"nbody={info.nbody} free_joints={info.free_joint_count}")
        print(f"[hud04] serial ankles present: {info.serial_ankles_present}")
        print(f"[hud04] achilles actuators present: {info.achilles_actuators_present}")
        return 0

    info = describe_model(load_model(args.xml), args.xml)
    if args.json:
        print(_to_json(info))
    else:
        print(f"nq={info.nq} nv={info.nv} nu={info.nu} nbody={info.nbody}")
        for act in info.actuators:
            print(f"  {act.index:2d} {act.actuator:22s} -> {act.joint:28s} "
                  f"range={list(act.joint_range)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

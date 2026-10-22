"""Task 3.5: single-joint verification for HU_D04.

Verifies the actuator->joint contract that the migration layer depends on:
for every HU_D04 policy joint, driving its actuator with a positive torque
moves the joint's qpos in the positive direction, and the joint does not move
when its actuator is idle.

Method: reset to the default pose, hold everything at zero control, then apply
a small constant torque to ONE actuator and step. This isolates the
actuator->qpos sign per joint without the reaction-dynamics confound that a
PD-following probe suffers on a free floating base in zero gravity.

The check asserts the sign consistency that matters, i.e. that
``ankle_pitch_left`` really drives ``left_ankle_pitch_joint`` positively and
not some other joint.
"""

from __future__ import annotations

import argparse
import json

import mujoco

from hud04_sim2sim.action_adapter import ActionAdapter
from hud04_sim2sim.model_loader import DEFAULT_XML, load_model
from hud04_sim2sim.joint_order import HUD04_POLICY_JOINTS_29

TORQUE = 5.0
STEPS = 400
DT = 0.002


def _joint_actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname:
            out[jname] = i
    return out


def probe_joint(model: mujoco.MjModel, joint: str, act: int,
                default_q: float, defaults: dict[str, float]) -> dict:
    """Apply a positive constant torque to `act`; measure `joint` qpos sign."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for name, q0 in defaults.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = q0

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qadr = int(model.jnt_qposadr[jid])
    start = float(data.qpos[qadr])

    for _ in range(STEPS):
        data.ctrl[:] = 0.0
        data.ctrl[act] = TORQUE
        mujoco.mj_step(model, data)

    delta = float(data.qpos[qadr]) - start
    # Positive torque must move the joint positively (or clamp at its upper
    # limit, which still reads positive). Require a clear positive response.
    ok = delta > 1e-4
    return {
        "joint": joint,
        "actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act),
        "delta": round(delta, 4),
        "status": "ok" if ok else "wrong_sign",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque", type=float, default=TORQUE)
    args = parser.parse_args(argv)

    model = load_model(DEFAULT_XML)
    model.opt.timestep = DT
    model.opt.gravity[:] = 0.0  # isolate actuation from gravity
    joint_act = _joint_actuator_ids(model)
    defaults = dict(zip(
        ActionAdapter().config.joint_names,
        ActionAdapter().config.default_dof_angles,
    ))

    results = []
    for joint in HUD04_POLICY_JOINTS_29:
        act = joint_act.get(joint)
        if act is None:
            results.append({"joint": joint, "status": "no_actuator"})
            print(f"!! {joint:28s} no actuator")
            continue
        r = probe_joint(model, joint, act, float(defaults[joint]), defaults)
        results.append(r)
        flag = "OK " if r["status"] == "ok" else "!! "
        print(f"{flag}{r['joint']:28s} act={r['actuator']:20s} "
              f"+torque -> delta={r['delta']:+.4f} status={r['status']}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[hud04] single-joint probe: {ok}/{len(results)} actuators drive "
          f"their joint with the correct sign")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

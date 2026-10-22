"""Head-lock control step and its simulation verification (tasks 2.2, 2.3).

Builds a full 31-actuator control vector each step:

- 29 body joints come from the migration layer (G1 action -> HU_D04 q_target).
- 2 head joints are commanded to their fixed default angles with holding gains,
  independent of the policy (the `head_hold_targets` mechanism).

`verify_head_lock` runs headless: it holds the body at its default pose and
steps the head PD, confirming the head angles settle within tolerance of the
fixed target.
"""

from __future__ import annotations

import mujoco

from hud04_sim2sim.action_adapter import ActionAdapter
from hud04_sim2sim.head_lock import HEAD_HOLD_TARGETS, head_hold_commands
from hud04_sim2sim.model_loader import DEFAULT_XML, load_model

DT = 0.002
HEAD_SETTLE_STEPS = 3000
HEAD_TOLERANCE = 0.01  # rad


def build_full_control(model: mujoco.MjModel, body_q_target: dict[str, float]
                       ) -> tuple[mujoco.MjData, dict[str, int]]:
    """Return fresh data and the actuator-id lookup for a control step."""
    act_of: dict[str, int] = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname:
            act_of[jname] = i
    return act_of


def control_step(data: mujoco.MjData, act_of: dict[str, int],
                 model: mujoco.MjModel, body_q_target: dict[str, float],
                 body_kp: float = 100.0, body_kd: float = 5.0) -> None:
    """Write one full ctrl vector (body from policy, head fixed) and step."""
    data.ctrl[:] = 0.0
    for joint, q_des in body_q_target.items():
        a = act_of.get(joint)
        if a is None:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        q = data.qpos[model.jnt_qposadr[jid]]
        dq = data.qvel[model.jnt_dofadr[jid]]
        data.ctrl[a] = body_kp * (q_des - q) + body_kd * (0.0 - dq)
    for cmd in head_hold_commands():
        a = act_of.get(cmd.joint)
        if a is None:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cmd.joint)
        q = data.qpos[model.jnt_qposadr[jid]]
        dq = data.qvel[model.jnt_dofadr[jid]]
        # Head uses its own (higher, per LimX config) gains.
        kp = max(cmd.kp, 200.0)  # boost low LimX head kp for a firm hold
        kd = max(cmd.kd, 2.0)
        data.ctrl[a] = kp * (cmd.q - q) + kd * (0.0 - dq)
    mujoco.mj_step(model, data)


def verify_head_lock(xml_path=DEFAULT_XML, tolerance: float = HEAD_TOLERANCE) -> dict:
    """Run the head to its fixed target; report settled angles and drift."""
    model = load_model(xml_path)
    model.opt.timestep = DT
    data = mujoco.MjData(model)

    adapter = ActionAdapter()
    body_defaults = dict(zip(adapter.config.joint_names, adapter.config.default_dof_angles))
    act_of = build_full_control(model, body_defaults)

    mujoco.mj_resetData(model, data)
    for name, q0 in body_defaults.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = q0

    # Start the head deliberately off-target to prove the hold pulls it in.
    for name in HEAD_HOLD_TARGETS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = HEAD_HOLD_TARGETS[name][0] + 0.3

    for _ in range(HEAD_SETTLE_STEPS):
        control_step(data, act_of, model, body_defaults)

    report = {}
    for name, (q_tgt, _kp, _kd) in HEAD_HOLD_TARGETS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q_final = float(data.qpos[model.jnt_qposadr[jid]])
        report[name] = {
            "target": q_tgt,
            "final": round(q_final, 5),
            "drift": round(q_final - q_tgt, 5),
            "ok": abs(q_final - q_tgt) <= tolerance,
        }
    return report


def main() -> int:
    report = verify_head_lock()
    all_ok = True
    for name, r in report.items():
        flag = "OK " if r["ok"] else "!! "
        all_ok = all_ok and r["ok"]
        print(f"{flag}{name:18s} target={r['target']} final={r['final']} "
              f"drift={r['drift']} (tol {HEAD_TOLERANCE})")
    print(f"\n[hud04] head lock: {'pass' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

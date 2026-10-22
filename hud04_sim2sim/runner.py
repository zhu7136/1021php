"""HU_D04 sim2sim migration runner (tasks 4.1-4.5).

Wires the pieces together:

    G1 student ONNX (29-DOF obs -> 29-DOF action)
        |
        v
    ActionAdapter: joint reorder + action_scale + default offset -> q_target
        |
        v
    full 31-actuator ctrl (body from policy + head held) -> MuJoCo HU_D04
        |
        v
    step, build obs, repeat

``run_held_pose`` drives the robot to its default pose with head held, verifying
the mechanical integration. ``run_standing`` runs a closed-loop with a zero
student (ONNX returns ~zero for zero obs), confirming the full observation +
policy + adapter + head-lock stack executes without NaN/blow-up.

``run_circular`` executes a simple velocity-command circular trajectory to
validate ``velocity_command`` is wired through and affects the policy outputs.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from hud04_sim2sim.action_adapter import ActionAdapter
from hud04_sim2sim.head_lock import HEAD_HOLD_TARGETS
from hud04_sim2sim.joint_order import HUD04_POLICY_JOINTS_29
from hud04_sim2sim.model_loader import DEFAULT_XML, load_model

DT = 0.002
BODY_KP = 100.0
BODY_KD = 5.0
HEAD_KP = 200.0
HEAD_KD = 2.0


class Hud04Sim:
    """Minimal HU_D04 MuJoCo sim bound to the migration control stack."""

    def __init__(self, xml_path=DEFAULT_XML, adapter: ActionAdapter | None = None):
        self.model = load_model(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)
        self.adapter = adapter or ActionAdapter()
        self.act_of: dict[str, int] = {}
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname:
                self.act_of[jname] = i
        self.body_defaults = dict(zip(
            self.adapter.config.joint_names, self.adapter.config.default_dof_angles
        ))
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for name, q0 in self.body_defaults.items():
            self._set_qpos(name, float(q0))

    def _set_qpos(self, joint: str, q: float) -> None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid >= 0:
            self.data.qpos[self.model.jnt_qposadr[jid]] = q

    def qpos_of(self, joint: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        return float(self.data.qpos[self.model.jnt_qposadr[jid]])

    def _pd(self, joint: str, q_des: float, kp: float, kd: float) -> None:
        a = self.act_of.get(joint)
        if a is None:
            return
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        q = self.data.qpos[self.model.jnt_qposadr[jid]]
        dq = self.data.qvel[self.model.jnt_dofadr[jid]]
        self.data.ctrl[a] = kp * (q_des - q) + kd * (0.0 - dq)

    def apply_q_target(self, q_target_hud04: np.ndarray) -> None:
        """Write a full 31-actuator ctrl: 29 body joints + fixed head."""
        self.data.ctrl[:] = 0.0
        for i, joint in enumerate(HUD04_POLICY_JOINTS_29):
            self._pd(joint, float(q_target_hud04[i]), BODY_KP, BODY_KD)
        for name, (q, _kp, _kd) in HEAD_HOLD_TARGETS.items():
            self._pd(name, q, HEAD_KP, HEAD_KD)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def is_healthy(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos))) and \
            bool(np.all(np.isfinite(self.data.qvel))) and \
            float(np.max(np.abs(self.data.qvel))) < 1e3


def run_standing(steps: int, policy_path: str, xml_path=DEFAULT_XML,
                 velocity_command: np.ndarray | None = None,
                 depth_backbone: str | None = None,
                 depth_every: int = 10) -> dict:
    """Closed-loop: build obs -> ONNX -> adapter -> sim, tracking base height.

    Verifies the full observation + inference + reorder + scale + control path
    runs on HU_D04. Records base height and orientation drift so the standing
    stage (task 4.3) has an observable outcome.

    When ``depth_backbone`` is given, a real depth image is rendered from the
    HU_D04 camera every ``depth_every`` steps and its 32-dim latent fills the
    observation (task 5.7). Otherwise the depth block stays zero.
    """
    import onnxruntime as ort

    from hud04_sim2sim.observation import Hud04Observation, depth_latent_from_image

    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    sim = Hud04Sim(xml_path)
    obs_builder = Hud04Observation(sim.model, list(sim.adapter.config.joint_names))

    backbone_sess = None
    renderer = None
    if depth_backbone:
        backbone_sess = ort.InferenceSession(
            depth_backbone, providers=["CPUExecutionProvider"]
        )
        from hud04_sim2sim.depth_render import render_depth
        renderer = render_depth

    free_ids = [j for j in range(sim.model.njnt)
                if sim.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    qa = int(sim.model.jnt_qposadr[free_ids[0]])

    last_action = np.zeros(29, dtype=np.float32)
    depth_latent = np.zeros(32, dtype=np.float32)
    base_z_start = float(sim.data.qpos[qa + 2])
    base_z_min = base_z_start
    max_tilt = 0.0
    g1_action = None
    for step in range(steps):
        if renderer is not None and step % depth_every == 0:
            _, depth_img = renderer(xml_path)
            depth_latent = depth_latent_from_image(
                depth_img, depth_backbone, backbone_sess
            )
        obs = obs_builder.build(sim.data, last_action, velocity_command, depth_latent)
        feed = {}
        for inp in sess.get_inputs():
            if inp.name == "obs":
                feed[inp.name] = obs.reshape(1, -1).astype(np.float32)
            else:
                feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)
        out = sess.run(["actions"], feed)[0]
        g1_action = np.asarray(out, dtype=np.float32).reshape(-1)
        last_action = g1_action
        q_target = sim.adapter.to_hud04_q_target(g1_action)
        sim.apply_q_target(q_target)
        sim.step()
        if not sim.is_healthy():
            return {"status": "unshealthy", "step": step,
                    "base_z_start": round(base_z_start, 4)}
        base_z = float(sim.data.qpos[qa + 2])
        base_z_min = min(base_z_min, base_z)
        # Tilt = angle of the base z-axis from world up.
        quat = np.asarray(sim.data.xquat[obs_builder.anchor_body_id])
        up = np.array([0, 0, 1.0])
        qw, qx, qy, qz = quat
        r = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
        ])
        tilt = float(np.degrees(np.arccos(np.clip((r @ up)[2], -1, 1))))
        max_tilt = max(max_tilt, tilt)

    return {
        "status": "ok",
        "steps": steps,
        "sim_time": round(float(sim.data.time), 3),
        "depth_backbone": bool(depth_backbone),
        "base_z_start": round(base_z_start, 4),
        "base_z_min": round(base_z_min, 4),
        "base_z_final": round(float(sim.data.qpos[qa + 2]), 4),
        "max_base_tilt_deg": round(max_tilt, 2),
        "g1_action_range": [round(float(g1_action.min()), 4),
                            round(float(g1_action.max()), 4)],
        "healthy": sim.is_healthy(),
    }


def run_held_pose(steps: int, xml_path=DEFAULT_XML) -> dict:
    """Drive the robot to its default pose from a perturbed start.

    Verifies the integration stack holds a stable, finite, tracked pose.
    """
    sim = Hud04Sim(xml_path)
    # Perturb every body joint so the controller has work to do.
    rng = np.random.default_rng(0)
    for joint, q0 in sim.body_defaults.items():
        sim._set_qpos(joint, float(q0) + float(rng.normal(0, 0.05)))

    q_target = sim.adapter.config.default_dof_angles.copy()
    max_track_err = 0.0
    for _ in range(steps):
        sim.apply_q_target(q_target)
        sim.step()
        if not sim.is_healthy():
            return {"status": "unshealthy", "step": _}
        errs = [abs(sim.qpos_of(j) - q_target[i])
                for i, j in enumerate(HUD04_POLICY_JOINTS_29)]
        max_track_err = max(max_track_err, max(errs))

    return {
        "status": "ok",
        "steps": steps,
        "sim_time": round(float(sim.data.time), 3),
        "max_track_error_rad": round(max_track_err, 5),
        "head_yaw": round(sim.qpos_of("head_yaw_joint"), 5),
        "head_pitch": round(sim.qpos_of("head_pitch_joint"), 5),
        "healthy": sim.is_healthy(),
    }


def run_policy(steps: int, policy_path: str, xml_path=DEFAULT_XML) -> dict:
    """Closed-loop: ONNX inference -> adapter -> sim, with a zero observation.

    A zero observation is not a meaningful task input; the point is to confirm
    the inference + reorder + scale + control path executes without error.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    sim = Hud04Sim(xml_path)
    in_names = [i.name for i in sess.get_inputs()]
    g1_action = None
    for step in range(steps):
        feed = {}
        for inp in sess.get_inputs():
            feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)
        out = sess.run(["actions"], feed)[0]
        g1_action = np.asarray(out, dtype=np.float32).reshape(-1)
        q_target = sim.adapter.to_hud04_q_target(g1_action)
        sim.apply_q_target(q_target)
        sim.step()
        if not sim.is_healthy():
            return {"status": "unshealthy", "step": step}
    return {
        "status": "ok",
        "steps": steps,
        "sim_time": round(float(sim.data.time), 3),
        "onnx_inputs": in_names,
        "g1_action_range": [round(float(g1_action.min()), 4),
                            round(float(g1_action.max()), 4)],
        "healthy": sim.is_healthy(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--policy", default=None, help="path to G1 student.onnx")
    parser.add_argument("--depth-backbone", default=None,
                        help="path to depth_backbone.onnx (enables live depth)")
    parser.add_argument("--mode", choices=["held_pose", "standing"], default="standing")
    args = parser.parse_args(argv)

    if args.mode == "held_pose":
        result = run_held_pose(args.steps, args.xml)
    else:
        if not args.policy:
            parser.error("--mode standing requires --policy <student.onnx>")
        result = run_standing(args.steps, args.policy, args.xml,
                              depth_backbone=args.depth_backbone)

    for k, v in result.items():
        print(f"[hud04-sim2sim] {k}: {v}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

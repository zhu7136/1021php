"""可视化 sim2sim：带 MuJoCo 被动查看器，实时看机器人被 G1 策略驱动。

用 ``viewer.launch_passive`` 开窗口，可暂停/步进/调视角。
站立模式：机器人从站姿起步，你会看到它逐渐失稳跌倒。
箱体模式：前方有箱体，机器人试图前进。

注意：窗口需要 DISPLAY 已设置（本地桌面）。
"""

from __future__ import annotations

import argparse
import os

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


def _scene_xml(with_box: bool) -> str:
    base = open(DEFAULT_XML).read()
    extra = '<geom name="ground" type="plane" size="6 6 0.01" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>'
    if with_box:
        extra += ('<body name="box" pos="1.2 0 0.15">'
                  '<geom type="box" size="0.3 0.3 0.15" rgba="0.8 0.4 0.2 1"/></body>')
    return base.replace("</mujoco>", f"<worldbody>{extra}</worldbody>\n</mujoco>")


class VisSim:
    def __init__(self, with_box: bool):
        import tempfile
        patched = _scene_xml(with_box)
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                         dir="hud04_sim2sim/assets") as fh:
            fh.write(patched)
            self._tmp = fh.name
        self.model = load_model(self._tmp)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)
        self.adapter = ActionAdapter()
        self.act_of = {}
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            jn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jn:
                self.act_of[jn] = i
        self.defaults = dict(zip(self.adapter.config.joint_names,
                                 self.adapter.config.default_dof_angles))
        mujoco.mj_resetData(self.model, self.data)
        for n, q in self.defaults.items():
            self._set_qpos(n, float(q))
        # base pose
        free = [j for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        qa = int(self.model.jnt_qposadr[free])
        self.data.qpos[qa + 2] = 1.0
        self._qa = qa

    def _set_qpos(self, joint, q):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid >= 0:
            self.data.qpos[self.model.jnt_qposadr[jid]] = q

    def _pd(self, joint, q_des, kp, kd):
        a = self.act_of.get(joint)
        if a is None:
            return
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        q = self.data.qpos[self.model.jnt_qposadr[jid]]
        dq = self.data.qvel[self.model.jnt_dofadr[jid]]
        self.data.ctrl[a] = kp * (q_des - q) + kd * (0.0 - dq)

    def apply(self, q_target):
        self.data.ctrl[:] = 0.0
        for i, j in enumerate(HUD04_POLICY_JOINTS_29):
            self._pd(j, float(q_target[i]), BODY_KP, BODY_KD)
        for n, (q, _kp, _kd) in HEAD_HOLD_TARGETS.items():
            self._pd(n, q, HEAD_KP, HEAD_KD)

    def cleanup(self):
        import os
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--depth-backbone", default=None)
    parser.add_argument("--box", action="store_true", help="place a box ahead")
    parser.add_argument("--mode", choices=["g1_policy", "hold_default"], default="g1_policy")
    args = parser.parse_args(argv)

    sim = VisSim(with_box=args.box)

    import onnxruntime as ort
    from hud04_sim2sim.observation import Hud04Observation, depth_latent_from_image

    sess = None
    backbone_sess = None
    renderer = None
    if args.mode == "g1_policy":
        sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
        if args.depth_backbone:
            backbone_sess = ort.InferenceSession(args.depth_backbone, providers=["CPUExecutionProvider"])
            from hud04_sim2sim.depth_render import render_depth
            renderer = render_depth

    obs_builder = Hud04Observation(sim.model, list(sim.adapter.config.joint_names))
    from hud04_sim2sim.observation import CMD_DIM
    cmd = np.zeros(CMD_DIM, dtype=np.float32)
    cmd[1 if args.box else 0] = 1.0  # forward for box, stand otherwise

    import mujoco.viewer as viewer

    last_action = np.zeros(29, dtype=np.float32)
    latent = np.zeros(32, dtype=np.float32)
    print("[vis] launching passive viewer... (close window to exit)")
    try:
        with viewer.launch_passive(sim.model, sim.data) as v:
            step = 0
            while v.is_running():
                if args.mode == "hold_default":
                    sim.apply(sim.adapter.config.default_dof_angles)
                else:
                    if renderer is not None and step % 10 == 0:
                        _, img = renderer()
                        latent = depth_latent_from_image(img, args.depth_backbone, backbone_sess)
                    obs = obs_builder.build(sim.data, last_action, cmd, latent)
                    feed = {i.name: (obs.reshape(1, -1).astype(np.float32)
                                     if i.name == "obs" else np.zeros(i.shape, dtype=np.float32))
                            for i in sess.get_inputs()}
                    a = np.asarray(sess.run(["actions"], feed)[0], dtype=np.float32).reshape(-1)
                    last_action = a
                    sim.apply(sim.adapter.to_hud04_q_target(a))
                sim.model.opt.timestep = DT
                mujoco.mj_step(sim.model, sim.data)
                step += 1
                if step % 50 == 0:
                    print(f"[vis] step={step} base_z={sim.data.qpos[sim._qa+2]:.3f} "
                          f"x={sim.data.qpos[sim._qa]:.3f}")
                v.sync()
    finally:
        sim.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

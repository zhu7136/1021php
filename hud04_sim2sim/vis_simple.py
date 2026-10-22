#!/usr/bin/env python
"""最简单可靠的 HuD04 可视化：你在自己终端跑，窗口弹到桌面。

用法:
    /home/xf/instinct0715/.venv/bin/python -m hud04_sim2sim.vis_simple --mode hold_default
    /home/xf/instinct0715/.venv/bin/python -m hud04_sim2sim.vis_simple --mode g1_policy \
        --policy ~/.cache/php-parkour/student/student.onnx \
        --depth-backbone ~/.cache/php-parkour/student/depth_backbone.onnx
    /home/xf/instinct0715/.venv/bin/python -m hud04_sim2sim.vis_simple --mode g1_policy --box \
        --policy ~/.cache/php-parkour/student/student.onnx \
        --depth-backbone ~/.cache/php-parkour/student/depth_backbone.onnx
"""
import argparse, os, sys

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
from hud04_sim2sim.action_adapter import ActionAdapter
from hud04_sim2sim.head_lock import HEAD_HOLD_TARGETS
from hud04_sim2sim.joint_order import HUD04_POLICY_JOINTS_29
from hud04_sim2sim.model_loader import DEFAULT_XML, load_model

DT, BODY_KP, BODY_KD, HEAD_KP, HEAD_KD = 0.002, 100.0, 5.0, 200.0, 2.0


def scene_xml(with_box):
    base = open(DEFAULT_XML).read()
    extra = '<geom name="ground" type="plane" size="6 6 0.01" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>'
    if with_box:
        extra += '<body name="box" pos="1.2 0 0.15"><geom type="box" size="0.3 0.3 0.15" rgba="0.8 0.4 0.2 1"/></body>'
    return base.replace("</mujoco>", f"<worldbody>{extra}</worldbody>\n</mujoco>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hold_default", "g1_policy"], default="hold_default")
    ap.add_argument("--box", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--depth-backbone", default=None)
    args = ap.parse_args()

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir="hud04_sim2sim/assets") as fh:
        fh.write(scene_xml(args.box)); tmp = fh.name

    model = load_model(tmp); model.opt.timestep = DT
    data = mujoco.MjData(model)
    adapter = ActionAdapter()
    act_of = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jn: act_of[jn] = i
    defaults = dict(zip(adapter.config.joint_names, adapter.config.default_dof_angles))

    def set_q(j, q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0: data.qpos[model.jnt_qposadr[jid]] = q

    def pd(j, qd, kp, kd):
        a = act_of.get(j)
        if a is None: return
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        data.ctrl[a] = kp*(qd-data.qpos[model.jnt_qposadr[jid]]) + kd*(0-data.qvel[model.jnt_dofadr[jid]])

    def apply(q):
        data.ctrl[:] = 0.0
        for i, j in enumerate(HUD04_POLICY_JOINTS_29): pd(j, float(q[i]), BODY_KP, BODY_KD)
        for n, (q_, _k, _d) in HEAD_HOLD_TARGETS.items(): pd(n, q_, HEAD_KP, HEAD_KD)

    mujoco.mj_resetData(model, data)
    for n, q in defaults.items(): set_q(n, float(q))
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    qa = int(model.jnt_qposadr[free]); data.qpos[qa+2] = 1.0

    sess = backbone_sess = renderer = obs_builder = None
    last_action = np.zeros(29, dtype=np.float32); latent = np.zeros(32, dtype=np.float32)
    if args.mode == "g1_policy":
        import onnxruntime as ort
        from hud04_sim2sim.observation import Hud04Observation, CMD_DIM, depth_latent_from_image
        assert args.policy, "g1_policy needs --policy"
        sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
        obs_builder = Hud04Observation(model, list(adapter.config.joint_names))
        cmd = np.zeros(CMD_DIM, dtype=np.float32); cmd[1 if args.box else 0] = 1.0
        if args.depth_backbone:
            backbone_sess = ort.InferenceSession(args.depth_backbone, providers=["CPUExecutionProvider"])
            from hud04_sim2sim.depth_render import render_depth
            renderer = render_depth

    print("[vis] opening MuJoCo window on", os.environ.get("DISPLAY"), flush=True)
    import mujoco.viewer as viewer
    step = 0
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            if args.mode == "hold_default":
                apply(adapter.config.default_dof_angles)
            else:
                if renderer is not None and step % 10 == 0:
                    _, img = renderer(); latent = depth_latent_from_image(img, args.depth_backbone, backbone_sess)
                obs = obs_builder.build(data, last_action, cmd, latent)
                feed = {i.name: (obs.reshape(1,-1).astype(np.float32) if i.name=="obs"
                                 else np.zeros(i.shape, dtype=np.float32)) for i in sess.get_inputs()}
                last_action = np.asarray(sess.run(["actions"], feed)[0], dtype=np.float32).reshape(-1)
                apply(adapter.to_hud04_q_target(last_action))
            mujoco.mj_step(model, data); step += 1
            if step % 100 == 0:
                print(f"[vis] step={step} base_z={data.qpos[qa+2]:.3f} x={data.qpos[qa]:.3f}", flush=True)
            v.sync()
    os.unlink(tmp)
    os._exit(0)  # avoid teardown segfault noise


if __name__ == "__main__":
    main()

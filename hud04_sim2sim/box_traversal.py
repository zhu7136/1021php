"""Task 4.4: HU_D04 box-traversal recording.

Places a box in front of HU_D04, drives it forward with the G1 student (with
live depth), and records whether the robot makes forward progress and whether
its base rises over the box. The point is to RECORD the outcome, not to make it
succeed: the standing stage already showed a morphology gap that blocks
zero-shot transfer, so this quantifies the same failure on a box.

Scene: a 0.6 x 0.6 x 0.3 m box at x=1.2 m, robot spawned standing behind it.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import tempfile  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hud04_sim2sim.model_loader import DEFAULT_XML  # noqa: E402
from hud04_sim2sim.observation import (  # noqa: E402
    CMD_DIM,
    Hud04Observation,
    depth_latent_from_image,
)
from hud04_sim2sim.runner import BODY_KD, BODY_KP, HEAD_KD, HEAD_KP, Hud04Sim  # noqa: E402

BOX_X = 1.2
BOX_SIZE = (0.3, 0.3, 0.15)   # half-sizes
BOX_TOP = 2 * BOX_SIZE[2]     # 0.30 m
STEPS_DEFAULT = 3000


def _scene_xml() -> str:
    base = open(DEFAULT_XML).read()
    box = f'''
  <worldbody>
    <geom name="ground" type="plane" size="6 6 0.01" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>
    <body name="box" pos="{BOX_X} 0 {BOX_SIZE[2]}">
      <geom type="box" size="{BOX_SIZE[0]} {BOX_SIZE[1]} {BOX_SIZE[2]}" rgba="0.8 0.4 0.2 1"/>
    </body>
  </worldbody>
'''
    return base.replace("</mujoco>", box + "\n</mujoco>")


def run_box_traversal(steps: int, policy_path: str, backbone_path: str | None,
                      depth_every: int = 10) -> dict:
    import onnxruntime as ort

    patched = _scene_xml()
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                     dir="hud04_sim2sim/assets") as fh:
        fh.write(patched)
        tmp = fh.name
    try:
        sim = Hud04Sim(tmp)
    finally:
        os.unlink(tmp)

    # Spawn standing behind the box.
    free = [j for j in range(sim.model.njnt)
            if sim.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    qa = int(sim.model.jnt_qposadr[free])
    sim.data.qpos[qa] = 0.0
    sim.data.qpos[qa + 1] = 0.0
    sim.data.qpos[qa + 2] = 1.0

    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    obs_builder = Hud04Observation(sim.model, list(sim.adapter.config.joint_names))

    backbone_sess = None
    renderer = None
    if backbone_path:
        backbone_sess = ort.InferenceSession(backbone_path, providers=["CPUExecutionProvider"])
        from hud04_sim2sim.depth_render import render_depth
        renderer = render_depth

    cmd = np.zeros(CMD_DIM, dtype=np.float32)
    cmd[1] = 1.0  # forward command
    last_action = np.zeros(29, dtype=np.float32)
    latent = np.zeros(32, dtype=np.float32)

    x_start = float(sim.data.qpos[qa])
    x_max = x_start
    z_max = float(sim.data.qpos[qa + 2])
    fell_at = None
    for step in range(steps):
        if renderer is not None and step % depth_every == 0:
            _, img = renderer()
            latent = depth_latent_from_image(img, backbone_path, backbone_sess)
        obs = obs_builder.build(sim.data, last_action, cmd, latent)
        feed = {i.name: (obs.reshape(1, -1).astype(np.float32)
                         if i.name == "obs" else np.zeros(i.shape, dtype=np.float32))
                for i in sess.get_inputs()}
        a = np.asarray(sess.run(["actions"], feed)[0], dtype=np.float32).reshape(-1)
        last_action = a
        sim.apply_q_target(sim.adapter.to_hud04_q_target(a))
        sim.step()
        if not sim.is_healthy():
            fell_at = step
            break
        x_max = max(x_max, float(sim.data.qpos[qa]))
        z_max = max(z_max, float(sim.data.qpos[qa + 2]))
        if float(sim.data.qpos[qa + 2]) < 0.4:  # collapsed
            fell_at = step
            break

    x_final = float(sim.data.qpos[qa])
    return {
        "status": "recorded",
        "steps_run": step + 1,
        "box_top_z": round(BOX_TOP, 3),
        "x_start": round(x_start, 3),
        "x_final": round(x_final, 3),
        "x_max": round(x_max, 3),
        "forward_progress_m": round(x_final - x_start, 3),
        "base_z_max": round(z_max, 3),
        "climbed_over_box": bool(x_final > BOX_X and float(sim.data.qpos[qa + 2]) > BOX_TOP),
        "collapsed_at_step": fell_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--depth-backbone", default=None)
    args = parser.parse_args(argv)

    result = run_box_traversal(args.steps, args.policy, args.depth_backbone)
    for k, v in result.items():
        print(f"[hud04-box] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

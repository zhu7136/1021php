"""HU_D04 student observation builder (task 4.3).

The G1 depth student's observation is proprio + command + depth latent:

    obs(140) = robot_anchor_projected_gravity(3)
             + base_ang_vel(3)
             + dof_pos(29) + dof_vel(29) + actions(29)
             + velocity_command(15)
             + depth_latent(32)

Every term except the depth latent is derivable from HU_D04 simulation state,
so a valid HU_D04 observation can be assembled. `dof_pos` / `dof_vel` use the
HU_D04 policy joint order (identical names to G1; see joint_order.py), so no
body-frame motion reference is required.

The depth latent is produced by the unchanged depth backbone from the camera
image; here it defaults to zeros so the control loop can run without a depth
producer (matching the inference stack's `--task.depth-shm.no-required` mode).
"""

from __future__ import annotations

import mujoco
import numpy as np

GRAVITY_DIM = 3
ANG_VEL_DIM = 3
DOF_DIM = 29
CMD_DIM = 15
DEPTH_LATENT_DIM = 32
OBS_DIM = GRAVITY_DIM + ANG_VEL_DIM + 3 * DOF_DIM + CMD_DIM + DEPTH_LATENT_DIM  # 140

# World gravity direction, unit vector (MuJoCo convention: -z).
_WORLD_G = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity in the anchor base frame (projected gravity), 3 dims."""
    qw, qx, qy, qz = quat_wxyz
    # Rotate world gravity into body frame: R^T @ g.
    # Quaternion (w,x,y,z) -> rotation matrix applied as R^T @ g.
    r = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
        [2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)],
        [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])
    return (r.T @ _WORLD_G).astype(np.float32)


class Hud04Observation:
    """Assembles the 140-dim student observation from HU_D04 sim state."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str],
                 anchor_body: str = "base_link"):
        self.model = model
        self.joint_names = list(joint_names)
        self.anchor_body = anchor_body
        self.anchor_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, anchor_body
        )
        if self.anchor_body_id < 0:
            raise ValueError(f"anchor body {anchor_body!r} not in model")
        # Precompute qpos/dof addresses per policy joint.
        self._qadr = []
        self._vadr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"policy joint {name!r} not in model")
            self._qadr.append(int(model.jnt_qposadr[jid]))
            self._vadr.append(int(model.jnt_dofadr[jid]))

    def build(self, data: mujoco.MjData, last_action: np.ndarray,
              velocity_command: np.ndarray | None = None,
              depth_latent: np.ndarray | None = None) -> np.ndarray:
        quat = np.asarray(data.xquat[self.anchor_body_id], dtype=np.float32)
        grav = projected_gravity(quat)
        # Anchor angular velocity in world frame; use body-frame z-ish proxy.
        ang_vel = np.asarray(data.cvel[self.anchor_body_id, :3], dtype=np.float32)
        dof_pos = np.array([data.qpos[a] for a in self._qadr], dtype=np.float32)
        dof_vel = np.array([data.qvel[a] for a in self._vadr], dtype=np.float32)
        act = np.asarray(last_action, dtype=np.float32).reshape(-1)
        cmd = (np.zeros(CMD_DIM, dtype=np.float32) if velocity_command is None
               else np.asarray(velocity_command, dtype=np.float32).reshape(-1))
        latent = (np.zeros(DEPTH_LATENT_DIM, dtype=np.float32) if depth_latent is None
                  else np.asarray(depth_latent, dtype=np.float32).reshape(-1))

        obs = np.concatenate([grav, ang_vel, dof_pos, dof_vel, act, cmd, latent])
        if obs.shape[0] != OBS_DIM:
            raise ValueError(f"obs dim {obs.shape[0]} != {OBS_DIM}")
        return obs


def main() -> int:
    from hud04_sim2sim.action_adapter import ActionAdapter
    from hud04_sim2sim.model_loader import DEFAULT_XML, load_model

    model = load_model(DEFAULT_XML)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    adapter = ActionAdapter()
    obs_builder = Hud04Observation(model, list(adapter.config.joint_names))
    obs = obs_builder.build(data, np.zeros(29, dtype=np.float32))
    print(f"[hud04] obs dim: {obs.shape[0]} (expected {OBS_DIM})")
    print(f"[hud04] gravity block: {obs[:3]}")
    print(f"[hud04] dof_pos block[0:4]: {obs[6:10]}")
    return 0


def depth_latent_from_image(depth_image: np.ndarray,
                            backbone_path: str,
                            session=None) -> np.ndarray:
    """Run the unchanged depth backbone on a 58x87 depth image -> (32,) latent.

    ``depth_image`` may be (58, 87) or (1, 58, 87).
    """
    import onnxruntime as ort

    img = np.asarray(depth_image, dtype=np.float32)
    if img.ndim == 2:
        img = img.reshape(1, 58, 87)
    if img.shape != (1, 58, 87):
        raise ValueError(f"depth image shape {img.shape} != (1,58,87)")

    if session is None:
        session = ort.InferenceSession(backbone_path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out = session.run(None, {in_name: img})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


if __name__ == "__main__":
    raise SystemExit(main())

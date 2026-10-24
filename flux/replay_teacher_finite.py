"""Read-only, finite policy replay; no training, optimizer steps, or checkpoint writes."""
from __future__ import annotations

import dataclasses
import atexit
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "flux_replay_latest_1000"
EXPECTED_SHA256 = "5c8573b1a954f3e58a4242f3a1858aa5de06c20c860465dddf7d373250ff2489"
STEPS = 1000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(dataset: Path):
    # Inputs are mounted by Flux from the explicitly selected source checkpoint.
    candidates = list((ROOT / "flux" / "replay-input").glob("*.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one mounted checkpoint, found {len(candidates)}")
    checkpoint = candidates[0]
    if digest(checkpoint) != EXPECTED_SHA256:
        raise RuntimeError("Mounted checkpoint SHA256 does not match the selected latest checkpoint")
    OUT.mkdir(parents=True, exist_ok=True)

    from holosoma.utils.eval_utils import CheckpointConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    from holosoma.config_types.video import VideoConfig, SphericalCameraConfig
    import holosoma.utils.sim_utils as sim_utils
    from holosoma.utils.sim_utils import close_simulation_app
    from holosoma.utils.helpers import get_class
    from wbt_training.utils.checkpoint import load_saved_experiment_config, apply_preprocess_hook
    from startup_diagnostics import bounded_launcher

    sim_utils.setup_isaaclab_launcher = bounded_launcher(sim_utils.setup_isaaclab_launcher)

    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    # Keep training environment, rewards, termination, physics and randomization settings.
    # Change only evaluation plumbing, environment count, assets and output destination.
    training = dataclasses.replace(
        saved.training, num_envs=1, headless=True, max_eval_steps=STEPS,
        export_onnx=False, registry_name=dataset.resolve().as_uri(),
    )
    kwargs = saved.training.preprocess_hook_kwargs
    kwargs = json.loads(kwargs) if isinstance(kwargs, str) else dict(kwargs or {})
    assert kwargs.get("num_variants") == 32, "Preserve original 32 terrain variants"
    cfg = dataclasses.replace(
        saved, training=training,
        algo=dataclasses.replace(saved.algo, config=dataclasses.replace(saved.algo.config, load_optimizer=False)),
        logger=DisabledLoggerConfig(
            base_dir=str(OUT), headless_recording=True,
            video=VideoConfig(enabled=True, interval=1, width=640, height=360,
                              save_dir=str(OUT / "videos"), upload_to_wandb=False,
                              camera=SphericalCameraConfig(distance=3.0, azimuth=135.0, elevation=25.0)),
        ),
    )
    cfg = apply_preprocess_hook(cfg)
    cfg.save_config(str(OUT / "replay_config.yaml"))
    # The bootstrap has already consumed its own CLI args; Isaac's parser must not see them.
    sys.argv = [sys.argv[0]]
    env, device, app = sim_utils.setup_simulation_environment(cfg)
    atexit.register(close_simulation_app, app)
    import numpy as np
    import torch

    torch.set_grad_enabled(False)
    if device != "cuda:0" or env.num_envs != 1:
        raise RuntimeError("Replay requires exactly one GPU environment; refusing fallback")
    sim = env.simulator
    native_names = list(sim._robot.joint_names)
    ids = list(sim.dof_ids)
    assert len(set(ids)) == len(ids) == 29
    assert [native_names[i] for i in ids] == list(cfg.robot.dof_names)
    assert torch.equal(sim.dof_pos, sim._robot.data.joint_pos[:, ids])
    mapping = [{"policy_index": i, "joint": name, "isaac_index": ids[i]}
               for i, name in enumerate(cfg.robot.dof_names)]
    (OUT / "joint_mapping.json").write_text(json.dumps(mapping, indent=2))
    print("REPLAY_JOINT_MAPPING " + json.dumps(mapping), flush=True)
    # PPO.__init__ discards a reset_all result. Skip only that redundant warm-up;
    # the instrumented reset below is the sole initialization control step.
    original_reset_all = env.reset_all
    env.reset_all = lambda: None
    try:
        algo = get_class(cfg.algo._target_)(env=env, config=cfg.algo.config,
            log_dir=str(OUT / "tensorboard"), device=device)
    finally:
        env.reset_all = original_reset_all
    def forbidden(*args, **kwargs):
        raise RuntimeError("Training and checkpoint saving are disabled for this evaluation")
    algo.learn = forbidden
    algo.save = forbidden
    algo.setup()
    # The 32-env adaptive sampler state is not portable to a one-env regenerated registry.
    # Load policy/value/normalization weights only; never restore training sampler state.
    algo._restore_env_state = lambda state: None
    algo.load(str(checkpoint))
    policy = algo.get_inference_policy(device=device)
    def policy_hash():
        h = hashlib.sha256()
        for name, tensor in sorted(algo.actor.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    weights_before = policy_hash()
    names = list(env.simulator.body_names)
    contact_term = cfg.reward.terms["undesired_contacts"]
    pattern = contact_term.params["undesired_contacts_body_names"]
    threshold = float(contact_term.params["threshold"])
    undesired = [i for i, name in enumerate(names) if re.match(pattern, name)]
    frame_arrays = {k: [] for k in (
        "root_pos", "root_quat_xyzw", "body_pos", "joint_pos", "joint_vel",
        "contact_force_norm", "root_tilt_rad", "root_clearance_proxy_m", "reset", "timeout", "actions",
    )}
    term_counts = {}
    term_flags = {}
    # Capture each original termination function's result without evaluating it twice.
    class RecordedTerm:
        def __init__(self, name, original):
            self.name, self.original = name, original
        def __call__(self, *args, **kwargs):
            result = self.original(*args, **kwargs)
            term_flags[self.name] = result.clone()
            return result
        def reset(self, *args, **kwargs):
            return self.original.reset(*args, **kwargs)
    manager = env.termination_manager
    for name, original in list(manager._term_instances.items()):
        manager._term_instances[name] = RecordedTerm(name, original)
    for name, original in list(manager._term_funcs.items()):
        manager._term_funcs[name] = RecordedTerm(name, original)
    original_check = env._check_termination
    event_log = (OUT / "events.jsonl").open("w", buffering=1)
    current_step = [-1]

    def capture_before_reset():
        original_check()
        if current_step[0] < 0:
            return
        sim = env.simulator
        root = sim.robot_root_states[0].detach()
        body_pos = sim._rigid_body_pos[0].detach()
        quat = root[3:7]
        tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[0] ** 2 + quat[1] ** 2), -1.0, 1.0))
        forces = torch.linalg.vector_norm(sim.contact_forces_history[0], dim=-1).amax(dim=0)
        values = {
            "root_pos": root[:3], "root_quat_xyzw": quat, "body_pos": body_pos,
            "joint_pos": sim.dof_pos[0], "joint_vel": sim.dof_vel[0],
            "contact_force_norm": forces, "root_tilt_rad": tilt,
            "root_clearance_proxy_m": root[2] - body_pos[:, 2].min(),
            "reset": env.reset_buf[0], "timeout": env.time_out_buf[0],
        }
        for key, value in values.items():
            frame_arrays[key].append(value.detach().cpu().numpy().copy())
        active_terms = []
        for name, flag in term_flags.items():
            if bool(flag[0].item()):
                term_counts[name] = term_counts.get(name, 0) + 1
                active_terms.append(name)
        if bool(env.reset_buf[0].item()):
            event_log.write(json.dumps({"step": current_step[0], "event": "pre_reset",
                                       "termination_terms": active_terms, "tilt_deg": float(tilt.item() * 180 / np.pi),
                                       "contact_bodies_above_threshold": [names[i] for i in range(len(names)) if forces[i] > threshold]}) + "\n")

    env._check_termination = capture_before_reset
    env.set_is_evaluating()
    # reset_all performs one zero-action control step; include it in the 1,000-step budget.
    current_step[0] = 0
    frame_arrays["actions"].append(np.zeros(29, dtype=np.float32))
    obs = env.reset_all()
    print("REPLAY_BEGIN " + json.dumps({"steps": STEPS, "num_envs": 1, "checkpoint_sha256": EXPECTED_SHA256,
          "checkpoint_iteration": algo.current_learning_iteration, "train_updates": 0,
          "adaptive_sampler_restored": False, "body_names": names}), flush=True)
    try:
        for step in range(1, STEPS):
            current_step[0] = step
            actions = policy(obs)
            frame_arrays["actions"].append(actions[0].detach().cpu().numpy().copy())
            obs, _, _, _ = env.step({"actions": actions})
            if (step + 1) % 100 == 0:
                print("REPLAY_PROGRESS " + json.dumps({"steps": step + 1, "termination_counts": term_counts}), flush=True)
        if env.simulator.video_recorder:
            env.simulator.video_recorder.stop_recording()
        arrays = {name: np.asarray(values) for name, values in frame_arrays.items()}
        assert all(len(value) == STEPS for value in arrays.values())
        np.savez_compressed(OUT / "trajectory.npz", **arrays)
        forces = arrays["contact_force_norm"]
        undesired_mask = forces[:, undesired] > threshold
        summary = {
            "steps": STEPS, "num_envs": 1, "control_dt_s": float(env.dt),
            "simulated_time_s": float(STEPS * env.dt), "training_updates": 0,
            "policy_steps": STEPS - 1, "initialization_zero_action_steps": 1,
            "checkpoint_sha256": EXPECTED_SHA256, "checkpoint_unchanged": digest(checkpoint) == EXPECTED_SHA256,
            "actor_weights_unchanged": policy_hash() == weights_before,
            "checkpoint_iteration": algo.current_learning_iteration,
            "joint_mapping_runtime_verified": True,
            "terrain_variants": 32,
            "terrain_regenerated": True,
            "termination_counts": term_counts, "resets": int(arrays["reset"].sum()),
            "timeouts": int(arrays["timeout"].sum()),
            "undesired_contact_threshold_N": threshold, "undesired_body_names": [names[i] for i in undesired],
            "steps_with_undesired_contact": int(undesired_mask.any(axis=1).sum()),
            "undesired_contact_step_fraction": float(undesired_mask.any(axis=1).mean()),
            "contact_steps_by_body": {name: int((forces[:, i] > threshold).sum()) for i, name in enumerate(names)},
            "max_force_N_by_body": {name: float(forces[:, i].max()) for i, name in enumerate(names)},
            "root_tilt_max_deg": float(np.rad2deg(arrays["root_tilt_rad"]).max()),
            "steps_tilt_over_60_deg": int((arrays["root_tilt_rad"] > np.pi / 3).sum()),
            "action_abs_max": float(np.abs(arrays["actions"]).max()),
            "joint_speed_abs_max_rad_s": float(np.abs(arrays["joint_vel"]).max()),
            "joint_limit_violation_steps": int(((arrays["joint_pos"] < np.asarray(cfg.robot.dof_pos_lower_limit_list) - 0.01)
                | (arrays["joint_pos"] > np.asarray(cfg.robot.dof_pos_upper_limit_list) + 0.01)).any(axis=1).sum()),
            "all_states_and_actions_finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
            "body_names": names,
            "interpretation": "Contacts use the original reward regex/threshold and physics history; tilt is a fall proxy, not a validated fall label. Resets sampled before automatic reset. One-env replay is diagnostic, not success-rate estimation.",
            "videos": [str(p.relative_to(OUT)) for p in (OUT / "videos").rglob("*.mp4")],
        }
        if not summary["actor_weights_unchanged"] or not summary["checkpoint_unchanged"]:
            raise RuntimeError("Read-only replay invariant failed")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print("REPLAY_RESULT " + json.dumps(summary), flush=True)
    finally:
        event_log.close()
        atexit.unregister(close_simulation_app)
        close_simulation_app(app)


if __name__ == "__main__":
    raise SystemExit("Use the bounded replay bootstrap, not this module directly")
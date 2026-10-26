"""The r11 physical loop unchanged, but owned by a separate returning frame."""
import faulthandler
import os
from pathlib import Path
import weakref

def run_physics(event, DT, STEPS, HEAD_MAX_RAD):
    import torch
    event('physics_import_started')
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from hud04_robot import (SDK_JOINT_NAMES, HEAD_JOINT_NAMES, POLICY_JOINT_NAMES,
                             build_hud04_implicit_actuators, default_position, resolve_policy_indices)
    event('physics_import_returned')
    root = Path(__file__).resolve().parent
    usd = root / ('assets-' + os.environ['HUD04_OWNED_TASK']) / 'asset/usd/HU_D04_01.usd'
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device='cuda:0', render_interval=4))
    event('physics_context_created')
    sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
    cfg = ArticulationCfg(
        prim_path='/World/HuD04',
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
                solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .9046898903895483),
            joint_pos={n: default_position(n) for n in SDK_JOINT_NAMES}, joint_vel={'.*': 0.}),
        actuators=build_hud04_implicit_actuators(), soft_joint_pos_limit_factor=.9)
    robot = Articulation(cfg)
    event('robot_spawn_requested')
    sim.reset()
    robot.reset()
    robot.update(DT)
    names = tuple(robot.joint_names)
    indices = resolve_policy_indices(names)
    heads = tuple(names.index(n) for n in HEAD_JOINT_NAMES)
    if robot.num_instances != 1 or bool(robot.is_fixed_base):
        raise ValueError('Expected exactly one floating-base HuD04')
    event('robot_initialized', joint_names=names, physical_joints=len(names),
          policy_joints=len(indices), policy_names=POLICY_JOINT_NAMES,
          head_indices=heads, policy_indices=indices, floating_base=True, robot_loaded=True)
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    targets = robot.data.default_joint_pos.clone()
    targets[:, list(heads)] = 0.
    robot.write_joint_state_to_sim(targets, torch.zeros_like(targets))
    robot.reset()
    head_max = 0.
    root_z_min = float('inf')
    for step in range(STEPS):
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
        for tensor in [robot.data.joint_pos, robot.data.joint_vel, robot.data.root_state_w]:
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('Nonfinite native physics state')
        head_max = max(head_max, float(robot.data.joint_pos[:, list(heads)].abs().max()))
        root_z_min = min(root_z_min, float(robot.data.root_pos_w[:, 2].min()))
        if step in (0, 15, 31, 63):
            event('physics_progress', steps=step + 1, head_max_abs_rad=head_max,
                  root_z_min_m=root_z_min, states_finite=True)
    event('physics_steps_completed', steps=STEPS, simulated_seconds=STEPS * DT,
          states_finite=True, head_zero_target=True, head_max_abs_rad=head_max,
          head_threshold_rad=HEAD_MAX_RAD, head_short_probe_passed=head_max <= HEAD_MAX_RAD,
          root_z_min_m=root_z_min, root_z_final_m=float(robot.data.root_pos_w[0, 2]),
          ppo_updates=0, locomotion_or_reference_feasibility_claimed=False)
    if head_max > HEAD_MAX_RAD:
        raise ValueError('Short-probe head displacement exceeds preregistered 0.05rad threshold')
    event('physics_cleanup_started', phase_seconds=30)
    faulthandler.enable()
    faulthandler.dump_traceback_later(5, repeat=True)
    try:
        event('physics_stop_started')
        from stop_guard_candidate import guarded_stop
        guarded_stop(sim,event)
        event('physics_stop_returned')
        event('physics_callbacks_clear_started')
        sim.clear_all_callbacks()
        event('physics_callbacks_clear_returned')
        event('physics_instance_clear_started')
        sim.clear_instance()
        event('physics_instance_clear_returned')
        event('physics_cleanup_returned')
    finally:
        faulthandler.cancel_dump_traceback_later()
    # Weak references are measurements, not handles that keep PhysX owners alive.
    return {'robot': weakref.ref(robot), 'sim': weakref.ref(sim)}
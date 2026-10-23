"""HU_D04 WBT experiment configurations.

Builds on the G1 terrain-teacher recipe (``g1_wbt_terrain_ref_noscale``) but
swaps in the HU_D04 robot asset, HU_D04 contact whitelist reward, HU_D04 body
names for termination/tracking, and unit action scaling (no effort/Kp scale
like G1 — HU_D04's leg Kp is already 2-5.5x G1's).

Imports ``wbt_training.config_values.terrain`` directly (not the top-level
experiment module) to avoid a circular import with
``wbt_training.config_values.experiment``.
"""

from __future__ import annotations

import json
from dataclasses import replace

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.algo import OptimizerConfig
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_values import robot as holosoma_robot
from holosoma.config_values import simulator
from holosoma.config_values.algo import ppo as algo_ppo
from holosoma.config_values.logger import wandb as wandb_logger
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum

from wbt_training.config_values.terrain import (
    g1_29dof_wbt_terrain_observation,
    g1_29dof_wbt_terrain_randomization,
    hu_d04_terrain_command,
    hu_d04_terrain_command_root,
    hu_d04_terrain_reward,
    hu_d04_terrain_termination,
)

# Implicit position-target action term (same as G1 terrain teacher).
_hu_d04_joint_pos_implicit = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionTargetActionTerm",
            params={},
            scale=1.0,
            clip=None,
        )
    }
)

_ref_adamw_no_wd = OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.0)

# Terrain teacher with preprocess (registry motion + on-path obstacles).
hu_d04_wbt_terrain_ref_noscale = ExperimentConfig(
    training=TrainingConfig(
        project="HuD04_WBT",
        name="hu_d04_wbt_terrain_teacher",
        num_envs=200,
        preprocess_hook="wbt_training.config_values.terrain:apply_terrain_preprocess",
        preprocess_hook_kwargs=json.dumps(
            {
                "add_onpath_obstacle": True,
                "num_variants": 64,
            }
        ),
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo_ppo,
        config=replace(
            algo_ppo.config,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            num_learning_epochs=5,
            num_learning_iterations=30000,
            save_interval=500,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            empirical_normalization=False,
            actor_optimizer=_ref_adamw_no_wd,
            critic_optimizer=_ref_adamw_no_wd,
        ),
    ),
    logger=wandb_logger,
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
        ),
    ),
    scene=replace(g1_29dof_wbt.scene, env_spacing=0.0),
    robot=replace(
        holosoma_robot.hu_d04,
        control=replace(
            holosoma_robot.hu_d04.control,
            action_scale=1.0,
            action_scales_by_effort_limit_over_p_gain=False,
            control_mode="implicit_position_target",
            implicit_actuator_builder="wbt_training.config_values.robot_config:build_hu_d04_implicit_actuators",
        ),
        asset=replace(
            holosoma_robot.hu_d04.asset,
            enable_self_collisions=True,
        ),
        init_state=replace(
            holosoma_robot.hu_d04.init_state, pos=[0.0, 0.0, 0.91]
        ),
    ),
    action=_hu_d04_joint_pos_implicit,
    observation=g1_29dof_wbt_terrain_observation,
    termination=hu_d04_terrain_termination,
    randomization=g1_29dof_wbt_terrain_randomization,
    command=hu_d04_terrain_command,
    reward=hu_d04_terrain_reward,
    curriculum=g1_29dof_wbt_curriculum,
)

# Same config with root-relative motion reference (base_link anchor).
hu_d04_wbt_terrain_ref_noscale_root = replace(
    hu_d04_wbt_terrain_ref_noscale,
    training=replace(
        hu_d04_wbt_terrain_ref_noscale.training,
        name="hu_d04_wbt_terrain_teacher_root",
    ),
    command=hu_d04_terrain_command_root,
)

# Depth distillation finetune placeholder (same geometry/robot as teacher;
# algorithm/observation overrides come from depth_distillation when needed).
hu_d04_wbt_terrain_warp_distill_finetune = replace(
    hu_d04_wbt_terrain_ref_noscale,
    training=replace(
        hu_d04_wbt_terrain_ref_noscale.training,
        name="hu_d04_wbt_terrain_distill_finetune",
        preprocess_hook_kwargs=json.dumps(
            {"add_onpath_obstacle": True, "num_variants": 3}
        ),
    ),
)

__all__ = [
    "hu_d04_wbt_terrain_ref_noscale",
    "hu_d04_wbt_terrain_ref_noscale_root",
    "hu_d04_wbt_terrain_warp_distill_finetune",
]

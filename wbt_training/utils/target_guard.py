"""Opt-in HU_D04 target safety recipe; original presets remain unchanged."""
from dataclasses import replace
import math
import torch

ACTION_SCALE = 0.25
TARGET_MARGIN_RAD = 0.02

def protect_targets(requested, hard_limits, margin=TARGET_MARGIN_RAD):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid target margin')
    if requested.ndim != 2 or hard_limits.shape != (requested.shape[1], 2):
        raise ValueError('Joint target/limit shape mismatch')
    if not torch.isfinite(requested).all() or not torch.isfinite(hard_limits).all():
        raise FloatingPointError('Nonfinite target or joint limit')
    lower, upper = hard_limits[:, 0] + margin, hard_limits[:, 1] - margin
    if not (lower < upper).all():
        raise ValueError('Target margin consumes joint range')
    return torch.maximum(torch.minimum(requested, upper), lower)

def _term(env):
    matches = [t for t in env.action_manager._term_instances.values()
               if hasattr(t, '_requested_position_targets')]
    if len(matches) != 1:
        raise ValueError('Exactly one protected action term required')
    return matches[0]

def requested_target_violation(env):
    """Penalize PRE-protection requests so clipping cannot hide policy drift."""
    term = _term(env)
    gap = term._requested_position_targets - term._processed_actions
    value = gap.square().sum(dim=1)
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite requested target penalty')
    return value

def physical_action_rate(env):
    """Same rate penalty in physical target-delta units, before target clamp."""
    term = _term(env)
    delta = (env.action_manager.action - env.action_manager.prev_action) * term.action_scales
    return delta.square().sum(dim=1)

def failure_event(env):
    """Reward manager multiplies by dt; compensate to charge once per failure."""
    if not math.isfinite(float(env.dt)) or env.dt <= 0:
        raise ValueError('Positive finite control dt required')
    return env.termination_manager.terminated.float() / env.dt

def protected_config(cfg):
    from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
    from holosoma.config_types.reward import RewardTermCfg
    if set(cfg.termination.terms) != {'bad_tracking', 'timeout'}:
        raise ValueError('Failure penalty requires known termination recipe')
    rewards = dict(cfg.reward.terms)
    if any(k in rewards for k in ('requested_target_violation', 'failure_event')):
        raise ValueError('Protection must not be applied twice')
    rewards['action_rate_l2'] = replace(rewards['action_rate_l2'],
        func='wbt_training.utils.target_guard:physical_action_rate')
    rewards['requested_target_violation'] = RewardTermCfg(
        func='wbt_training.utils.target_guard:requested_target_violation', weight=-1.0)
    rewards['failure_event'] = RewardTermCfg(
        func='wbt_training.utils.target_guard:failure_event', weight=-1.0)
    return replace(cfg,
        robot=replace(cfg.robot, control=replace(cfg.robot.control,
            action_scale=ACTION_SCALE, action_scales_by_effort_limit_over_p_gain=False)),
        action=ActionManagerCfg(terms={'joint_control': ActionTermCfg(
            func='wbt_training.utils.guarded_action:GuardedJointPositionTargetActionTerm',
            params={}, scale=1.0, clip=None)}),
        reward=replace(cfg.reward, terms=rewards))
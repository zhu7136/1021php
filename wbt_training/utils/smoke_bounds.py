"""Fail-closed verification of the parsed smoke configuration, before Isaac."""
import json
import os

BOUND_ENV = 'HUD04_EXPECTED_SMOKE_BOUNDS'


def variant_override(variants):
    if not isinstance(variants, int) or isinstance(variants, bool) or variants < 1:
        raise ValueError('motion variants must be positive')
    return json.dumps({'add_onpath_obstacle': True, 'num_variants': variants}, sort_keys=True)


def verify_training_bounds(config):
    raw = os.environ.get(BOUND_ENV)
    if not raw:
        return None
    expected = json.loads(raw)
    hook = json.loads(config.training.preprocess_hook_kwargs)
    actual = {'num_envs': config.training.num_envs,
              'motion_variants': hook.get('num_variants'),
              'ppo_iterations': config.algo.config.num_learning_iterations}
    if hook.get('add_onpath_obstacle') is not True or actual != expected:
        raise ValueError('Parsed training configuration disagrees with approved smoke bounds')
    print(json.dumps(dict(event='parsed_training_bounds_verified', **actual), sort_keys=True), flush=True)
    return actual
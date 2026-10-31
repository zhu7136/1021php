"""Hu-D04 observations/assets with G1's DAgger+PPO combined-loss recipe.

CPU configuration only. This does not submit a task or authorize GPU spend.
"""
import copy
from .contract import checkpoint_contract
from .smoke_config import build_smoke_dict

def ppo_checkpoint_contract():
    result = checkpoint_contract()
    result.update(schema='hud04-three-teacher-dagger-ppo-v1', algorithm='dagger_ppo',
                  critic_width=575, ppo_start_epoch=0, dagger_end_epoch=10000)
    return result

def build_ppo_smoke_dict(teacher_template, **kwargs):
    config = build_smoke_dict(teacher_template, **kwargs)
    # Preserve the Hu-D04 privileged critic terms, not G1's observation layout.
    config['observation']['groups']['critic_obs'] = copy.deepcopy(
        teacher_template['observation']['groups']['critic_obs'])
    config['algo']['_target_'] = 'three_adapter.native_ppo.HuD04ThreeTeacherPPO'
    c = config['algo']['config']
    c['module']['critic_hidden_dims'] = [512, 256, 128]
    c.update(clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0,
             entropy_coef=0.001, learning_rate=3e-4, max_grad_norm=1.0,
             schedule='adaptive', desired_kl=0.01, init_noise_std=0.01,
             ppo_start_epoch=0, dagger_end_epoch=10000,
             dagger_loss_coef=10.0, weight_decay=0.0,
             distillation_warmup_steps=0)
    # The five-update/6-env bound is a pipeline test, NOT the 30000-iteration
    # production recipe. Never accelerate the PPO ramp to fake a PPO smoke.
    config['training']['name'] = 'hud04-three-teacher-dagger-ppo-validation'
    return config

def validate_ppo_checkpoint(data):
    if data.get('three_teacher_contract') != ppo_checkpoint_contract():
        raise ValueError('Reject pure-DAgger, five-skill or changed-teacher state')
    if data.get('num_teachers') != 3 or data.get('teacher_loaded') != [True]*3:
        raise ValueError('Three loaded teachers required')
    for key in ('model_state_dict', 'optimizer_state_dict', 'ppo_coef', 'learning_rate', 'iter'):
        if key not in data:
            raise ValueError('Incomplete PPO checkpoint: '+key)
    if not any(k.startswith('critic.') for k in data['model_state_dict']):
        raise ValueError('Missing critic parameters')
    if len(data['optimizer_state_dict']['param_groups']) != 4:
        raise ValueError('Expected student/critic/depth/std optimizer groups')
    return data
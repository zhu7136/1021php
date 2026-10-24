"""Keep simulator-owned tensors writable across PPO inference/eval boundaries."""
def install_environment_tensor_boundary(env):
    import torch
    if getattr(env, '_normal_tensor_boundary', False):
        raise ValueError('Environment tensor boundary already installed')
    original_step, original_reset = env.step, env.reset_all
    def step(actor_state):
        with torch.inference_mode(False), torch.no_grad():
            # Actor inference tensors must not become simulator-owned buffers.
            data = dict(actor_state)
            data['actions'] = actor_state['actions'].clone()
            return original_step(data)
    def reset_all(*args, **kwargs):
        with torch.inference_mode(False), torch.no_grad():
            return original_reset(*args, **kwargs)
    env.step, env.reset_all = step, reset_all
    env._normal_tensor_boundary = True
    return dict(environment_tensor_mode='normal_tensors_no_grad',
                policy_rollout_inference_unchanged=True, action_input_cloned=True)
"""Headless policy rollout on unchanged training terrain, then TWO native PPO updates.

No formal evaluation protocol, video, convergence or exact environment/RNG resume
claim. Live optimizer placement and upstream learning/indexing are unchanged.
"""
import copy
import json
import os
from pathlib import Path
import torch
from flux.resume_input import CHECKPOINT_SHA256, digest
from flux.teacher_algo import BoundedPPO, _require_same_state

MODULES = ('actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer')
OPTIMIZERS = ('actor_optimizer', 'critic_optimizer')
SCALARS = ('current_learning_iteration', 'actor_learning_rate', 'critic_learning_rate')
SOURCE_KEYS = dict(zip(MODULES + OPTIMIZERS, (
    'actor_model_state_dict', 'critic_model_state_dict',
    'actor_obs_normalizer_state_dict', 'critic_obs_normalizer_state_dict',
    'actor_optimizer_state_dict', 'critic_optimizer_state_dict')))


def snapshot(algo):
    return {name: copy.deepcopy(getattr(algo, name).state_dict()) for name in MODULES + OPTIMIZERS} | {
        name: copy.deepcopy(getattr(algo, name)) for name in SCALARS}


def finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError('Nonfinite replay/state tensor')
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite(child)


def require_loaded_state(algo, saved, before):
    """Honor native save/load's disabled-normalizer None/Identity contract.

    Do NOT enable normalization, fill checkpoint fields or relax tensor checks.
    A None is legitimate only for disabled, stateless native Identity modules.
    """
    for name, key in SOURCE_KEYS.items():
        if name in ('actor_obs_normalizer', 'critic_obs_normalizer'):
            enabled = getattr(algo, 'empirical_normalization', None)
            if enabled is False:
                if (saved[key] is not None or not isinstance(getattr(algo, name), torch.nn.Identity)
                        or before[name]):
                    raise ValueError('Invalid disabled native normalizer: ' + name)
                continue
            if enabled is not True or saved[key] is None:
                raise ValueError('Missing enabled native normalizer: ' + name)
        _require_same_state(saved[key], before[name], 'loaded.' + name)


def require_updated(before, after):
    changes = {}
    steps = {}
    for name in ('actor', 'critic'):
        changed = any(not torch.equal(v.detach().cpu(), after[name][k].detach().cpu())
                      for k, v in before[name].items())
        if not changed:
            raise ValueError('No parameter update: ' + name)
        changes[name] = changed
    for name in OPTIMIZERS:
        old, new = before[name]['state'], after[name]['state']
        if not old or old.keys() != new.keys():
            raise ValueError('Optimizer buffers missing or replaced: ' + name)
        deltas = []
        for key, row in old.items():
            previous = float(row['step'].detach().cpu().item())
            current = float(new[key]['step'].detach().cpu().item())
            if not current > previous:
                raise ValueError('Optimizer step did not advance: ' + name)
            deltas.append(current - previous)
        steps[name] = {'min_step_increment': min(deltas), 'max_step_increment': max(deltas)}
    finite(after)
    return changes, steps


class ReplayResumePPO(BoundedPPO):
    def learn(self):
        output = Path(os.environ['PHP_FLUX_OUTPUT'])
        source = Path(os.environ['PHP_FLUX_RESUME_CHECKPOINT'])
        if digest(source) != CHECKPOINT_SHA256 or self.config.num_learning_iterations != 2:
            raise ValueError('Unexpected resume source or update budget')
        if not self.config.load_optimizer or self.current_learning_iteration != 19:
            raise ValueError('Native checkpoint was not fully loaded before learn')
        saved = torch.load(source, weights_only=True, map_location='cpu')
        before = snapshot(self)
        require_loaded_state(self, saved, before)
        for name, optimizer in (('actor_learning_rate', 'actor_optimizer_state_dict'),
                                ('critic_learning_rate', 'critic_optimizer_state_dict')):
            _require_same_state(saved[optimizer]['param_groups'][0]['lr'], before[name], name)
        _require_same_state(saved['iter'], before['current_learning_iteration'], 'loaded.iteration')
        finite(before)
        (output / 'resume_loaded.json').write_text(json.dumps({
            'native_loaded_state_matches_source': True, 'source_sha256': CHECKPOINT_SHA256,
            'source_iteration': 19, 'checked_modules': list(MODULES), 'checked_optimizers': list(OPTIMIZERS),
            'empirical_normalization': self.empirical_normalization,
            'disabled_normalizers_verified_as_native_none_and_stateless_identity':
                not self.empirical_normalization}))
        self._eval_mode()
        policy = self.get_inference_policy()
        done_count = 0
        with torch.inference_mode():
            obs = self.env.reset_all()
            for step in range(32):
                finite(obs)
                actor_obs = torch.cat([obs[key] for key in self.actor_obs_keys], dim=1).to(self.device)
                actions = policy({'actor_obs': actor_obs})
                finite(actions)
                if tuple(actions.shape) != (self.env.num_envs, self.num_act):
                    raise ValueError('Unexpected replay action dimensions')
                obs, rewards, dones, _extras = self.env.step({'actions': actions})
                finite((obs, rewards, dones))
                done_count += int(dones.sum().item())
        after_replay = snapshot(self)
        _require_same_state(before, after_replay, 'read_only_replay')
        if digest(source) != CHECKPOINT_SHA256:
            raise ValueError('Source checkpoint changed during replay')
        (output / 'replay_validation.json').write_text(json.dumps({
            'passed': True, 'steps': 32, 'num_envs': self.env.num_envs,
            'environment_steps': 32 * self.env.num_envs, 'done_events': done_count,
            'all_observations_actions_rewards_finite': True,
            'model_normalizers_optimizer_scalars_unchanged': True, 'source_checkpoint_unchanged': True,
            'headless': True, 'video_recorded': False, 'formal_eval_protocol': False,
            'environment_mode': 'unchanged training terrain/commands; inference network mode'}))
        print(json.dumps({'event': 'checkpoint_replay_passed', 'steps': 32, 'num_envs': self.env.num_envs}), flush=True)
        # Native PPO.learn switches back to train mode and resets environments.
        # Keep native iteration range: starting at19, TWO updates end at index20.
        result = super().learn()
        after = snapshot(self)
        changes, steps = require_updated(before, after)
        if self._flux_completed_updates != 2 or self.current_learning_iteration != 20:
            raise ValueError('Incomplete resumed updates or unexpected native indexing')
        if digest(source) != CHECKPOINT_SHA256:
            raise ValueError('Source checkpoint changed during continuation')
        native = json.loads((output / 'native_reload.json').read_text())
        if native['native_load_passed'] is not True or native['completed_updates'] != 2:
            raise ValueError('Resumed final checkpoint roundtrip failed')
        (output / 'resume_validation.json').write_text(json.dumps({
            'passed': True, 'source_sha256': CHECKPOINT_SHA256, 'source_iteration': 19,
            'completed_updates': 2, 'native_final_iteration': 20, 'parameters_changed': changes,
            'optimizer_step_advancement': steps, 'source_checkpoint_unchanged': True,
            'final_checkpoint_native_reload_passed': True,
            'exact_environment_rng_resume_claimed': False}))
        print(json.dumps({'event': 'checkpoint_continuation_passed', 'completed_updates': 2}), flush=True)
        return result
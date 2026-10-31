"""Explicit route validity with native PPO loss unchanged.

Failed teacher rows mask DAgger only; reward/value learning remains active.
Zero-valued valid teacher actions must not be discarded. Preserve G1's full
batch-mean denominator rather than rescaling by the surviving label count.
"""
import time
import torch
from .masked_training import routing_validity

class ExplicitRoutePPOMixin:
    def _update_step(self, batch):
        budget = getattr(self, 'update_budget', None)
        if budget is not None and time.monotonic() >= budget.deadline:
            raise TimeoutError('Training deadline before PPO minibatch')
        if hasattr(self, '_dagger_route_valid'):
            raise RuntimeError('Nested update not supported')
        self._dagger_route_valid = routing_validity(batch['teacher_obs'], self.policy.routing_column)
        try:
            if self._warmup_active and not self._dagger_route_valid.any():
                # With no PPO/value loss in a teacher-driven warmup, do not
                # let AdamW momentum/weight decay move an unlabeled model.
                zero = batch['actor_obs'].new_zeros(())
                return {k: zero for k in ('total_loss','surrogate_loss','value_loss','behavior','entropy_mean','mean_kl')}
            return super()._update_step(batch)
        finally:
            del self._dagger_route_valid

    def _dagger_loss_per_sample(self, prediction, target):
        valid = self._dagger_route_valid
        if valid.shape != prediction.shape[:1]:
            raise ValueError('Shuffled routing mask/target batch mismatch')
        return (prediction-target).square().mean(-1) * valid.to(prediction.dtype)

    def _optimizer_step(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite DAgger/PPO loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.is_multi_gpu:
            self._reduce_parameters()
        if any(p.grad is not None or p.requires_grad for p in self.policy.teachers.parameters()):
            raise RuntimeError('Frozen teacher received gradients')
        params = [p for g in self.optimizer.param_groups for p in g['params']]
        torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.policy.project_action_std()
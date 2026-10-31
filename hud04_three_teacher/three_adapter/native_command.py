"""Runtime adapter. Import only inside the pinned native training environment."""
from holosoma.managers.command.terms.wbt import MotionCommand
from wbt_training.hud04_command_core import HuD04CommandMixin
from .command_contract import BalancedClipSampler

class BalancedSamplerProxy:
 """Use the core reset's existing global-timestep hook, not a copied reset.

 The underlying adaptive sampler still receives failure bookkeeping, but its
 probabilities DO NOT choose training timesteps. Any inherited bin metrics
 describe failure bookkeeping only, not the actual balanced sampling law.
 """
 def __init__(self,underlying,balanced):self.underlying=underlying;self.balanced=balanced
 def __getattr__(self,name):return getattr(self.underlying,name)
 def sample_global_time_steps(self,n):return self.balanced.sample(int(n))[1]

class HuD04BalancedMotionCommand(HuD04CommandMixin,MotionCommand):
 def setup(self):
  super().setup()
  if not self.motion_cfg.use_adaptive_timesteps_sampler:
   raise ValueError('Core global-timestep sampling hook must be enabled; balanced proxy replaces its probabilities')
  self.hud04_balanced_sampler=BalancedClipSampler(self.motion.motion_idxs,self.motion.motion_ends)
  self.adaptive_timesteps_sampler=BalancedSamplerProxy(self.adaptive_timesteps_sampler,self.hud04_balanced_sampler)
  # Keep core reset, HuD04 policy-state mapping and head-hold intact. Do not
  # inherit G1 PhpMotionCommand's reset or its unavailable vel_cmd attachment.
"""Single obstacle clip. Native physics/reset implementation is unchanged."""
from holosoma.managers.command.terms.wbt import MotionCommand
from wbt_training.hud04_command_core import HuD04CommandMixin

class HuD04FixedObstacleCommand(HuD04CommandMixin,MotionCommand):
    def setup(self):
        # Distillation.setup resets observations before learn(); select evaluation
        # mode before that first reset, not only after the model has loaded.
        self._env.set_is_evaluating()
        super().setup()
        cfg=self.motion_cfg
        expected={'use_adaptive_timesteps_sampler':False,'resample_on_motion_end':False,
            'start_at_timestep_zero_prob':1.,'freeze_at_timestep_zero_prob':0.,
            'default_pose_transition_strategy':'none','enable_default_pose_prepend':False,
            'enable_default_pose_append':False}
        if any(getattr(cfg,k)!=v for k,v in expected.items()):raise ValueError('Fixed clip configuration')
        if self.num_envs!=1 or self.motion.num_motions!=1 or int(self.motion.motion_ends.sum())!=1:
            raise ValueError('Exactly one isolated clip and physical environment required')
    def reset(self,env_ids):
        ids=self._ensure_index_tensor(env_ids)
        if ids.numel()==0:return
        if not self._env.is_evaluating:raise ValueError('Fixed preview must be evaluation')
        super().reset(ids)
        if int(self.time_steps[0])!=0:raise ValueError('Clip reset did not select the first frame')

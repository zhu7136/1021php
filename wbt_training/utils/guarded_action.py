"""Native implicit-PD target term with an explicit, observable target guard."""
import torch
from holosoma.managers.action.terms.joint_control import JointPositionTargetActionTerm
from wbt_training.utils.target_guard import protect_targets

class GuardedJointPositionTargetActionTerm(JointPositionTargetActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._requested_position_targets = torch.zeros_like(self._processed_actions)
        # Native BaseTask constructs ActionManager before its buffers/log_dict;
        # WBT adds default_dof_pos only after BaseTask.__init__ has returned.
        # Defer pose validation, but never permit actuation before it succeeds.
        self._default_pose_validated = False

    def _validate_default_pose(self):
        pose = getattr(self.env, 'default_dof_pos', None)
        if pose is None:
            raise RuntimeError('WBT default pose must be initialized before processing actions')
        safe = protect_targets(pose, self.env.simulator.hard_dof_pos_limits)
        if not torch.equal(safe, pose):
            raise ValueError('Default pose is outside guarded joint limits')
        self._default_pose_validated = True

    def process_actions(self, actions):
        if not self._default_pose_validated:
            self._validate_default_pose()
        # Preserve native raw-action/log-prob semantics. PPO still stores its
        # original Gaussian action, not the clipped physical target.
        super().process_actions(actions)
        requested = self.env.default_dof_pos + actions * self.action_scales
        self._requested_position_targets.copy_(requested)
        guarded = protect_targets(requested, self.env.simulator.hard_dof_pos_limits)
        self._processed_actions.copy_(guarded)
        self.env.log_dict['target_guard_fraction'] = (guarded != requested).float().mean()

    def apply_actions(self):
        if not self._default_pose_validated:
            raise RuntimeError('Cannot apply targets before default-pose validation')
        safe = protect_targets(self._processed_actions, self.env.simulator.hard_dof_pos_limits)
        if not torch.equal(safe, self._processed_actions):
            raise RuntimeError('Guarded target changed before native application')
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._requested_position_targets.zero_()
        else:
            self._requested_position_targets[env_ids] = 0.0
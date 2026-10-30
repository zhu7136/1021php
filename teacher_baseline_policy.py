"""Pure frozen high-step teacher inference; no student, depth, optimizer or simulator."""
import torch

@torch.no_grad()
def high_step_teacher_actions(policy,teacher_obs):
    if teacher_obs.ndim!=2 or teacher_obs.shape[1]!=450 or not torch.isfinite(teacher_obs).all():
        raise ValueError('Expected finite 450-wide privileged observations including route')
    col=policy.routing_column
    route=teacher_obs[:,col]
    if not torch.all((route==2)|(route==-1)):raise ValueError('Not the fixed route-2 clip')
    policy.require_loaded_teachers()
    privileged=torch.cat((teacher_obs[:,:col],teacher_obs[:,col+1:]),dim=-1)
    # Do not use the DAgger validity mask to zero actions when tracking drifts.
    # The same original native termination conditions decide whether to reset.
    result=policy.teachers[2](policy.teacher_obs_normalizers[2](privileged))
    if result.shape!=(teacher_obs.shape[0],29) or not torch.isfinite(result).all():
        raise ValueError('Invalid deterministic high-step teacher action')
    return result
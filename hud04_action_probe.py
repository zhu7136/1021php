"""Read-only same-state action/input diagnostics. Never selects control actions."""
import json
from pathlib import Path
import numpy as np
import torch


def tensor_stats(value):
    x=value.detach().float()
    finite=torch.isfinite(x)
    valid=x[finite]
    result={'shape':list(x.shape),'finite_fraction':float(finite.float().mean()),
            'zero_fraction':float((x==0).float().mean())}
    if valid.numel():
        result.update(min=float(valid.min()),max=float(valid.max()),mean=float(valid.mean()),
                      std=float(valid.std(unbiased=False)),rms=float(valid.square().mean().sqrt()))
    return result


class SameStateActionProbe:
    def __init__(self,policy,teacher_forward,output,live_contract,empirical_normalization=False):
        if empirical_normalization:
            raise ValueError('Unexpected student empirical normalization')
        self.policy=policy
        self.teacher_forward=teacher_forward
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.contract=live_contract
        widths=live_contract['widths']['actor_obs'];offsets=live_contract['offsets']['actor_obs']
        if sum(widths.values())!=96 or set(widths)!=set(offsets):
            raise ValueError('Actor layout does not match verified contract')
        occupied=[]
        for name,width in widths.items():occupied.extend(range(offsets[name],offsets[name]+width))
        if sorted(occupied)!=list(range(96)):raise ValueError('Actor layout overlaps or has gaps')
        self.count=0;self.compared_count=0;self.masked_count=0
        self.sum_squared=0.;self.max_abs_error=0.;self.previous_depth=None
        self.depth_changed_steps=0;self.snapshots={};self.per_joint_squared=torch.zeros(29,dtype=torch.float64)
        metadata={'schema':'hud04-same-state-action-probe-v1','student_empirical_normalization':False,
                  'teacher_normalization':'frozen checkpoint-specific native normalization',
                  'control_role':'student_only','teacher_actions_applied':False,
                  'action_units':'raw policy outputs, before native control scaling/clamping',
                  'live_contract':live_contract,'classification':'diagnosis only, not skill success'}
        (self.output/'action-probe-contract.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps({'event':'hud04_action_probe_contract',**metadata}),flush=True)

    @torch.no_grad()
    def observe(self,*,step,frame,actor,teacher,depth,student_actions,expected_route):
        if step!=self.count+1 or not 1<=step<=1000:raise ValueError('Probe step scope')
        if actor.shape!=(1,96) or teacher.shape!=(1,450) or depth.shape!=(1,58,87):
            raise ValueError('Observation shape contract')
        if student_actions.shape!=(1,29):raise ValueError('Student action shape')
        for value in [actor,teacher,depth,student_actions]:
            if not torch.isfinite(value).all():raise ValueError('Nonfinite action probe input')
        route_column=int(self.policy.routing_column)
        route=float(teacher[0,route_column])
        if expected_route!=2 or route not in (-1.,float(expected_route)):
            raise ValueError('Teacher observation route mismatch')
        # A read-only teacher forward uses precisely the current state's existing
        # privileged observation; no environment, command, or sensor is recomputed.
        # Native -1 means expert-invalid/loss-masked, not a different skill.
        # Preserve student control and native termination; do not compare a
        # masked/zero teacher output or override the mask to manufacture a target.
        target=None;delta=None
        if route!=-1.:
            target=self.teacher_forward(teacher)
            if target.shape!=(1,29) or not torch.isfinite(target).all():raise ValueError('Teacher action shape/finite')
            delta=student_actions-target
        actor_terms={name:tensor_stats(actor[:,self.contract['offsets']['actor_obs'][name]:
                               self.contract['offsets']['actor_obs'][name]+width])
                     for name,width in self.contract['widths']['actor_obs'].items()}
        change=None if self.previous_depth is None else float((depth-self.previous_depth).square().mean().sqrt())
        if change is not None and change>0:self.depth_changed_steps+=1
        self.previous_depth=depth.detach().clone()
        self.count+=1
        if delta is not None:
            squared=delta.detach().double().cpu().square().flatten()
            self.compared_count+=1;self.sum_squared+=float(squared.sum())
            self.per_joint_squared+=squared;self.max_abs_error=max(self.max_abs_error,float(delta.abs().max()))
        else:self.masked_count+=1
        record={'schema':'hud04-same-state-action-probe-v1','step':step,'reference_frame':frame,
                'route':int(route),'expected_route':expected_route,'teacher_actions_applied':False,
                'comparison_valid':target is not None,
                'comparison_skip_reason':'native_expert_invalid_mask' if target is None else None,
                'actor_stats':tensor_stats(actor),'actor_terms':actor_terms,'actor_values':actor[0].tolist(),
                'teacher_input_stats':tensor_stats(teacher),'depth_stats':tensor_stats(depth),
                'depth_change_rms_since_previous_observation':change,
                'student_action_stats':tensor_stats(student_actions),'teacher_action_stats':tensor_stats(target) if target is not None else None,
                'action_rmse':float(delta.square().mean().sqrt()) if delta is not None else None,
                'action_max_abs_error':float(delta.abs().max()) if delta is not None else None,
                'student_actions':student_actions[0].tolist(),'teacher_actions':target[0].tolist() if target is not None else None}
        with (self.output/'same-state-actions.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
        print(json.dumps({'event':'hud04_same_state_action',**record},allow_nan=False),flush=True)
        if step in [1,10,20,40,60,80,86] or step%100==0:
            self.snapshots[f'step_{step}_actor']=actor.detach().cpu().numpy().copy()
            self.snapshots[f'step_{step}_teacher']=teacher.detach().cpu().numpy().copy()
            self.snapshots[f'step_{step}_depth']=depth.detach().cpu().numpy().copy()

    def finish(self):
        result={'schema':'hud04-same-state-action-probe-v1','steps':self.count,
                'compared_steps':self.compared_count,'expert_invalid_steps':self.masked_count,
                'action_metric_denominator_steps':self.compared_count,
                'teacher_actions_applied':False,'action_units':'raw policy outputs',
                'aggregate_action_rmse':(self.sum_squared/(self.compared_count*29))**.5 if self.compared_count else None,
                'max_action_abs_error':self.max_abs_error if self.compared_count else None,
                'per_joint_rmse':(self.per_joint_squared/self.compared_count).sqrt().tolist() if self.compared_count else None,
                'depth_changed_observation_pairs':self.depth_changed_steps}
        if self.snapshots:np.savez_compressed(self.output/'selected-state-inputs.npz',**self.snapshots)
        (self.output/'same-state-action-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({'event':'hud04_same_state_action_summary',**result},allow_nan=False),flush=True)
        return result
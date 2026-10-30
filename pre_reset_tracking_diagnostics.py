"""Draft observational hook for an approved future native replay, not a launcher.

Captures native failure state before BaseTask.reset_envs_idx erases it.
Does not change actions, termination masks, thresholds, or physics.
Native Isaac integration remains untested; CPU regression uses a fake env.
"""
import json
from pathlib import Path
import torch

def tracking_snapshot(env,term):
    cmd=env.command_manager.get_state('motion_command')
    root_error=torch.norm(cmd.ref_pos_w-cmd.robot_ref_pos_w,dim=1)
    ids=term.bad_motion_body_pos_body_indexes
    body_error=torch.norm(cmd.body_pos_relative_w[:,ids]-cmd.robot_body_pos_w[:,ids],dim=-1)
    # These are the unchanged native term's pure boolean predicates, not a
    # second call to TerminationManager.check(), which mutates manager buffers.
    ref_pos=term.bad_ref_pos(cmd)
    ref_ori=term.bad_ref_ori(cmd)
    body_pos=term.bad_motion_body_pos(cmd)
    records=[]
    for i in env.reset_buf.nonzero().flatten().tolist():
        records.append({'env_id':i,'reference_frame':int(cmd.time_steps[i]),
            'episode_seconds':float(env.episode_length_buf[i])*float(env.dt),
            'timeout':bool(env.time_out_buf[i]),
            'bad_reference_position':bool(ref_pos[i]),
            'bad_reference_orientation':bool(ref_ori[i]),
            'bad_tracked_body_position':bool(body_pos[i]),
            'reference_position_error_m':float(root_error[i]),
            'tracked_body_errors_m':body_error[i].detach().cpu().tolist(),
            'tracked_body_names':[term.body_names_to_track[j] for j in ids.tolist()],
            'reference_position_threshold_m':float(term.bad_ref_pos_threshold),
            'tracked_body_position_threshold_m':float(term.bad_motion_body_pos_threshold),
            'root_xyz_before_reset':env.simulator.robot_root_states[i,:3].detach().cpu().tolist()})
    return records

class PreResetTrackingAudit:
    def __init__(self,env,output,max_checks=1005):
        if env.num_envs!=1 or set(env.termination_manager.cfg.terms)!={'timeout','bad_tracking'}:
            raise ValueError('Only audited single-env timeout + BadTracking config supported')
        cfg=env.termination_manager.cfg.terms['bad_tracking']
        if cfg.func!='holosoma.managers.termination.terms.wbt:BadTracking':
            raise ValueError('Native BadTracking implementation required')
        self.env=env;self.path=Path(output)/'pre-reset-tracking.jsonl'
        self.term=env.termination_manager._term_instances['bad_tracking']
        self.original=env._check_termination;self.calls=0;self.max_checks=max_checks
        self.installed=False

    def install(self):
        if self.installed:raise RuntimeError('Already installed')
        self.env._check_termination=self.check
        self.installed=True

    def check(self):
        self.calls+=1
        if self.calls>self.max_checks:raise RuntimeError('Read-only diagnostic step cap')
        result=self.original()  # Exactly one original termination evaluation.
        if bool(self.env.reset_buf.any()):
            rows=tracking_snapshot(self.env,self.term)
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a') as stream:
                for row in rows:
                    row['native_termination_check']=self.calls
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
        return result

    def restore(self):
        if self.installed:
            self.env._check_termination=self.original
            self.installed=False
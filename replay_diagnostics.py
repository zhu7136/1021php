"""Bounded observational telemetry; never changes actions, resets or physics."""
import json,math
from pathlib import Path

def values(tensor):
    return tensor.detach().cpu().tolist()

class ReplayDiagnostics:
    def __init__(self,env,output,max_steps=1000):
        self.env=env;self.output=Path(output);self.max_steps=max_steps
        self.steps=0;self.resets=0;self.wraps=0;self.distance=0.;self.initial=None;self.last=None
        self.actions_max_abs=0.;self.samples=[];self.last_phase=None

    def state(self):
        sim=self.env.simulator;command=self.env.command_manager.get_state('motion_command')
        root=values(sim.robot_root_states[0,:13]);phase=int(command.time_steps[0].item())
        return {'root_state':root,'joint_pos':values(sim.dof_pos[0]),'reference_frame':phase,
            'route':int(command.motion.motion_idxs[phase].item())}

    def invoke(self,original,request):
        if self.steps>=self.max_steps:raise RuntimeError('Diagnostic policy step cap')
        before=self.state();actions=values(request['actions'][0])
        if len(actions)!=29 or not all(math.isfinite(x) for x in actions):raise ValueError('Student action contract')
        result=original(request)  # Exactly one native step with the original actions.
        after=self.state();done=bool(result[2][0].item());self.steps+=1;self.resets+=int(done)
        wrapped=not done and after['reference_frame']<before['reference_frame'];self.wraps+=int(wrapped)
        distance=math.dist(before['root_state'][:2],after['root_state'][:2])
        if not done:self.distance+=distance  # Never count a reset teleport as locomotion.
        if self.initial is None:self.initial=before['root_state'][:3]
        self.last=after['root_state'][:3];self.last_phase=after['reference_frame']
        self.actions_max_abs=max(self.actions_max_abs,max(abs(x) for x in actions))
        if self.steps==1 or self.steps%10==0 or done or wrapped:
            sample={'step':self.steps,'done':done,'reference_wrapped':wrapped,
                'before_root_xyz':before['root_state'][:3],**after,
                'action_max_abs':max(abs(x) for x in actions),
                'action_rms':math.sqrt(sum(x*x for x in actions)/len(actions)),
                'joint_change_rms':math.sqrt(sum((a-b)**2 for a,b in zip(after['joint_pos'],before['joint_pos']))/len(after['joint_pos'])),
                'horizontal_step_m':distance,'includes_reset_teleport':done}
            self.samples.append(sample)
        return result

    def finish(self,error_type=None):
        report={'schema':'hud04-observational-replay-v1','observed_policy_steps':self.steps,
            'resets':self.resets,'reference_wraps_without_done':self.wraps,
            'horizontal_path_excluding_reset_steps_m':self.distance,
            'initial_root_xyz':self.initial,'final_root_xyz':self.last,'last_reference_frame':self.last_phase,
            'maximum_action_abs':self.actions_max_abs,'error_type':error_type,
            'training_updates':0,'physics_or_actions_modified':False,
            'nonrecorded_warmup_steps':getattr(self,'nonrecorded_steps',0),'samples':self.samples}
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'replay-diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({'event':'replay_diagnostics',**{k:v for k,v in report.items() if k!='samples'}},allow_nan=False),flush=True)
        return report

"""Student-only qualitative preview. Never performs an optimizer update."""
import hashlib,json,os,time
from pathlib import Path
import torch
from student_adapter.formal_algo import HuD04FormalDistillation

def model_digest(policy):
    h=hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class HuD04OfflineVideo(HuD04FormalDistillation):
    def _training_step(self):raise RuntimeError('Training is prohibited in offline video')
    def _rollout_step(self,*args,**kwargs):raise RuntimeError('Training rollout is prohibited')
    def learn(self,num_learning_iterations=None):
        request=json.loads(Path(os.environ['HUD04_VIDEO_REQUEST']).read_text())
        if request['scope']['training_updates']!=0:raise PermissionError('Inference only')
        if self.formal_next_iteration!=request['source_completed_updates']:raise ValueError('Wrong checkpoint counter')
        def denied(*args,**kwargs):raise RuntimeError('Optimizer step prohibited')
        self.optimizer.step=denied
        self.policy.eval()
        for p in self.policy.parameters():p.requires_grad_(False)
        before=model_digest(self.policy)
        self.env.set_is_evaluating()
        recorder=self.env.simulator.video_recorder
        if recorder is None or not recorder.enabled:raise ValueError('RGB recorder unavailable')
        # Keep all physics/termination/reset behavior. Disable only episode-based
        # video splitting, so falls and resets remain visible in one continuous clip.
        hooks=self.env.simulator.hooks
        for phase,records in list(hooks._records.items()):
            for record in list(records):
                if record.name in ['video.on_episode_start','video.on_episode_end']:
                    hooks._disable(record)
        if recorder.is_recording:recorder.stop_recording()
        obs=self.env.reset_all()
        self.policy.eval()
        dt=float(self.env.dt)
        steps=round(request['scope']['preview_seconds']/dt)
        if not 1<=steps<=request['scope']['max_policy_steps']:raise ValueError('Preview step budget')
        policy=self.get_inference_policy()
        recorder.start_recording(episode_id=0)
        resets=0;routes=[]
        with torch.inference_mode():
            for step in range(steps):
                if time.monotonic()>=float(os.environ['HUD04_VIDEO_DEADLINE']):raise TimeoutError('Preview deadline')
                actor,_,depth=self._extract_groups(obs)
                actions=policy({'actor_obs':actor,'depth_obs':depth})
                if actions.shape!=(self.env.num_envs,29) or not torch.isfinite(actions).all():raise ValueError('Invalid student action')
                obs,rewards,dones,extras=self.env.step({'actions':actions})
                resets+=int(dones[0].item())
                command=self.env.command_manager.get_state('motion_command')
                routes.append(int(command.motion.motion_idxs[command.time_steps[0]].item()))
        frame_count=len(recorder.video_frames)
        if frame_count<steps//2:raise ValueError('Insufficient recorded RGB frames')
        # Reject a black/static stream rather than reporting camera warmup as a video.
        import numpy as np
        samples=np.stack([recorder.video_frames[i] for i in [frame_count//4,frame_count//2,frame_count-1]])
        if float(samples.std())<2 or np.array_equal(samples[0],samples[-1]):raise ValueError('Black or static RGB stream')
        recorder.stop_recording()
        after=model_digest(self.policy)
        if before!=after:raise ValueError('Policy mutated during preview')
        out=Path(request['output'])
        videos=list((out/'videos').glob('*.mp4'))
        if not videos:raise ValueError('No encoded video')
        (out/'video-evidence.json').write_text(json.dumps({'schema':'hud04-offline-preview-v1',
            'task_id':request['task_id'],'_wb_agent':request['_wb_agent'],
            'source_checkpoint_sha256':request['source_checkpoint_sha256'],
            'source_completed_updates':self.formal_next_iteration,'training_updates':0,
            'model_parameters_unchanged':True,'model_sha256':after,
            'policy_steps':steps,'control_dt':dt,'simulated_seconds':steps*dt,'captured_frames':frame_count,
            'env0_resets':resets,'env0_route_counts':{str(i):routes.count(i) for i in sorted(set(routes))},
            'preview_only':True,'skills_evaluated':False,'videos':[str(p.relative_to(out)) for p in videos]},indent=2))
        print('STUDENT_PREVIEW_RENDERED_WITH_ZERO_TRAINING',flush=True)

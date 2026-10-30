"""Wrap the existing pure-student preview with observational diagnostics."""
import json,os
from pathlib import Path
from obstacle_video_algo import HuD04ObstacleVideo
from replay_diagnostics import ReplayDiagnostics

class HuD04DiagnosticVideo(HuD04ObstacleVideo):
    def learn(self,num_learning_iterations=None):
        request=json.loads(Path(os.environ['HUD04_VIDEO_REQUEST']).read_text())
        if request['scope']['training_updates']!=0 or request['scope']['max_policy_steps']>1000:
            raise PermissionError('Pure inference only; at most 1000 steps')
        diagnostics=ReplayDiagnostics(self.env,request['output'],request['scope']['max_policy_steps'])
        original=self.env.step;error=None
        diagnostics.nonrecorded_steps=0
        def observed_step(inputs):
            # Native reset_all() performs one zero-action physics warmup before
            # recording starts. It is not a policy action or a video frame.
            if not self.env.simulator.video_recorder.is_recording:
                diagnostics.nonrecorded_steps+=1
                if diagnostics.nonrecorded_steps>4:raise RuntimeError('Nonrecorded reset warmup cap')
                return original(inputs)
            return diagnostics.invoke(original,inputs)
        self.env.step=observed_step
        try:
            return super().learn(num_learning_iterations)
        except BaseException as exc:
            error=type(exc).__name__;raise
        finally:
            self.env.step=original
            try:diagnostics.finish(error)
            except Exception as exc:
                print(json.dumps({'event':'replay_diagnostics_write_failed','error_type':type(exc).__name__}),flush=True)
                if error is None:raise

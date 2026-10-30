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
        self.env.step=lambda inputs:diagnostics.invoke(original,inputs)
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

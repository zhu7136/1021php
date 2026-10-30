"""Zero-update teacher baseline on the existing fixed high-step native preview."""
import json,os
from pathlib import Path
from diagnostic_video_algo import HuD04DiagnosticVideo
from teacher_baseline_policy import high_step_teacher_actions
from pre_reset_tracking_diagnostics import PreResetTrackingAudit

class HuD04TeacherBaseline(HuD04DiagnosticVideo):
    def get_inference_policy(self,device=None):
        # The inherited preview requests a callable but passes only student
        # inputs. Deliberately ignore them; fetch current privileged observations.
        # modify_history=False and teacher enable_noise=False prevent changes.
        def frozen_teacher(_student_inputs):
            obs=self.env.observation_manager.compute_group('teacher_obs',modify_history=False)
            return high_step_teacher_actions(self.policy,obs)
        return frozen_teacher

    def learn(self,num_learning_iterations=None):
        req=json.loads(Path(os.environ['HUD04_VIDEO_REQUEST']).read_text())
        if req['scope']['skills']!=['high-step'] or req['scope']['max_tasks']!=1:
            raise PermissionError('Single high-step baseline only')
        audit=PreResetTrackingAudit(self.env,req['output'],max_checks=1005)
        audit.install()
        try:
            super().learn(num_learning_iterations)
        finally:
            audit.restore()
        path=Path(req['output'])/'video-evidence.json'
        evidence=json.loads(path.read_text())
        evidence.update(executed_policy='frozen_high_step_teacher',teacher_index=2,
            teacher_source_run='lh2kb3v0',student_used_for_actions=False,
            teacher_privileged_observations=True,zeroed_by_dagger_validity_mask=False,
            pre_reset_tracking_diagnostics_enabled=True,
            success_not_inferred_from_video_existence=True)
        path.write_text(json.dumps(evidence,indent=2))
        print(json.dumps({'event':'teacher_baseline_complete','env0_resets':evidence['env0_resets'],
            'executed_policy':evidence['executed_policy'],'training_updates':0}),flush=True)
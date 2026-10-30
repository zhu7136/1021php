"""HuD04 single-episode diagnostic; native GPU integration NOT yet validated."""
import json
import os
import time
from pathlib import Path

import torch
from student_adapter.formal_algo import HuD04FormalDistillation
from offline_video_algo import model_digest
from hud04_episode_probe import EpisodeProbe, diagnostic_outcome


class HuD04SingleEpisode(HuD04FormalDistillation):
    def _training_step(self):
        raise PermissionError('Training forbidden during evaluation')

    def _rollout_step(self, *args, **kwargs):
        raise PermissionError('Training rollout forbidden during evaluation')

    def learn(self, num_learning_iterations=None):
        request = json.loads(Path(os.environ['HUD04_VIDEO_REQUEST']).read_text())
        scope = request['scope']
        role = request['evaluation_role']
        maximum = scope['max_policy_steps']
        if (role not in ('student', 'teacher') or scope['training_updates'] != 0
                or scope['automatic_retries'] != 0 or self.env.num_envs != 1
                or type(maximum) is not int or not 1 <= maximum <= 1000):
            raise PermissionError('Unapproved diagnostic scope')
        if self.formal_next_iteration != request['source_completed_updates']:
            raise ValueError('Wrong restored student counter')
        def denied(*args, **kwargs):
            raise PermissionError('Optimizer/teacher access forbidden in student evaluation')
        self.optimizer.step = denied
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        probe_enabled=request.get('same_state_action_probe',False)
        if type(probe_enabled) is not bool or (probe_enabled and role!='student'):
            raise ValueError('Same-state probe requires explicit student-only scope')
        teacher_forward=self.policy.teacher_act if probe_enabled else None
        if role == 'student':
            self.policy.teacher_act = denied
        before = model_digest(self.policy)
        self.env.set_is_evaluating()
        output = Path(request['output'])
        action_probe=None
        if probe_enabled:
            from hud04_action_probe import SameStateActionProbe
            action_probe=SameStateActionProbe(self.policy,teacher_forward,output,self.live_contract,
                empirical_normalization=self.config.empirical_normalization)
        probe = EpisodeProbe(self.env, output)
        report = {
            'schema': 'hud04-single-episode-diagnostic-v1',
            'role': role, 'source_checkpoint_sha256': request['source_checkpoint_sha256'],
            'source_completed_updates': self.formal_next_iteration,
            'case': request['obstacle_case'], '_wb_agent': request['_wb_agent'],
            'training_updates': 0, 'policy_steps': 0,
            'task_success': None, 'skills_evaluated': False,
            'scope': 'native termination diagnosis only; no obstacle-success certification',
            'outcome': 'engineering_invalid_unfinished',
            'setup_resets_before_learn_not_observed': True,
        }
        error = None
        probe.install()
        try:
            with torch.inference_mode():
                # Native reset_all includes a zero-action warmup step. Its
                # termination is recorded separately, never hidden from results.
                obs = self.env.reset_all()
                report['startup_observation'] = probe.latest
                report['outcome'] = diagnostic_outcome(probe.latest)
                if report['outcome'] != 'running':
                    return
                probe.phase = 'policy'
                command = self.env.command_manager.get_state('motion_command')
                previous_frame = int(command.time_steps[0].item())
                for step in range(1, maximum + 1):
                    if time.monotonic() >= float(os.environ['HUD04_VIDEO_DEADLINE']):
                        raise TimeoutError('Evaluation deadline')
                    actor, teacher, depth = self._extract_groups(obs)
                    if actor.shape != (1, 96):
                        raise ValueError('Wrong actor observation contract')
                    current_frame = int(command.time_steps[0].item())
                    actual_route = int(command.motion.motion_idxs[current_frame].item())
                    if actual_route != request['obstacle_case']['route']:
                        raise ValueError('Wrong expert/case route')
                    if role == 'student':
                        actions = self.policy.act_inference({'actor_obs': actor, 'depth_obs': depth})
                    else:
                        actions = self.policy.teacher_act(teacher)
                    if actions.shape != (1, 29) or not torch.isfinite(actions).all():
                        raise ValueError('Invalid action output')
                    if action_probe is not None:
                        action_probe.observe(step=step,frame=current_frame,actor=actor,teacher=teacher,
                            depth=depth,student_actions=actions,expected_route=actual_route)
                    probe.policy_steps = step
                    checks_before = probe.check_count
                    obs, rewards, dones, extras = self.env.step({'actions': actions})
                    if probe.check_count != checks_before + 1:
                        raise RuntimeError('Expected exactly one native termination check per control step')
                    report['policy_steps'] = step
                    terminal = probe.latest
                    if bool(dones[0].item()) != (terminal['terminated'] or terminal['timeout']):
                        raise RuntimeError('Native done/termination masks disagree')
                    frame = terminal['reference_frame_before_reset']
                    report['outcome'] = diagnostic_outcome(
                        terminal, reference_wrapped=frame < previous_frame, horizon=step == maximum)
                    previous_frame = frame
                    if report['outcome'] != 'running':
                        break
        except BaseException as exc:
            error = exc
            report['outcome'] = 'engineering_invalid_exception'
            report['error_type'] = type(exc).__name__
            raise
        finally:
            probe.restore()
            if action_probe is not None:
                report['same_state_action_probe']=action_probe.finish()
            report.update(terminal_observation=probe.latest,
                          tracking_cache_sync=dict(getattr(self.env,'hud04_cache_sync',{})),
                          termination_events=probe.events,
                          simulated_policy_seconds=report['policy_steps'] * float(self.env.dt),
                          model_parameters_unchanged=before == model_digest(self.policy),
                          process_exit_success_not_yet_known=True)
            if not report['model_parameters_unchanged']:
                report['outcome'] = 'engineering_invalid_model_mutation'
            output.mkdir(parents=True, exist_ok=True)
            (output / 'single-episode-result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps({'event': 'hud04_single_episode_diagnostic', **report}), flush=True)
            if not report['model_parameters_unchanged'] and error is None:
                raise RuntimeError('Model changed during inference')
"""Read-only continuous-segment accounting at native command-update boundaries.

Termination is checked BEFORE command advance in the supported IsaacSim WBT path.
An internal reset after step k ends the old segment at k*dt; it is censoring,
not survival, failure, or success. No native reset/termination call is added.
"""
from __future__ import annotations
import math
import torch


class SegmentLedger:
    """CPU-compatible state machine, also usable for explicitly inferred replay."""
    def __init__(self, num_envs, dt, horizon):
        if num_envs <= 0 or horizon <= 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid ledger dimensions')
        self.n, self.dt, self.horizon = num_envs, dt, horizon
        self.starts = [0] * num_envs
        self.segment_ids = [0] * num_envs
        self.alive = [True] * num_envs
        self.records, self.events = [], []
        self.step = 0
        self.finished = False

    def _end(self, env, step, outcome):
        start = self.starts[env]
        if step <= start:
            raise ValueError('Nonpositive segment duration')
        self.records.append(dict(env=env, segment_id=self.segment_ids[env],
            start_boundary_step=start, end_boundary_step=step,
            duration_steps=step-start, duration_seconds=(step-start)*self.dt,
            outcome=outcome, failure_observed=outcome == 'native_termination',
            censored=outcome != 'native_termination'))

    def advance(self, step, done, reset_events):
        if self.finished or step != self.step + 1 or step > self.horizon:
            raise ValueError('Nonconsecutive or out-of-budget ledger step')
        if len(done) != self.n or any(type(v) is not bool for v in done):
            raise ValueError('Expected boolean done vector')
        ids = [r['env'] for r in reset_events]
        if len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < self.n for i in ids):
            raise ValueError('Invalid or repeated reset environment')
        for event in reset_events:
            i = event['env']
            if self.alive[i]:
                self.events.append(dict(event, boundary_after_step=step,
                    boundary_seconds=step*self.dt, segment_id=self.segment_ids[i],
                    native_termination_same_step=done[i]))
        for i in range(self.n):
            if not self.alive[i]:
                continue
            if done[i]:
                self._end(i, step, 'native_termination')
                self.alive[i] = False
            elif i in ids:
                self._end(i, step, 'censored_internal_reset')
                self.starts[i] = step
                self.segment_ids[i] += 1
        self.step = step

    def finish(self):
        if self.finished or self.step != self.horizon:
            raise ValueError('Incomplete or already finalized ledger')
        for i in range(self.n):
            if self.alive[i] and self.starts[i] < self.horizon:
                self._end(i, self.horizon, 'censored_horizon')
        self.finished = True
        first = sorted((r for r in self.records if r['segment_id'] == 0), key=lambda r:r['env'])
        if len(first) != self.n:
            raise ValueError('Missing first-segment outcomes')
        return dict(schema_version=1, native_environment_episode_metrics_preserved=True,
            timing='Native termination precedes command update; reset boundary is after step k',
            censoring_warning='Observed duration is NOT an unbiased survival estimate; clip-end censoring may be informative',
            first_segment_mean_observed_seconds=sum(r['duration_seconds'] for r in first)/self.n,
            first_segment_native_terminations=sum(r['failure_observed'] for r in first),
            first_segment_internal_reset_censored=sum(r['outcome']=='censored_internal_reset' for r in first),
            first_segment_horizon_survivors=sum(r['outcome']=='censored_horizon' for r in first),
            internal_reset_events=self.events, segments=self.records)


class ContinuousSegmentCapture:
    """Wrap one existing command callback, once per evaluation step; restore on exit."""
    def __init__(self, env, motion, *, horizon=128):
        if env._update_tasks_before_termination:
            raise ValueError('Continuous accounting requires post-termination command update')
        flag = getattr(motion, 'motion_end_reset', None)
        if not torch.is_tensor(flag) or flag.shape != (env.num_envs,) or flag.dtype != torch.bool:
            raise ValueError('Explicit native motion_end_reset boolean vector required')
        self.env, self.motion = env, motion
        self.ledger = SegmentLedger(env.num_envs, float(env.dt), horizon)
        self.current_step = None
        self.done = None
        self.calls = 0
        self.original = env._update_tasks_callback
        self.had_instance = '_update_tasks_callback' in vars(env)
        self.installed = True
        self.result = None
        env._update_tasks_callback = self._updated

    def begin_step(self, step):
        if self.current_step is not None or step != self.ledger.step + 1:
            raise ValueError('Previous control step was not completed')
        self.current_step, self.calls = step, 0
        self._alive_before = self.ledger.alive.copy()

    @torch.no_grad()
    def _updated(self, *args, **kwargs):
        if self.current_step is None or self.calls:
            raise ValueError('Unexpected command callback count')
        self.calls += 1
        m = self.motion
        before = m.time_steps.detach().clone()
        # reset_buf is the native PRE-command termination decision. Never recompute it.
        done = self.env.reset_buf.bool().detach().cpu().tolist()
        self.done = done
        result = self.original(*args, **kwargs)
        flag = m.motion_end_reset
        if flag.shape != (self.env.num_envs,) or flag.dtype != torch.bool:
            raise ValueError('Native reset flag contract changed')
        after = m.time_steps.detach().clone()
        events = [dict(env=i, frame_before=int(before[i]), frame_after=int(after[i]),
            evidence='native_motion_end_reset') for i in flag.nonzero().flatten().cpu().tolist()]
        self.ledger.advance(self.current_step, done, events)
        return result

    def end_step(self, done):
        if self.calls != 1 or self.current_step is None:
            raise ValueError('Missing command update')
        actual = done.bool().detach().cpu().tolist()
        # Ledger alive states must match native done, including previously dead envs.
        expected_alive = [a and not b for a,b in zip(self._alive_before, actual)]
        if actual != self.done or expected_alive != self.ledger.alive:
            raise ValueError('Native termination changed after command callback')
        self.current_step = None

    def finish(self):
        if self.current_step is not None:
            raise ValueError('Unfinished control step')
        self.result = self.ledger.finish()
        self.result['boundary_evidence'] = 'explicit_native_motion_end_reset_after_command_update'
        return self.result

    def close(self):
        if self.installed:
            if self.had_instance:
                self.env._update_tasks_callback = self.original
            else:
                delattr(self.env, '_update_tasks_callback')
            self.installed = False
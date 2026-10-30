"""Observational, single-environment termination probe. No simulator imports.

Wraps native callbacks exactly once and snapshots state before native auto-reset.
Does not redefine termination, alter actions, or classify obstacle success.
"""
import json
from pathlib import Path
from hud04_tracking_details import TrackingProxy


def plain(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class TermProxy:
    def __init__(self, target, callback):
        self.target = target
        self.callback = callback

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        result = self.target(*args, **kwargs)
        self.callback(plain(result))
        return result


class EpisodeProbe:
    def __init__(self, env, output):
        if env.num_envs != 1:
            raise ValueError('Single physical environment required')
        self.env = env
        self.manager = env.termination_manager
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.latest = None
        self.phase = 'initialization'
        self.policy_steps = 0
        self.check_count = 0
        self.events = []
        self.term_values = {}
        self.tracking_details = {}
        self.original_check = None
        self.original_funcs = None
        self.original_instances = None

    def install(self):
        if self.original_check is not None:
            raise RuntimeError('Already installed')
        m = self.manager
        self.original_check = m.check
        self.original_funcs = m._term_funcs
        self.original_instances = m._term_instances
        def wrap(name, term):
            if name == 'bad_tracking':
                return TrackingProxy(term, lambda value: self.term_values.__setitem__(name, value),
                    lambda detail: self.tracking_details.__setitem__(name, detail))
            return TermProxy(term, lambda value: self.term_values.__setitem__(name, value))
        m._term_funcs = {n: wrap(n, t) for n, t in m._term_funcs.items()}
        m._term_instances = {n: wrap(n, t) for n, t in m._term_instances.items()}
        m.check = self.check

    def restore(self):
        if self.original_check is not None:
            m = self.manager
            m.check = self.original_check
            m._term_funcs = self.original_funcs
            m._term_instances = self.original_instances
            self.original_check = None

    def check(self):
        self.term_values = {}
        self.tracking_details = {}
        result = self.original_check()
        reset, timeout = result
        command = self.env.command_manager.get_state('motion_command')
        self.check_count += 1
        values = dict(self.term_values)
        snapshot = {
            'phase': self.phase,
            'policy_step': self.policy_steps,
            'check_count': self.check_count,
            'terminated': bool(plain(reset)[0]),
            'timeout': bool(plain(timeout)[0]),
            'terms': values,
            'tracking_details': dict(self.tracking_details),
            'triggered_terms': [n for n, v in values.items() if bool(v[0])],
            'timeout_terms': [n for n, c in zip(self.manager._term_names, self.manager._term_cfgs) if c.is_timeout],
            'root_state_before_reset': plain(self.env.simulator.robot_root_states[0]),
            'joint_pos_before_reset': plain(self.env.simulator.dof_pos[0]),
            'reference_frame_before_reset': int(plain(command.time_steps)[0]),
        }
        self.latest = snapshot
        # Every check is flushed; a native interpreter shutdown crash must not
        # erase the last episode's terminal evidence.
        with (self.output / 'termination-checks.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot, allow_nan=False) + '\n')
            stream.flush()
        if snapshot['terminated'] or snapshot['timeout']:
            self.events.append(snapshot)
        return result


def diagnostic_outcome(snapshot, reference_wrapped=False, horizon=False):
    """No reference end, timeout, or survival horizon is task-success proof."""
    if snapshot is None:
        return 'engineering_invalid_missing_termination_observation'
    if snapshot['phase'] != 'policy' and (snapshot['terminated'] or snapshot['timeout']):
        return 'engineering_invalid_initialization_termination'
    if snapshot['terminated']:
        return 'native_termination'
    if snapshot['timeout']:
        return 'native_timeout'
    if reference_wrapped:
        return 'reference_wrapped_not_success'
    if horizon:
        return 'horizon_reached_not_success'
    return 'running'
"""Local, opt-in numeric depth diagnostics. No simulator imports or tasks."""
from dataclasses import replace
import json
import time

from hud04_depth_reset_guard import PendingDepthResetMixin

NATIVE_TERMS = {
    'holosoma.managers.observation.terms.depth:WarpDepthImageObsTerm',
    'holosoma.managers.observation.terms.depth.WarpDepthImageObsTerm',
}


def options(value):
    if (not isinstance(value, dict) or set(value) != {'fix_pending_reset', 'max_events'}
            or type(value['fix_pending_reset']) is not bool
            or type(value['max_events']) is not int
            or not 32 <= value['max_events'] <= 512):
        raise ValueError('Depth diagnostic options rejected')
    return dict(value)


def configure_depth_diagnostics(cfg, request):
    """Pure config edit, only for an explicitly requested one-env diagnostic."""
    if 'depth_reset_diagnostics' not in request:
        return cfg
    selected = options(request['depth_reset_diagnostics'])
    scope = request['scope']
    if (request.get('evaluation_role') not in ('student', 'teacher')
            or type(scope.get('training_updates')) is not int or scope['training_updates'] != 0
            or type(scope.get('automatic_retries')) is not int or scope['automatic_retries'] != 0
            or type(scope.get('max_policy_steps')) is not int or not 1 <= scope['max_policy_steps'] <= 1000
            or type(cfg.training.num_envs) is not int or cfg.training.num_envs != 1):
        raise PermissionError('Depth diagnostics require single-env zero-training scope')
    group = cfg.observation.groups['depth_camera']
    term = group.terms['depth_cam']
    if term.func not in NATIVE_TERMS or 'hud04_depth_diagnostics' in term.params:
        raise ValueError('Unexpected depth term configuration')
    updated = replace(term, func='hud04_depth_runtime:HuD04DiagnosticDepth',
                      params={**term.params, 'hud04_depth_diagnostics': selected})
    groups = {**cfg.observation.groups,
              'depth_camera': replace(group, terms={**group.terms, 'depth_cam': updated})}
    return replace(cfg, observation=replace(cfg.observation, groups=groups))


class NumericDepthEventsMixin:
    """Preserve native capture/latency; optional reset guard plus bounded logs.

    No sensor read, command lookup, RNG draw, or buffer write is added by logging.
    Host monotonic timestamps are NOT sensor exposure timestamps.
    Logging exceptions are counted and suppressed; native exceptions propagate.
    """
    def __init__(self, cfg, env):
        if type(env.num_envs) is not int or env.num_envs != 1:
            raise PermissionError('Depth diagnostics require one environment')
        self.hud04_depth_options = options(cfg.params['hud04_depth_diagnostics'])
        self.hud04_depth_state = {'events_emitted': 0, 'truncated': False,
                                  'logging_failed': False, 'logging_failures': 0,
                                  'capture_calls': 0, 'completed_captures': 0,
                                  'observation_calls': 0, 'reset_calls': 0}
        self._hud04_start_ns = time.monotonic_ns()
        super().__init__(cfg, env)
        self._depth_event('constructed')

    def _write_depth_event(self, record):
        print(json.dumps(record, allow_nan=False, separators=(',', ':')), flush=True)

    def _depth_event(self, kind, **fields):
        state = self.hud04_depth_state
        if state['logging_failed'] or state['truncated']:
            return
        try:
            record = {'event': 'hud04_depth_reset_trace', 'schema': 'hud04-depth-reset-trace-v1',
                      'sequence': state['events_emitted'] + 1,
                      'host_elapsed_ns': time.monotonic_ns() - self._hud04_start_ns,
                      'kind': kind, 'fix_pending_reset': self.hud04_depth_options['fix_pending_reset'],
                      'capture_calls': state['capture_calls'],
                      'completed_captures': state['completed_captures']}
            if state['events_emitted'] >= self.hud04_depth_options['max_events'] - 1:
                record['kind'] = 'event_limit'; state['truncated'] = True
            else:
                command = getattr(self, 'command', None)
                record.update(pending_reset=bool(self.reset_episodes[0].item()),
                              reference_frame=None if command is None else int(command.time_steps[0].item()),
                              **fields)
            self._write_depth_event(record)
            state['events_emitted'] += 1
        except Exception:
            state['logging_failed'] = True
            state['logging_failures'] += 1

    def reset(self, env_ids=None):
        self.hud04_depth_state['reset_calls'] += 1
        self._depth_event('reset_before', reset_count=self.num_envs if env_ids is None else int(env_ids.numel()))
        try:
            if self.hud04_depth_options['fix_pending_reset']:
                PendingDepthResetMixin.reset(self, env_ids)
            else:
                super().reset(env_ids)
        except BaseException:
            self._depth_event('native_reset_failed')
            raise
        self._depth_event('reset_after')

    def _get_depth_images(self, env):
        self.hud04_depth_state['capture_calls'] += 1
        self._depth_event('capture_before')
        try:
            result = super()._get_depth_images(env)
        except BaseException:
            self._depth_event('native_capture_failed')
            raise
        self.hud04_depth_state['completed_captures'] += 1
        self._depth_event('capture_after')
        return result

    def __call__(self, env, *, modify_history=True, **kwargs):
        self.hud04_depth_state['observation_calls'] += 1
        self._depth_event('observation_before', history_write=bool(modify_history))
        try:
            result = super().__call__(env, modify_history=modify_history, **kwargs)
        except BaseException:
            self._depth_event('native_observation_failed', history_write=bool(modify_history))
            raise
        # Statistics are strictly diagnostic. Never mask a successful native
        # observation just because a statistic/log transport fails.
        if self.hud04_depth_state['logging_failed'] or self.hud04_depth_state['truncated']:
            return result
        try:
            all_zero = bool(result.eq(0).all().item())
        except Exception:
            all_zero = None
            self.hud04_depth_state['logging_failed'] = True
            self.hud04_depth_state['logging_failures'] += 1
        self._depth_event('observation_after', history_write=bool(modify_history), output_all_zero=all_zero)
        return result


def depth_diagnostic_status(env):
    manager = getattr(env, 'observation_manager', None)
    terms = getattr(manager, '_term_instances', {}).get('depth_camera', {})
    term = terms.get('depth_cam')
    if term is None or not hasattr(term, 'hud04_depth_state'):
        return {'enabled': False}
    return {'enabled': True, **term.hud04_depth_options, **term.hud04_depth_state,
            'timestamp_clock': 'host_monotonic_not_sensor_exposure',
            'native_gpu_validated': False}
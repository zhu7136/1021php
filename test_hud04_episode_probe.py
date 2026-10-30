import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hud04_episode_probe import EpisodeProbe, diagnostic_outcome, plain


class StatefulTerm:
    def __init__(self):
        self.calls = 0
        self.reset_calls = 0

    def __call__(self, env, **kwargs):
        self.calls += 1
        return torch.tensor([env.trigger])

    def reset(self, env_ids=None):
        self.reset_calls += 1


class FakeManager:
    def __init__(self, env):
        self.env = env
        self.term = StatefulTerm()
        self._term_names = ['tracking', 'timeout']
        self._term_cfgs = [SimpleNamespace(is_timeout=False), SimpleNamespace(is_timeout=True)]
        self._term_instances = {'tracking': self.term}
        self._term_funcs = {'timeout': lambda env: torch.tensor([env.timeout])}

    def check(self):
        return self._term_instances['tracking'](self.env), self._term_funcs['timeout'](self.env)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.command = SimpleNamespace(time_steps=torch.tensor([10]))
        self.env = SimpleNamespace(num_envs=1, trigger=False, timeout=False,
            simulator=SimpleNamespace(robot_root_states=torch.tensor([[1., 2., 3.]]), dof_pos=torch.tensor([[.1, .2]])),
            command_manager=SimpleNamespace(get_state=lambda name: self.command))
        self.env.termination_manager = FakeManager(self.env)
        self.probe = EpisodeProbe(self.env, self.tmp.name)
        self.probe.install()

    def tearDown(self):
        self.probe.restore()
        self.tmp.cleanup()

    def test_native_callbacks_called_once(self):
        result = self.env.termination_manager.check()
        self.assertEqual(self.env.termination_manager.term.calls, 1)
        self.assertFalse(result[0].item())

    def test_terminal_snapshot_survives_auto_reset(self):
        self.env.trigger = True
        self.env.termination_manager.check()
        self.env.simulator.robot_root_states.fill_(999.)
        self.command.time_steps.zero_()
        self.assertEqual(self.probe.latest['root_state_before_reset'], [1., 2., 3.])
        self.assertEqual(self.probe.latest['reference_frame_before_reset'], 10)

    def test_stateful_reset_forwarded(self):
        self.env.termination_manager._term_instances['tracking'].reset()
        self.assertEqual(self.env.termination_manager.term.reset_calls, 1)

    def test_records_exact_trigger_names(self):
        self.env.trigger = True
        self.env.termination_manager.check()
        self.assertEqual(self.probe.latest['triggered_terms'], ['tracking'])

    def test_timeout_separate_from_termination(self):
        self.env.timeout = True
        reset, timeout = self.env.termination_manager.check()
        self.assertFalse(reset.item())
        self.assertTrue(timeout.item())
        self.assertEqual(self.probe.latest['timeout_terms'], ['timeout'])

    def test_startup_termination_not_hidden(self):
        self.env.trigger = True
        self.env.termination_manager.check()
        self.assertEqual(diagnostic_outcome(self.probe.latest), 'engineering_invalid_initialization_termination')

    def test_policy_termination_is_not_called_fall(self):
        self.env.trigger = True
        self.probe.phase = 'policy'
        self.env.termination_manager.check()
        self.assertEqual(diagnostic_outcome(self.probe.latest), 'native_termination')

    def test_timeout_not_task_success(self):
        self.env.timeout = True
        self.probe.phase = 'policy'
        self.env.termination_manager.check()
        self.assertEqual(diagnostic_outcome(self.probe.latest), 'native_timeout')

    def test_horizon_not_task_success(self):
        self.env.termination_manager.check()
        self.assertEqual(diagnostic_outcome(self.probe.latest, horizon=True), 'horizon_reached_not_success')

    def test_reference_wrap_not_task_success(self):
        self.env.termination_manager.check()
        self.assertEqual(diagnostic_outcome(self.probe.latest, reference_wrapped=True), 'reference_wrapped_not_success')

    def test_missing_termination_observation_invalid(self):
        self.assertEqual(diagnostic_outcome(None), 'engineering_invalid_missing_termination_observation')

    def test_output_is_incremental(self):
        self.env.termination_manager.check()
        self.env.termination_manager.check()
        rows = [json.loads(s) for s in (Path(self.tmp.name) / 'termination-checks.jsonl').read_text().splitlines()]
        self.assertEqual([r['check_count'] for r in rows], [1, 2])

    def test_restores_original_manager(self):
        manager = self.env.termination_manager
        self.probe.restore()
        self.assertIs(manager._term_instances['tracking'], manager.term)
        self.assertEqual(manager.check.__func__, FakeManager.check)

    def test_multiple_environments_rejected(self):
        self.env.num_envs = 2
        with self.assertRaises(ValueError):
            EpisodeProbe(self.env, self.tmp.name)

    def test_duplicate_install_rejected(self):
        with self.assertRaises(RuntimeError):
            self.probe.install()

    def test_tensor_conversion_detaches_copy(self):
        value = torch.tensor([2.], requires_grad=True)
        self.assertEqual(plain({'value': value}), {'value': [2.]})


if __name__ == '__main__':
    unittest.main()
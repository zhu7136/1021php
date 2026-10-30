"""CPU fake-environment integration tests; not a native physics validation."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_hud04_episode_probe import FakeManager


def digest(policy):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in policy.state_dict().values())).hexdigest()


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.student_calls = 0
        self.teacher_calls = 0

    def act_inference(self, obs):
        self.student_calls += 1
        return torch.zeros(1, 29)

    def teacher_act(self, obs):
        self.teacher_calls += 1
        return torch.ones(1, 29)


def load_algorithm():
    parent = types.ModuleType('student_adapter')
    formal = types.ModuleType('student_adapter.formal_algo')
    formal.HuD04FormalDistillation = object
    preview = types.ModuleType('offline_video_algo')
    preview.model_digest = digest
    with patch.dict(sys.modules, {'student_adapter': parent, 'student_adapter.formal_algo': formal, 'offline_video_algo': preview}):
        spec = importlib.util.spec_from_file_location('isolated_episode_algo', Path(__file__).with_name('hud04_single_episode_algo.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.HuD04SingleEpisode


Algorithm = load_algorithm()


class AlgorithmTests(unittest.TestCase):
    def run_case(self, role='student', fail_at=2, startup_failure=False, maximum=8):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            command = SimpleNamespace(time_steps=torch.tensor([1]), motion=SimpleNamespace(motion_idxs=torch.full((30,), 2)))
            env = SimpleNamespace(num_envs=1, dt=.02, trigger=False, timeout=False,
                simulator=SimpleNamespace(robot_root_states=torch.tensor([[1.,2.,3.]]), dof_pos=torch.tensor([[.1,.2]])),
                command_manager=SimpleNamespace(get_state=lambda name: command), steps=0)
            env.termination_manager = FakeManager(env)
            env.set_is_evaluating = lambda: None
            def step(inputs):
                env.steps += 1
                env.trigger = (startup_failure and env.steps == 1) or (fail_at is not None and env.steps == fail_at + 1)
                reset, timeout = env.termination_manager.check()
                done = reset | timeout
                if done.item():
                    env.simulator.robot_root_states.fill_(999.)
                    command.time_steps.zero_()
                    env.termination_manager._term_instances['tracking'].reset()
                else:
                    command.time_steps += 1
                return {}, torch.zeros(1), done.clone(), {}
            env.step = step
            env.reset_all = lambda: step({'actions':torch.zeros(1,29)})[0]
            runner = Algorithm()
            runner.env = env
            runner.policy = Policy()
            runner.optimizer = SimpleNamespace(step=lambda: None)
            runner.formal_next_iteration = 3686
            runner._extract_groups = lambda obs: (torch.zeros(1,96),torch.zeros(1,450),torch.zeros(1,58,87))
            req = {'scope':{'training_updates':0,'automatic_retries':0,'max_policy_steps':maximum},
                'evaluation_role':role, 'source_completed_updates':3686,
                'source_checkpoint_sha256':'test-fixture-only','obstacle_case':{'route':2},
                'output':str(root),'_wb_agent':{'thread_id':'test','turn_id':'test'}}
            request = root/'request.json'
            request.write_text(json.dumps(req))
            with patch.dict(os.environ, {'HUD04_VIDEO_REQUEST':str(request),'HUD04_VIDEO_DEADLINE':str(time.monotonic()+20)}):
                runner.learn()
            result = json.loads((root/'single-episode-result.json').read_text())
            return result, runner, env

    def test_stops_at_first_policy_termination(self):
        result, runner, env = self.run_case()
        self.assertEqual(result['policy_steps'],2)
        self.assertEqual(env.steps,3)  # one warmup plus two policy actions
        self.assertEqual(result['outcome'],'native_termination')

    def test_student_does_not_use_teacher_actions(self):
        result, runner, env = self.run_case()
        self.assertEqual(runner.policy.student_calls,2)
        self.assertEqual(runner.policy.teacher_calls,0)
        with self.assertRaises(PermissionError):
            runner.policy.teacher_act(None)

    def test_teacher_role_does_not_use_student_actions(self):
        result, runner, env = self.run_case(role='teacher')
        self.assertEqual(runner.policy.teacher_calls,2)
        self.assertEqual(runner.policy.student_calls,0)

    def test_pre_reset_terminal_state_preserved(self):
        result, runner, env = self.run_case()
        self.assertEqual(result['terminal_observation']['root_state_before_reset'],[1.,2.,3.])
        self.assertEqual(env.simulator.robot_root_states[0].tolist(),[999.,999.,999.])

    def test_initialization_failure_aborts_before_policy(self):
        result, runner, env = self.run_case(startup_failure=True)
        self.assertEqual(result['policy_steps'],0)
        self.assertEqual(result['outcome'],'engineering_invalid_initialization_termination')

    def test_horizon_is_not_success(self):
        result, runner, env = self.run_case(fail_at=None,maximum=3)
        self.assertEqual(result['outcome'],'horizon_reached_not_success')
        self.assertIsNone(result['task_success'])

    def test_optimizer_disabled_and_weights_unchanged(self):
        result, runner, env = self.run_case()
        self.assertTrue(result['model_parameters_unchanged'])
        with self.assertRaises(PermissionError):
            runner.optimizer.step()

    def test_training_methods_disabled(self):
        runner = Algorithm()
        with self.assertRaises(PermissionError):
            runner._training_step()
        with self.assertRaises(PermissionError):
            runner._rollout_step({})


if __name__ == '__main__':
    unittest.main()
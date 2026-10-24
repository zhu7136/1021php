"""CPU mock tests: finite budget, original termination evaluation, pre-reset capture, no updates."""
import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
import torch
import cv2  # Keep native modules loaded across temporary sys.modules mock scopes.

FLUX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLUX))
spec = importlib.util.spec_from_file_location("finite_replay_under_test", FLUX / "replay_teacher_finite.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

video_path = FLUX.parent / 'thirdparty/holosoma/src/holosoma/holosoma/utils/video_utils.py'
video_spec = importlib.util.spec_from_file_location('pinned_replay_video_utils', video_path)
video_utils = importlib.util.module_from_spec(video_spec)
with patch.dict(sys.modules, {'loguru': types.SimpleNamespace(logger=Mock())}):
    video_spec.loader.exec_module(video_utils)


@dataclasses.dataclass
class Train:
    preprocess_hook_kwargs: str = '{"add_onpath_obstacle":true,"num_variants":32}'
    num_envs: int = 32
    headless: bool = True
    max_eval_steps: int = 0
    export_onnx: bool = True
    registry_name: str = "original"


@dataclasses.dataclass
class AlgoInner:
    load_optimizer: bool = True


@dataclasses.dataclass
class Algo:
    config: AlgoInner = dataclasses.field(default_factory=AlgoInner)
    _target_: str = "FakePPO"


@dataclasses.dataclass
class Config:
    training: Train
    algo: Algo
    robot: object
    reward: object
    logger: object = None
    def save_config(self, path):
        Path(path).write_text("mock_config: true\n")


class PermissiveConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Termination:
    def __init__(self, env):
        self._term_instances = {}
        self._term_funcs = {
            "fall_proxy": lambda e: torch.tensor([e.steps % 100 == 0]),
            "timeout": lambda e: torch.tensor([e.steps % 250 == 0]),
        }


class Environment:
    def __init__(self, names):
        self.steps, self.num_envs, self.dt = 0, 1, 0.02
        self.reset_calls = 0
        native_names = list(reversed(names))
        native_pos = torch.zeros(1, 29)
        self.simulator = types.SimpleNamespace(
            _robot=types.SimpleNamespace(joint_names=native_names, data=types.SimpleNamespace(joint_pos=native_pos)),
            dof_ids=list(reversed(range(29))), dof_pos=torch.zeros(1,29), dof_vel=torch.zeros(1,29),
            body_names=["foot", "pelvis", "hand"], video_recorder=None,
            robot_root_states=torch.tensor([[0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]]),
            _rigid_body_pos=torch.zeros(1,3,3), contact_forces_history=torch.zeros(1,4,3,3))
        self.reset_buf = torch.zeros(1,dtype=torch.bool)
        self.time_out_buf = torch.zeros(1,dtype=torch.bool)
        self.termination_manager = Termination(self)
    def _check_termination(self):
        terms = self.termination_manager._term_funcs
        self.time_out_buf[:] = terms["timeout"](self)
        self.reset_buf[:] = terms["fall_proxy"](self) | self.time_out_buf
    def set_is_evaluating(self):
        self.is_evaluating = True
    def reset_all(self):
        self.reset_calls += 1
        return self.step({"actions":torch.zeros(1,29)})[0]
    def step(self, state):
        assert self.steps < 1000, "Exceeded approved control-step budget"
        self.steps += 1
        self.simulator.contact_forces_history.zero_()
        if self.steps % 10 == 0:
            self.simulator.contact_forces_history[0,0,1,2] = 25.
        self._check_termination()
        # Simulate automatic reset clearing flags and contacts after terminal-state hook.
        self.reset_buf.zero_()
        self.time_out_buf.zero_()
        self.simulator.contact_forces_history.zero_()
        return {"actor_obs":torch.zeros(1,29)},None,None,None


class FakePPO:
    def __init__(self, env, config, log_dir, device):
        self.env = env
        assert config.load_optimizer is False
        env.reset_all()  # The real PPO constructor discards this warm-up result.
        self.actor = torch.nn.Linear(29,29)
        self.current_learning_iteration = 1
    def setup(self):
        pass
    def load(self, path):
        pass
    def get_inference_policy(self, device):
        self.actor.eval()
        return lambda obs: torch.zeros(1,29)


def modules_for_test(env, cfg):
    modules = {}
    def add(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        module.__path__ = []
        modules[name] = module
        if "." in name:
            parent, field = name.rsplit(".", 1)
            if parent not in modules:
                add(parent)
            setattr(modules[parent], field, module)
        return module
    add("holosoma.utils.eval_utils", CheckpointConfig=PermissiveConfig)
    add("holosoma.config_types.logger", DisabledLoggerConfig=PermissiveConfig)
    add("holosoma.config_types.video", VideoConfig=PermissiveConfig, SphericalCameraConfig=PermissiveConfig)
    def setup(c):
        assert c.logger.video.output_format == 'mp4'
        def save_video():
            video_utils.create_video(np.zeros((3,360,640,3),dtype=np.uint8),50,
                Path(c.logger.video.save_dir),output_format=c.logger.video.output_format,
                wandb_logging=False,episode_id=1)
        env.simulator.video_recorder = types.SimpleNamespace(stop_recording=save_video)
        return env,"cuda:0",None
    add("holosoma.utils.sim_utils", setup_isaaclab_launcher=lambda:None,
        setup_simulation_environment=setup, close_simulation_app=lambda app:None)
    add("holosoma.utils.video_utils", create_video=video_utils.create_video)
    add("holosoma.utils.helpers", get_class=lambda name:FakePPO)
    add("wbt_training.utils.checkpoint", load_saved_experiment_config=lambda c:(cfg,None), apply_preprocess_hook=lambda c:c)
    add("startup_diagnostics", bounded_launcher=lambda f:f)
    return modules


class ReplayTests(unittest.TestCase):
    def test_full_1000_step_mock_replay(self):
        names = ["joint_"+str(i) for i in range(29)]
        cfg = Config(Train(), Algo(), types.SimpleNamespace(dof_names=names,
            dof_pos_lower_limit_list=[-3.]*29, dof_pos_upper_limit_list=[3.]*29),
            types.SimpleNamespace(terms={"undesired_contacts": types.SimpleNamespace(
                params={"undesired_contacts_body_names":"^pelvis$", "threshold":1.})}))
        env = Environment(names)
        with tempfile.TemporaryDirectory(dir="/workspace") as directory:
            root = Path(directory)
            inputs = root / "flux/replay-input"
            inputs.mkdir(parents=True)
            checkpoint = inputs / "selected.pt"
            checkpoint.write_bytes(b"immutable_mock_checkpoint")
            out = root / "result"
            with patch.dict(sys.modules, modules_for_test(env,cfg)), \
                    patch.object(replay,"ROOT",root), patch.object(replay,"OUT",out), \
                    patch.object(replay,"EXPECTED_SHA256",hashlib.sha256(checkpoint.read_bytes()).hexdigest()), \
                    contextlib.redirect_stdout(io.StringIO()):
                previous_grad = torch.is_grad_enabled()
                previous_argv = sys.argv[:]
                try:
                    replay.main(root)
                finally:
                    torch.set_grad_enabled(previous_grad)
                    sys.argv = previous_argv
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(env.steps,1000)
            self.assertEqual(env.reset_calls,1)
            self.assertEqual(summary["policy_steps"],999)
            self.assertEqual(summary["resets"],12)
            self.assertEqual(summary["timeouts"],4)
            self.assertEqual(summary["termination_counts"],{"fall_proxy":10,"timeout":4})
            self.assertEqual(summary["steps_with_undesired_contact"],100)
            self.assertTrue(summary["actor_weights_unchanged"])
            self.assertTrue(summary["checkpoint_unchanged"])
            self.assertEqual(summary["training_updates"],0)
            self.assertEqual(summary['video_output_format'],'mp4')
            from replay_integrity import verify_completion
            self.assertEqual(verify_completion(out, summary['checkpoint_sha256'])['steps'], 1000)
            manifest = json.loads((out/'completion.json').read_text())
            self.assertEqual(next(iter(manifest['videos'].values()))['frames'], 3)
            with np.load(out / "trajectory.npz",allow_pickle=False) as data:
                self.assertTrue(all(len(data[key])==1000 for key in data.files))

    def test_initial_reset_failure_cannot_seal_completion(self):
        cfg=Config(Train(), Algo(), types.SimpleNamespace(dof_names=['joint_'+str(i) for i in range(29)]),
            types.SimpleNamespace(terms={'undesired_contacts':types.SimpleNamespace(
                params={'undesired_contacts_body_names':'pelvis','threshold':1.})}))
        env=Environment(cfg.robot.dof_names)
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            root=Path(directory)
            inputs=root/'flux/replay-input'; inputs.mkdir(parents=True)
            checkpoint=inputs/'selected.pt'; checkpoint.write_bytes(b'fixture')
            out=root/'result'
            modules=modules_for_test(env,cfg)
            with patch.dict(sys.modules,modules), patch.object(replay,'ROOT',root), \
                    patch.object(replay,'OUT',out), \
                    patch.object(replay,'EXPECTED_SHA256',hashlib.sha256(b'fixture').hexdigest()), \
                    patch.object(env,'reset_all',side_effect=RuntimeError('initial reset fixture')), \
                    contextlib.redirect_stdout(io.StringIO()):
                previous_grad=torch.is_grad_enabled(); previous_argv=sys.argv[:]
                try:
                    with self.assertRaisesRegex(RuntimeError,'initial reset fixture'):
                        replay.main(root)
                finally:
                    torch.set_grad_enabled(previous_grad); sys.argv=previous_argv
            self.assertFalse((out/'summary.json').exists())
            self.assertFalse((out/'completion.json').exists())
            self.assertEqual(env.steps,0)

    def test_correct_source_checkpoint_hash(self):
        source = Path("/workspace/flux_inspect_TASK_20261024_030/model_00001.pt")
        if source.exists():
            self.assertEqual(replay.digest(source),replay.EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
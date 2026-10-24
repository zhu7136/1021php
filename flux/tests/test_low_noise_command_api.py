"""CPU contract regression for TASK_20261024_047's real initialization failure."""
import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
import train_low_noise as entry


class MotionCommandAPITests(unittest.TestCase):
    def motion(self):
        return NS(time_steps=[0],motion_cfg=NS(use_adaptive_timesteps_sampler=True))

    def test_pinned_native_get_state_returns_the_exact_motion_instance(self):
        source=ROOT/'thirdparty/holosoma/src/holosoma/holosoma/managers/command/manager.py'
        cls=next(n for n in ast.parse(source.read_text()).body
                 if isinstance(n,ast.ClassDef) and n.name=='CommandManager')
        self.assertNotIn('get_term',[n.name for n in cls.body if isinstance(n,ast.FunctionDef)])
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_state')
        ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec',
                          flags=__import__('__future__').annotations.compiler_flag),ns)
        motion=self.motion();manager=NS(_state_terms={'motion_command':motion})
        manager.get_state=types.MethodType(ns['get_state'],manager)
        self.assertIs(entry.resolve_motion_command(NS(command_manager=manager)),motion)
        self.assertTrue(motion.motion_cfg.use_adaptive_timesteps_sampler)

    def test_worker_uses_the_checked_resolver_not_get_term(self):
        tree=ast.parse(Path(entry.__file__).read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=='resolve_motion_command' for n in calls))
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr=='get_term' for n in calls))

    def test_manager_without_get_state_is_rejected(self):
        with self.assertRaisesRegex(TypeError,'get_state'):
            entry.resolve_motion_command(NS(command_manager=NS()))

    def test_missing_motion_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing'):
            entry.resolve_motion_command(NS(command_manager=NS(get_state=lambda name:None)))

    def test_incomplete_motion_state_is_rejected(self):
        for motion in (NS(),NS(motion_cfg=NS()),NS(time_steps=[])):
            with self.subTest(motion=motion),self.assertRaisesRegex(TypeError,'evaluation state'):
                entry.resolve_motion_command(NS(command_manager=NS(get_state=lambda name:motion)))

    def test_missing_sampler_configuration_is_rejected(self):
        motion=NS(time_steps=[],motion_cfg=NS())
        with self.assertRaisesRegex(TypeError,'sampler configuration'):
            entry.resolve_motion_command(NS(command_manager=NS(get_state=lambda name:motion)))


if __name__=='__main__':unittest.main()
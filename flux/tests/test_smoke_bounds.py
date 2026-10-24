"""Smoke counts are verified on parsed config before any Isaac setup."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from wbt_training.utils.smoke_bounds import BOUND_ENV, variant_override, verify_training_bounds


class SmokeBoundsTests(unittest.TestCase):
    def setUp(self):
        self.expected=dict(num_envs=32,motion_variants=32,ppo_iterations=2)
        self.cfg=NS(training=NS(num_envs=32,preprocess_hook_kwargs=variant_override(32)),
                    algo=NS(config=NS(num_learning_iterations=2)))

    def verify(self):
        with patch.dict(os.environ,{BOUND_ENV:json.dumps(self.expected)}),redirect_stdout(io.StringIO()):
            return verify_training_bounds(self.cfg)

    def test_parsed_approved_values_pass(self):
        self.assertEqual(self.verify(),self.expected)

    def test_preset_64_variants_is_rejected(self):
        self.cfg.training.preprocess_hook_kwargs=variant_override(64)
        with self.assertRaises(ValueError):self.verify()

    def test_wrong_environment_count_rejected(self):
        self.cfg.training.num_envs=200
        with self.assertRaises(ValueError):self.verify()

    def test_wrong_iteration_count_rejected(self):
        self.cfg.algo.config.num_learning_iterations=10000
        with self.assertRaises(ValueError):self.verify()

    def test_disabled_obstacles_rejected(self):
        self.cfg.training.preprocess_hook_kwargs=json.dumps(dict(add_onpath_obstacle=False,num_variants=32))
        with self.assertRaises(ValueError):self.verify()

    def test_general_training_without_bound_is_unchanged(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertIsNone(verify_training_bounds(object()))

    def test_variant_override_validation(self):
        self.assertEqual(json.loads(variant_override(32))['num_variants'],32)
        for bad in (0,-1,True,'32'):
            with self.assertRaises(ValueError):variant_override(bad)

    def test_bounds_check_is_before_simulator_setup(self):
        source=(ROOT/'wbt_training/train_agent.py').read_text()
        tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls={n.func.id:n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertLess(calls['verify_training_bounds'],calls['setup_simulator_imports'])


if __name__=='__main__':unittest.main()
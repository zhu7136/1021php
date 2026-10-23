import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('flux_launcher', ROOT/'flux/train_hud04.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class SourceChecks(unittest.TestCase):
    def test_recovered_hashes_and_submodule_base(self):
        self.assertEqual(launcher.inspect_sources()['motion_terrain_pairs'], 56)

    def test_all_data_shapes_names_and_finiteness(self):
        tree = ast.parse((ROOT/'flux/holosoma_overlay/config_values/robot.py').read_text())
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('_HU_D04_DOF_NAMES', '_HU_D04_BODY_NAMES'):
                        values[target.id] = ast.literal_eval(node.value)
        for path in (ROOT/'flux/data/motions-low-step-hu-d04').glob('*_motion.npz'):
            with self.subTest(motion=path.name), np.load(path, allow_pickle=False) as motion:
                self.assertEqual(motion['joint_names'].tolist(), values['_HU_D04_DOF_NAMES'])
                self.assertEqual(motion['body_names'].tolist(), values['_HU_D04_BODY_NAMES'])
                frames = motion['joint_pos'].shape[0]
                self.assertEqual(motion['joint_pos'].shape, (frames, 36))
                self.assertEqual(motion['joint_vel'].shape, (frames, 35))
                self.assertEqual(motion['body_pos_w'].shape, (frames, 32, 3))
                for key in motion.files:
                    array = motion[key]
                    if array.dtype.kind in 'fiu':
                        self.assertTrue(np.isfinite(array).all(), key)
            terrain = np.load(path.with_name(path.name.replace('_motion.npz','_terrain.npy')), allow_pickle=False)
            self.assertEqual(terrain.ndim, 3)
            self.assertEqual(terrain.shape[2], 10)
            self.assertTrue(np.isfinite(terrain).all())

    def test_default_mode_does_not_launch_runtime(self):
        p = subprocess.run([sys.executable,str(ROOT/'flux/train_hud04.py')],capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('motion_terrain_pairs',p.stdout)
        self.assertNotIn('training_start',p.stdout)

    def test_reject_invalid_training_bounds(self):
        p = subprocess.run([sys.executable,str(ROOT/'flux/train_hud04.py'),'--iterations','0'],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,2)


if __name__ == '__main__':
    unittest.main()
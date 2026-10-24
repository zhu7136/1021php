import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'diagnose_short_episodes.py'
spec=importlib.util.spec_from_file_location('short_diag',PATH)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class ShortDiagnosticTests(unittest.TestCase):
    def test_fixed_budget(self):
        self.assertEqual((d.NUM_ENVS,d.STEPS_PER_MODE,d.WORKER_SECONDS),(32,256,900))
        self.assertEqual(d.MODES,('policy_mean','policy_sample','reference_pd'))
    def test_preserved_checkpoint(self):
        self.assertEqual(d.SHA,'634f343946005e72303b9c100aa90f6a9ff76e8224316a59af1f0842e3edaf85')
    def test_flat_triangle_height(self):
        t=np.array([[[0,0,2],[1,0,2],[0,1,2]]])
        self.assertEqual(d.heights_at_xy(SimpleNamespace(triangles=t),[[.1,.1,100],[2,2,0]]),[2.,None])
    def test_highest_surface(self):
        t=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,.4],[1,0,.4],[0,1,.4]]])
        self.assertAlmostEqual(d.heights_at_xy(SimpleNamespace(triangles=t),[[.1,.1,0]])[0],.4)
    def test_sloped_triangle_height(self):
        t=np.array([[[0,0,0],[1,0,1],[0,1,1]]])
        self.assertAlmostEqual(d.heights_at_xy(SimpleNamespace(triangles=t),[[.2,.3,0]])[0],.5)
    def test_vertical_faces_ignored(self):
        t=np.array([[[0,0,0],[0,0,1],[0,1,0]]])
        self.assertEqual(d.heights_at_xy(SimpleNamespace(triangles=t),[[0,.1,0]]),[None])
    def test_no_learning_or_save_calls(self):
        tree=ast.parse(PATH.read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ['learn','backward','zero_grad','save_checkpoint','save_checkpoint_artifact']:
            self.assertNotIn(forbidden,calls)
        self.assertIn('algo.learn=forbidden;algo.save=forbidden',PATH.read_text())
    def test_diagnostic_checks_preserve_termination(self):
        text=PATH.read_text()
        self.assertIn('original_check()',text)
        self.assertIn("assert torch.equal(done,expected)",text)
        self.assertIn('actor_weights_unchanged',text)
        self.assertIn('right_censored_environment_ages',text)
    def test_secrets_not_forwarded(self):
        text=PATH.read_text()
        self.assertIn("['key','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']",text)
        self.assertIn('e.pop(n,None)',text)

if __name__=='__main__':unittest.main()
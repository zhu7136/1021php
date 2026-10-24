import ast,subprocess,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose_action_scale as entry

class Tests(unittest.TestCase):
    def test_baseline_identity(self):
        a=torch.randn(32,29);self.assertIs(entry.transform(None,'policy_mean',a),a)
    def test_reference_identity(self):
        a=torch.randn(32,29);self.assertIs(entry.transform(None,'reference_pd',a),a)
    def test_quarter_output_all_joints(self):
        a=torch.linspace(-8,8,32*29).reshape(32,29);old=a.clone()
        self.assertTrue(torch.equal(entry.transform(None,'policy_mean_scaled025',a),a*.25));self.assertTrue(torch.equal(a,old))
    def test_zero_means_default_pose(self):
        a=torch.randn(32,29);default=torch.randn(32,29)*.1
        out=entry.transform(None,'default_pose_pd',a)
        self.assertTrue(torch.equal(out,torch.zeros_like(a)));self.assertTrue(torch.equal(default+out,default))
    def test_nonfinite_fails_closed(self):
        for mode in ('policy_mean_scaled025','default_pose_pd'):
            with self.assertRaises(AssertionError):entry.transform(None,mode,torch.tensor([[float('nan')]]))
    def test_bounds(self):
        self.assertEqual((entry.NUM_ENVS,entry.STEPS_PER_MODE,entry.WORKER_SECONDS),(32,64,600))
        self.assertEqual(entry.MODES,('policy_mean','policy_mean_scaled025','default_pose_pd','reference_pd'))
    def test_bootstrap_stdlib_only(self):
        flux=Path(__file__).resolve().parents[1]
        code=f'import sys;sys.path.insert(0,{str(flux)!r});import diagnose_action_scale as e;e.configure();assert "torch" not in sys.modules;assert "typing_extensions" not in sys.modules;assert "control_chain_recorder" not in sys.modules;assert len(e.base.MODES)==4'
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr[-2000:])
    def test_no_training_or_clamping(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'diagnose_action_scale.py').read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ['learn','backward','zero_grad','save_checkpoint','clamp']:self.assertNotIn(forbidden,attrs)

if __name__=='__main__':unittest.main(verbosity=2)
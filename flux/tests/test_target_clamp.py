import ast,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose_target_clamp as entry

def fixture():
    limits=torch.tensor([[-.8727,2.094],[-2.094,.8727],[-.436,.436]])
    return NS(action_scales=torch.tensor([1.,.5,2.]),default_dof_pos=torch.tensor([[.005,-.008,.03],[-.004,.003,-.05]]),
        simulator=NS(hard_dof_pos_limits=limits),robot_config=NS(control=NS(clip_actions=True,action_clip_value=100.)))

def target(env,a):
    return env.default_dof_pos+a.clamp(-100,100)*env.action_scales

class Tests(unittest.TestCase):
    def test_baseline_identity(self):
        env=fixture();a=torch.full((2,3),9.);self.assertIs(entry.transform(env,'policy_mean',a),a)
    def test_reference_identity(self):
        env=fixture();a=torch.full((2,3),9.);self.assertIs(entry.transform(env,'reference_pd',a),a)
    def test_inside_identity(self):
        env=fixture();a=torch.zeros(2,3);out=entry.transform(env,'policy_mean_clamped',a);self.assertTrue(torch.equal(out,a))
    def test_signed_limits_and_offsets(self):
        env=fixture();a=torch.tensor([[-4.,4.,-4.],[4.,-8.,4.]])
        out=entry.transform(env,'policy_mean_clamped',a);q=target(env,out);lim=env.simulator.hard_dof_pos_limits
        self.assertTrue(((q>=lim[:,0])&(q<=lim[:,1])).all())
        self.assertAlmostEqual(q[0,0].item(),-.8727+entry.MARGIN_RAD,places=6)
        self.assertAlmostEqual(q[1,1].item(),-2.094+entry.MARGIN_RAD,places=6)
    def test_does_not_mutate_inputs(self):
        env=fixture();a=torch.full((2,3),4.);saved=[x.clone() for x in [a,env.default_dof_pos,env.action_scales,env.simulator.hard_dof_pos_limits]]
        entry.transform(env,'policy_mean_clamped',a)
        for old,new in zip(saved,[a,env.default_dof_pos,env.action_scales,env.simulator.hard_dof_pos_limits]):self.assertTrue(torch.equal(old,new))
    def test_raw_clip_preserved(self):
        env=fixture();a=torch.tensor([[1e5,-1e5,1e5],[-1e5,1e5,-1e5]])
        q=target(env,entry.transform(env,'policy_mean_clamped',a));lim=env.simulator.hard_dof_pos_limits
        self.assertTrue(((q>=lim[:,0])&(q<=lim[:,1])).all())
    def test_zero_scale_fails_closed(self):
        env=fixture();env.action_scales[0]=0
        with self.assertRaises(AssertionError):entry.transform(env,'policy_mean_clamped',torch.zeros(2,3))
    def test_nonfinite_fails_closed(self):
        env=fixture();a=torch.zeros(2,3);a[0,0]=float('nan')
        with self.assertRaises(AssertionError):entry.transform(env,'policy_mean_clamped',a)
    def test_random_batches_respect_limits(self):
        env=fixture();torch.manual_seed(754)
        for _ in range(100):
            a=torch.randn(2,3)*100;q=target(env,entry.transform(env,'policy_mean_clamped',a));lim=env.simulator.hard_dof_pos_limits
            self.assertTrue(((q>=lim[:,0])&(q<=lim[:,1])).all())
    def test_bounds(self):
        self.assertEqual((entry.NUM_ENVS,entry.STEPS_PER_MODE,entry.WORKER_SECONDS),(32,64,600))
        self.assertEqual(entry.MODES,('policy_mean','policy_mean_clamped','reference_pd'))
    def test_bootstrap_stdlib_only(self):
        flux=Path(__file__).resolve().parents[1]
        code=f'import sys;sys.path.insert(0,{str(flux)!r});import diagnose_target_clamp as e;e.configure();assert "torch" not in sys.modules;assert "typing_extensions" not in sys.modules;assert "control_chain_recorder" not in sys.modules;assert len(e.base.MODES)==3'
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr[-2000:])
    def test_no_optimizer_or_training(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'diagnose_target_clamp.py').read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ['learn','backward','zero_grad','save_checkpoint']:self.assertNotIn(forbidden,attrs)

if __name__=='__main__':unittest.main(verbosity=2)
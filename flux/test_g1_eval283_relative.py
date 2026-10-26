"""CPU regression tests execute the pinned native math without importing IsaacSim."""
import ast,copy,unittest,types
from pathlib import Path
import torch
from flux.g1_eval283_protocol import relative_refresh_source

class RelativeTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        roots=[Path(__file__).resolve().parents[1]/'thirdparty/holosoma/src/holosoma',
               Path('/workspace/php_g1/thirdparty/holosoma/src/holosoma')]
        root=next((r for r in roots if (r/'holosoma/managers/command/terms/wbt.py').exists()),None)
        if root is None:raise unittest.SkipTest('Pinned native submodule required for native CPU math tests')
        cls.source=(root/'holosoma/managers/command/terms/wbt.py').read_text()
        tree=ast.parse((root/'holosoma/utils/rotations.py').read_text())
        wanted={'yaw_quat','quat_apply','quat_conjugate','quat_inverse','quat_mul'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        assert len(nodes)==5
        for n in nodes:n.decorator_list=[]
        ns={'torch':torch,'Tensor':torch.Tensor,'normalize':lambda x:torch.nn.functional.normalize(x,dim=-1)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'native_cpu_quaternions','exec'),ns)
        exec(compile(relative_refresh_source(cls.source),'native_cpu_relative_refresh','exec'),ns)
        cls.refresh=staticmethod(ns['refresh_relative_targets'])

    def command(self):
        c=types.SimpleNamespace()
        c._env=types.SimpleNamespace(episode_length_buf=torch.ones(2,dtype=torch.long))
        c.motion_end_reset=torch.zeros(2,dtype=torch.bool)
        c.motion_cfg=types.SimpleNamespace(body_names_to_track=['one','two'])
        c.time_steps=torch.tensor([0,400]);c.motion_ids=torch.tensor([0,1])
        c.ref_pos_w=torch.tensor([[300.,400.,1.],[500.,600.,1.]])
        c.robot_ref_pos_w=c.ref_pos_w+torch.tensor([.03,-.02,.01])
        c.robot_root_pos_w=c.robot_ref_pos_w.clone()
        c.ref_quat_w=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1)
        c.robot_ref_quat_w=c.ref_quat_w.clone();c.robot_root_quat_w=c.ref_quat_w.clone()
        c.body_pos_w=c.ref_pos_w[:,None,:]+torch.tensor([[[.1,0.,.2],[-.1,0.,-.2]]])
        c.body_quat_w=c.ref_quat_w[:,None,:].repeat(1,2,1)
        c.body_pos_relative_w=torch.zeros(2,2,3);c.body_quat_relative_w=torch.zeros(2,2,4)
        return c

    def test_first_warmup_stale_hundreds_of_meters_fixed(self):
        c=self.command();robot_body=c.body_pos_w+torch.tensor([.03,-.02,.01])
        self.assertGreater(float(torch.linalg.vector_norm(c.body_pos_relative_w-robot_body,dim=-1).max()),400)
        self.refresh(c)
        self.assertLess(float(torch.linalg.vector_norm(c.body_pos_relative_w-robot_body,dim=-1).max()),.011)

    def test_reference_time_motion_identity_and_robot_state_not_mutated(self):
        c=self.command();before=copy.deepcopy(c)
        self.refresh(c)
        for k in ['time_steps','motion_ids','robot_ref_pos_w','robot_root_pos_w','ref_pos_w','body_pos_w']:
            self.assertTrue(torch.equal(getattr(c,k),getattr(before,k)),k)
        self.assertTrue(torch.equal(c._env.episode_length_buf,before._env.episode_length_buf))

    def test_global_translation_equivariance(self):
        a=self.command();b=copy.deepcopy(a);shift=torch.tensor([123.,-456.,0.])
        for k in ['ref_pos_w','robot_ref_pos_w','robot_root_pos_w','body_pos_w']:setattr(b,k,getattr(b,k)+shift)
        self.refresh(a);self.refresh(b)
        self.assertTrue(torch.allclose(b.body_pos_relative_w,a.body_pos_relative_w+shift,atol=1e-4))

    def test_repeated_refresh_idempotent_without_reference_advance(self):
        c=self.command();self.refresh(c);a=c.body_pos_relative_w.clone();q=c.body_quat_relative_w.clone()
        self.refresh(c);self.assertTrue(torch.equal(a,c.body_pos_relative_w));self.assertTrue(torch.equal(q,c.body_quat_relative_w))

    def test_schema_drift_rejected(self):
        with self.assertRaises((ValueError,StopIteration)):
            relative_refresh_source(self.source.replace('self.body_pos_relative_w = (','self.other_pose = ('))

if __name__=='__main__':unittest.main()
"""Execute the pinned native proxy/converters on CPU, not a proxy-shaped mock.

AST loading avoids importing the simulator/IsaacLab package; class/function bodies
are taken intact from hash-checked Holosoma 4a0bf04. Physics itself remains mocked.
"""
import ast,dataclasses,hashlib,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
import paired_evaluation_tools as tools
import low_noise_tools
import test_paired_evaluation as previous

PIN='4a0bf04cde5930af5ed59da797568d32be972525'
HASHES={'proxy_utils.py':'46ba8453529561c8ef4d1d7d5447b0d651e6588d0e5e4bb21094757bf01ce52d',
        'state_utils.py':'a226831bbdf0d43e3eb5711f52412fe602c41c2f1cf1117a04bee9ea43896293'}

def native_proxy_class():
    candidates=[ROOT/'thirdparty/holosoma/src/holosoma/holosoma/simulator/isaacsim',
        Path('/workspace/task047/proxy-contract/src/holosoma/holosoma/simulator/isaacsim')]
    directory=next((p for p in candidates if all((p/n).is_file() for n in HASHES)),None)
    if directory is None:raise FileNotFoundError('Initialize the pinned Holosoma submodule for proxy contract tests')
    namespace={'torch':torch}
    for name in ['state_utils.py','proxy_utils.py']:
        path=directory/name;data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=HASHES[name]:raise ValueError('Pinned native source changed: '+name)
        parsed=ast.parse(data)
        nodes=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',
            flags=__import__('__future__').annotations.compiler_flag),namespace)
    return namespace['RootStatesProxy']

@dataclasses.dataclass
class MotionConfig:
    use_adaptive_timesteps_sampler:bool=True

class NativeProxyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.Proxy=native_proxy_class()
    def state(self,n=32,width=13):
        state=torch.arange(n*width,dtype=torch.float32).reshape(n,width)/100.
        state[:,3:7]=torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
        return state
    def capture(self,directory,state=None):
        capture,env,motion,stats=previous.CaptureTests().make(Path(directory)/'steps.jsonl')
        backing=self.state() if state is None else state
        env.simulator.robot_root_states=self.Proxy(backing)
        self.addCleanup(capture.sink.close)
        return capture,env,motion,stats,backing
    def test_original_099_expression_reproduces_real_attribute_error(self):
        proxy=self.Proxy(self.state())
        self.assertFalse(torch.is_tensor(proxy))
        with self.assertRaisesRegex(AttributeError,"has no attribute 'detach'"):
            proxy.detach().cpu().tolist()
    def test_native_index_is_a_tensor_in_xyzw_order(self):
        backing=self.state();proxy=self.Proxy(backing)
        expected=backing.clone();expected[:,3:7]=backing[:,[4,5,6,3]]
        self.assertTrue(torch.is_tensor(proxy[:]))
        self.assertTrue(torch.equal(proxy[:],expected))
        self.assertFalse(torch.equal(proxy[:],backing))
    def test_first_reset_accepts_real_native_proxy(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,motion,stats,backing=self.capture(d)
            obs=env.reset_all()
            expected=backing.clone();expected[:,3:7]=backing[:,[4,5,6,3]]
            self.assertEqual(capture.initial['root_states'],expected.tolist())
            self.assertEqual(tuple(torch.tensor(capture.initial['root_states']).shape),(32,13))
            self.assertEqual(capture.steps,0)
            self.assertEqual(obs['actor_obs'].shape,(32,4))
    def test_snapshot_does_not_modify_backing_or_proxy(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,motion,stats,backing=self.capture(d)
            before=backing.clone();proxy_before=env.simulator.robot_root_states[:].clone()
            env.reset_all()
            self.assertTrue(torch.equal(backing,before))
            self.assertTrue(torch.equal(env.simulator.robot_root_states[:],proxy_before))
    def test_snapshot_is_independent_of_later_proxy_mutations(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,motion,stats,backing=self.capture(d);env.reset_all()
            saved=capture.initial['root_states'][0][0]
            env.simulator.robot_root_states.tensor_xyzw[0,0]=123.
            self.assertEqual(capture.initial['root_states'][0][0],saved)
    def test_second_reset_reads_refreshed_native_proxy(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,motion,stats,backing=self.capture(d);env.reset_all()
            refreshed=self.state();refreshed[:,0]+=3.
            env.simulator.robot_root_states.reset(refreshed);env.reset_all()
            self.assertEqual(capture.initial['root_states'][0][0],3.)
            self.assertEqual(capture.steps,0)
    def test_nonfinite_root_states_fail_closed(self):
        state=self.state();state[0,0]=float('nan')
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,*_=self.capture(d,state)
            with self.assertRaisesRegex(ValueError,'Nonfinite root'):env.reset_all()
    def test_wrong_root_state_batch_fails_closed(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,*_=self.capture(d,self.state(n=31))
            with self.assertRaisesRegex(ValueError,'Root-state shape'):env.reset_all()
    def test_wrong_root_state_width_fails_closed(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,*_=self.capture(d,self.state(width=12))
            with self.assertRaisesRegex(ValueError,'Root-state shape'):env.reset_all()
    def test_no_private_wxyz_storage_or_proxy_detach_in_snapshot(self):
        tree=ast.parse((ROOT/'flux/paired_evaluation_tools.py').read_text())
        attributes=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('tensor_wxyz',attributes);self.assertNotIn('_get_wxyz',attributes)
    def test_128_step_evaluation_control_flow_with_real_proxy(self):
        # The proxy and conversion code are real; the simulator dynamics below are not.
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            capture,env,motion,stats,backing=self.capture(d)
            env.is_evaluating=False;env.num_envs=32;env.device='cpu'
            motion.motion_cfg=MotionConfig()
            stats.clear=lambda:None
            stats.summary=lambda:dict(environment_steps=capture.samples,mean_reward=0.,reward_terms={},termination_flags={})
            def step(actor_state):
                if stats.active and capture.steps>=9:motion.body_pos_relative_w[:,0,2]=.6
                env._check_termination()
                return dict(actor_obs=torch.zeros(32,4)),torch.zeros(32),env.reset_buf.clone(),{}
            env.step=step
            algo=NS(env=env,device='cpu',actor_obs_keys=['actor_obs'],_eval_mode=lambda:None,
                _normalize_actor_obs=lambda raw,update=False:raw,
                actor=NS(act_inference=lambda obs:torch.zeros(32,29)))
            result=low_noise_tools.evaluate(algo,motion,stats,seed=1402)
            diagnostics=capture.finish(result)
            self.assertEqual(result['first_episode_steps'],[10]*32)
            self.assertEqual(result['survivors_at_horizon'],0)
            self.assertEqual(diagnostics['first_episode_samples'],320)
            self.assertEqual(len(diagnostics['first_failures']),32)
            self.assertEqual(capture.steps,128)
            self.assertTrue(all(x['step']==10 and x['failed_bodies']==['left_ankle'] for x in diagnostics['first_failures']))
            self.assertFalse(stats.active)

if __name__=='__main__':unittest.main()
"""CPU regression against pinned native PPO save/load, plus fresh-run safety contracts."""
import ast,contextlib,copy,sys,tempfile,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
import checkpoint_contract as contract
import train_fresh_pilot as fresh

def native_method(path,cls,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns=dict(torch=torch,logger=NS(info=lambda *a:None))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec',
        flags=__import__('__future__').annotations.compiler_flag),ns)
    return ns[name]

class Actor(torch.nn.Linear):
    def __init__(self):super().__init__(2,2);self.std=torch.nn.Parameter(torch.ones(2))
    def project_action_std(self):pass

class FreshTests(unittest.TestCase):
    def make_algo(self,trained=False):
        a=NS(actor=Actor(),critic=torch.nn.Linear(2,1),current_learning_iteration=3 if trained else 0,
            empirical_normalization=False,device='cpu',config=NS(load_optimizer=True),
            _experiment_config=None,_wandb_run_path=None)
        a.actor_optimizer=torch.optim.AdamW(a.actor.parameters(),lr=.001)
        a.critic_optimizer=torch.optim.AdamW(a.critic.parameters(),lr=.001)
        if trained:
            for module,opt in [(a.actor,a.actor_optimizer),(a.critic,a.critic_optimizer)]:
                sum(p.sum() for p in module.parameters()).backward();opt.step();opt.zero_grad()
        base=ROOT/'thirdparty/holosoma/src/holosoma/holosoma/agents/base_algo/base_algo.py'
        ppo=ROOT/'thirdparty/holosoma/src/holosoma/holosoma/agents/ppo/ppo.py'
        for path,cls,names in [(base,'BaseAlgo',['attach_checkpoint_metadata','_checkpoint_metadata']),
                               (ppo,'PPO',['save','load'])]:
            for name in names:setattr(a,name,types.MethodType(native_method(path,cls,name),a))
        a._collect_env_state=lambda:dict(probe=1)
        a._restore_env_state=lambda state:None
        a.logging_helper=NS(save_checkpoint_artifact=lambda data,path:torch.save(data,path))
        return a
    def config(self):return NS(to_serializable_dict=lambda:dict(training=dict(checkpoint=None),robot='hu_d04'))
    def test_original_missing_metadata_failure_is_reproduced(self):
        a=self.make_algo()
        with tempfile.TemporaryDirectory(dir='/workspace') as td:
            path=Path(td)/'never_written.pt'
            with self.assertRaisesRegex(RuntimeError,'metadata missing'):a.save(str(path))
            self.assertFalse(path.exists())
    def test_native_metadata_attached_and_embedded(self):
        a=self.make_algo();cfg=self.config();contract.attach_metadata(a,cfg,'team/project/run')
        self.assertIs(a._experiment_config,cfg)
        self.assertEqual(a._checkpoint_metadata(0)['wandb_run_path'],'team/project/run')
    def test_native_fresh_roundtrip(self):
        a=self.make_algo();contract.attach_metadata(a,self.config())
        with tempfile.TemporaryDirectory(dir='/workspace') as td:
            r=contract.roundtrip(a,Path(td)/'fresh.pt')
            self.assertTrue(r['native_save_reload_verified']);self.assertEqual(r['iteration'],0)
            self.assertFalse(a.actor_optimizer.state)
    def test_native_trained_optimizer_roundtrip(self):
        a=self.make_algo(True);contract.attach_metadata(a,self.config())
        with tempfile.TemporaryDirectory(dir='/workspace') as td:
            r=contract.roundtrip(a,Path(td)/'trained.pt')
            self.assertTrue(r['actor_critic_optimizer_iteration_exact']);self.assertTrue(a.actor_optimizer.state)
    def test_corrupted_reload_stops_trial(self):
        a=self.make_algo();contract.attach_metadata(a,self.config());load=a.load
        def broken(path):
            load(path)
            with torch.no_grad():a.actor.weight.add_(1)
        a.load=broken
        with tempfile.TemporaryDirectory(dir='/workspace') as td:
            with self.assertRaisesRegex(ValueError,'changed'):contract.roundtrip(a,Path(td)/'probe.pt')
    def test_never_overwrites_output(self):
        a=self.make_algo();contract.attach_metadata(a,self.config())
        with tempfile.TemporaryDirectory(dir='/workspace') as td:
            path=Path(td)/'exists.pt';path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):contract.roundtrip(a,path)
            self.assertEqual(path.read_bytes(),b'preserve')
    def test_unsupported_metadata_api_rejected(self):
        with self.assertRaises(TypeError):contract.attach_metadata(NS(),self.config())
    def test_old_resume_entrypoint_attaches_before_load(self):
        source=(ROOT/'flux/train_low_noise.py').read_text()
        self.assertLess(source.index('algo.attach_checkpoint_metadata'),source.index('algo.load(str(checkpoint))'))
    def test_fresh_only_loads_its_own_generated_files(self):
        tree=ast.parse(Path(fresh.__file__).read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
        self.assertEqual(len(calls),1)
        self.assertIn('model_preflight.pt',ast.unparse(calls[0]))
        self.assertNotIn('load_saved_experiment_config',ast.unparse(worker))
    def test_preflight_before_updates_and_fresh_metadata(self):
        source=Path(fresh.__file__).read_text()
        self.assertLess(source.index('preflight=roundtrip'),source.index('trained=train'))
        self.assertIn('checkpoint=None,teacher_checkpoint=None,wandb_path=None',source)
        self.assertIn('algo.current_learning_iteration=1',source)
        self.assertEqual((fresh.WORKER_SECONDS,fresh.MAX_UPDATES,fresh.TRAIN_SECONDS),(1800,200,300))
    def test_tracking_reads_native_targets_without_mutating_them(self):
        target=torch.zeros(32,29);target[:,0]=2
        motion=NS(motion_cfg=NS(body_names_to_track=['left_ankle_roll_link']),
            body_pos_relative_w=torch.ones(32,1,3),robot_body_pos_w=torch.zeros(32,1,3),joint_pos=torch.zeros(32,29))
        env=NS(robot_config=NS(dof_names=['left_hip_roll_joint']+[str(i) for i in range(28)]),
            action_manager=NS(_term_instances={'position':NS(_applied_position_targets=target)}),
            simulator=NS(_robot=NS(data=NS(joint_pos_target=target)),dof_ids=list(range(29)),
                hard_dof_pos_limits=torch.tensor([[-1.,1.]]*29)),reset_buf=torch.zeros(32,dtype=torch.bool),
            _check_termination=lambda:17)
        original=target.clone()
        with tempfile.TemporaryDirectory(dir='/workspace') as td:
            path=Path(td)/'tracking.jsonl';capture=fresh.TrackingCapture(env,motion,path)
            capture.set_phase('training');self.assertEqual(env._check_termination(),17);capture.close()
            row=__import__('json').loads(path.read_text())
            self.assertEqual(row['left_hip_target_outside_fraction'],1.)
            self.assertEqual(row['left_ankle_exceeded_count'],32)
            self.assertTrue(torch.equal(target,original))

if __name__=='__main__':unittest.main()
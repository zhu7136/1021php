"""CPU contracts for frozen paired evaluation; no claim of GPU simulation coverage."""
import ast,copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
import evaluate_protected_multiseed as entry
import paired_evaluation_tools as tools

class BadTracking:
    bad_motion_body_pos_body_names=['left_ankle','right_ankle']
    bad_motion_body_pos_body_indexes=torch.tensor([0,1])
    bad_motion_body_pos_threshold=.5
    bad_ref_pos_threshold=.5
    bad_ref_ori_threshold=.8
    def bad_ref_pos(self,m):return torch.linalg.vector_norm(m.ref_pos_w-m.robot_ref_pos_w,dim=-1)>.5
    def bad_ref_ori(self,m):return torch.zeros(32,dtype=torch.bool)

class PairTests(unittest.TestCase):
    def values(self):
        ev=dict(seed=1402,horizon_steps=128,dt=.02,initial_joint_positions=[[0.]*29]*32,
            initial_motion_frames=list(range(32)),first_episode_steps=[5]*32,
            survivors_at_horizon=0,restricted_mean_survival_seconds=.1)
        return ev,dict(joint_positions=[[0.]*29]*32,motion_frames=list(range(32)))
    def test_three_declared_seeds_six_fresh_processes(self):
        self.assertEqual(entry.SEEDS,tools.SEEDS)
        self.assertEqual(len(entry.ARMS),6)
        self.assertEqual(len(set(k for k,_ in entry.ARMS)),6)
        self.assertEqual(entry.WORKER_SECONDS,1800)
        self.assertEqual(entry.INPUT_HASHES,tools.INPUTS)
        self.assertEqual(entry.BUNDLE_SHA,tools.BUNDLE_SHA)
    def test_exact_pair(self):
        b,s=self.values();a=copy.deepcopy(b);a['first_episode_steps']=[7]*30+[5,3]
        r=tools.compare_pair(b,a,s,s)
        self.assertEqual((r['improved'],r['unchanged'],r['worse']),(30,1,1))
        self.assertTrue(r['initial_state_exact'])
    def test_mismatched_seed_rejected(self):
        b,s=self.values();a=copy.deepcopy(b);a['seed']=2402
        with self.assertRaisesRegex(ValueError,'Seed'):tools.compare_pair(b,a,s,s)
    def test_mismatched_initial_state_rejected(self):
        b,s=self.values();other=copy.deepcopy(s);other['motion_frames'][0]=99
        with self.assertRaisesRegex(ValueError,'not exactly paired'):tools.compare_pair(b,b,s,other)
    def test_mismatched_state_shape_rejected(self):
        b,s=self.values();other=copy.deepcopy(s);other['motion_frames']=[0]
        with self.assertRaisesRegex(ValueError,'Invalid initial'):tools.compare_pair(b,b,s,other)
    def test_nonfinite_state_rejected(self):
        b,s=self.values();other=copy.deepcopy(s);other['joint_positions'][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid initial'):tools.compare_pair(b,b,s,other)
    def test_horizon_mismatch_rejected(self):
        b,s=self.values();a=copy.deepcopy(b);a['horizon_steps']=129
        with self.assertRaisesRegex(ValueError,'Horizon'):tools.compare_pair(b,a,s,s)
    def test_input_hash_failure_precedes_extraction(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            p=Path(d)/'bad.zip';p.write_bytes(b'bad');target=Path(d)/'inputs'
            with self.assertRaisesRegex(ValueError,'hash/size'):entry.extract_inputs(p,target)
            self.assertFalse(target.exists())
    def test_arbitrary_input_network_rejected(self):
        for url in ['http://storage.googleapis.com/a','https://example.org/a','https://user@storage.googleapis.com/a']:
            with self.assertRaisesRegex(ValueError,'Only scoped'):entry.fetch_inputs(url)
    def test_no_training_or_checkpoint_output_calls(self):
        tree=ast.parse((ROOT/'flux/evaluate_protected_multiseed.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('learn',calls);self.assertNotIn('backward',calls);self.assertNotIn('save',[
            n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id in ['algo','torch']])
        text=(ROOT/'flux/evaluate_protected_multiseed.py').read_text()
        self.assertIn('load_optimizer=False',text)
        self.assertIn('num_learning_iterations=0',text)
        self.assertIn('algo.actor_optimizer.step=forbid',text)
        self.assertIn('algo.critic_optimizer.step=forbid',text)

class CaptureTests(unittest.TestCase):
    def make(self,path):
        motion=NS(body_pos_relative_w=torch.zeros(32,2,3),robot_body_pos_w=torch.zeros(32,2,3),
            ref_pos_w=torch.zeros(32,3),robot_ref_pos_w=torch.zeros(32,3),motion=NS(has_object=False),
            time_steps=torch.arange(32))
        stats=NS(active=False,mask=torch.ones(32,dtype=torch.bool),terms={})
        action=NS(_requested_position_targets=torch.zeros(32,29),_applied_position_targets=torch.zeros(32,29))
        limits=torch.tensor([[-1.,1.]]*29)
        sim=NS(dof_pos=torch.zeros(32,29),dof_vel=torch.zeros(32,29),robot_root_states=torch.zeros(32,13),
            hard_dof_pos_limits=limits,dof_ids=list(range(29)),_robot=NS(data=NS(joint_pos_target=action._applied_position_targets)))
        env=NS(simulator=sim,dr_episode_count=torch.zeros(32,dtype=torch.long),dt=.02,reset_buf=torch.zeros(32,dtype=torch.long),
            termination_manager=NS(_term_instances={'bad_tracking':BadTracking()}),
            action_manager=NS(_term_instances={'action':action}))
        def original():
            t=env.termination_manager._term_instances['bad_tracking']
            body=(motion.body_pos_relative_w-motion.robot_body_pos_w).norm(dim=-1).gt(.5).any(-1)
            stats.terms={'bad_tracking':t.bad_ref_pos(motion)|t.bad_ref_ori(motion)|body,
                'timeout':torch.zeros(32,dtype=torch.bool)}
            env.reset_buf.copy_(stats.terms['bad_tracking']);return None
        env._check_termination=original
        env.reset_all=lambda:dict(actor_obs=torch.zeros(32,4))
        capture=tools.FirstEpisodeCapture(env,motion,stats,path)
        return capture,env,motion,stats
    def test_warmup_not_sampled(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(Path(d)/'steps.jsonl')
            m.body_pos_relative_w.fill_(100.)
            e._check_termination()
            self.assertEqual(c.steps,0);self.assertEqual(c.failures,[])
            c.sink.close()
    def test_failure_attribution_before_reset(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(Path(d)/'steps.jsonl');s.active=True
            m.body_pos_relative_w[3,0,2]=.6
            e._check_termination()
            self.assertEqual(len(c.failures),1)
            self.assertEqual(c.failures[0]['env'],3)
            self.assertEqual(c.failures[0]['failed_bodies'],['left_ankle'])
            self.assertEqual(c.failures[0]['seconds'],.02)
            c.sink.close()
    def test_dead_env_excluded(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(Path(d)/'steps.jsonl');s.active=True;s.mask[3]=False
            m.body_pos_relative_w[3,0,2]=100.
            e._check_termination()
            self.assertEqual(c.failures,[]);self.assertEqual(c.samples,31)
            c.sink.close()
            rows=[json.loads(x) for x in (Path(d)/'steps.jsonl').read_text().splitlines()]
            self.assertNotIn(3,rows[0]['active_envs'])
    def test_target_escape_fails_closed(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(Path(d)/'steps.jsonl');s.active=True
            e.action_manager._term_instances['action']._applied_position_targets[0,0]=2.
            with self.assertRaisesRegex(ValueError,'escaped hard'):e._check_termination()
            c.sink.close()
    def test_incomplete_capture_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(Path(d)/'steps.jsonl')
            with self.assertRaisesRegex(ValueError,'Incomplete'):c.finish({})
    def test_reset_snapshot_is_copied(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(Path(d)/'steps.jsonl');e.reset_all()
            e.simulator.dof_pos.fill_(2.)
            self.assertEqual(c.initial['joint_positions'][0][0],0.)
            c.sink.close()

if __name__=='__main__':unittest.main()
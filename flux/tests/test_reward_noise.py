import ast,io,json,subprocess,sys,tempfile,unittest,zipfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose_reward_noise as entry
from diagnostic_bundle import build_bundle
from reward_noise_recorder import Recorder
from test_control_chain import fixture

def compact_fixture():
    env,_,old=fixture();old.uninstall()
    env.time_out_buf=torch.tensor([False,False])
    sink=io.StringIO();recorder=Recorder(env,old.motion,sink);recorder.install()
    return env,sink,recorder

class RewardNoiseTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual((entry.NUM_ENVS,entry.STEPS_PER_MODE,entry.WORKER_SECONDS),(32,64,600))
        self.assertEqual(entry.MODES,('policy_mean','policy_sample'))
    def test_bootstrap_stdlib_only(self):
        flux=Path(__file__).resolve().parents[1]
        code=f'import sys;sys.path.insert(0,{str(flux)!r});import diagnose_reward_noise as e;e.configure();assert "torch" not in sys.modules;assert "reward_noise_recorder" not in sys.modules;assert e.base.WORKER_SECONDS==600;assert e.base.MODES==e.MODES'
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr[-2000:])
    def test_all_steps_configured(self):
        source=Path(entry.__file__).read_text()
        self.assertIn('capture_steps=STEPS_PER_MODE',source)
        self.assertIn('step<capture_steps',Path(entry.base.__file__).read_text())
    def test_mean_and_sample_use_distinct_native_paths(self):
        source=Path(entry.base.__file__).read_text()
        self.assertIn("else algo.actor.act({'actor_obs':normalized})",source)
        self.assertNotIn('policy_sample',source.split('if mode in (',1)[1].split(') else',1)[0])
    def test_no_learning_no_reward_edits(self):
        for name in ('diagnose_reward_noise.py','reward_noise_recorder.py','diagnostic_bundle.py'):
            tree=ast.parse((Path(entry.__file__).parent/name).read_text())
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            for forbidden in ('learn','backward','zero_grad','step_optimizer','save_checkpoint'):
                self.assertNotIn(forbidden,calls)
        self.assertNotIn('action_transform=',Path(entry.__file__).read_text())
    def test_pre_reset_actions_and_rewards(self):
        env,sink,r=compact_fixture()
        obs=torch.tensor([[.1,.2,5.],[.3,.4,6.]])
        mean=torch.ones(2,2)*.4;sample=mean+.8
        r.begin_step('policy_sample',0,obs,mean,sample)
        obs.fill_(999);sample.fill_(999)
        env._check_termination();env._compute_reward()
        env.simulator.dof_pos.fill_(999)
        row=json.loads(sink.getvalue())
        self.assertEqual(row['capture_point'],'post_physics_pre_reset')
        self.assertAlmostEqual(row['values']['actions'][0][0],1.2,places=6)
        self.assertAlmostEqual(row['values']['actor_mean'][0][0],.4,places=6)
        self.assertAlmostEqual(row['values']['previous_actions'][1][0],.3,places=6)
        self.assertEqual(row['values']['q_after'],[[0,0],[0,0]])
        self.assertEqual(row['values']['done'],[True,False])
        self.assertEqual(row['values']['timeout'],[False,False])
        self.assertAlmostEqual(sum(v[0] for v in row['reward_contributions'].values()),-.1,places=6)
        self.assertEqual(env.calls,2);r.uninstall()
    def test_episode_ids_and_phase_restart(self):
        env,sink,r=compact_fixture();a=torch.zeros(2,2)
        for i,phase in enumerate(('policy_mean','policy_mean','policy_sample')):
            r.begin_step(phase,i,torch.ones(2,5),a,a)
            env._check_termination();env._compute_reward()
            self.assertIsNone(r.phase);self.assertIsNone(r.pending)
        rows=[json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertEqual([row['values']['episode_ids'] for row in rows],[[0,0],[1,0],[0,0]])
    def test_inactive_passthrough(self):
        env,sink,r=compact_fixture();env._check_termination();env._compute_reward()
        self.assertEqual(env.calls,2);self.assertEqual(sink.getvalue(),'');r.uninstall()
    def test_target_mismatch(self):
        env,sink,r=compact_fixture();a=torch.zeros(2,2)
        r.begin_step('policy_mean',0,torch.ones(2,5),a,a)
        env.simulator._robot.data.joint_pos_target.fill_(99)
        with self.assertRaisesRegex(AssertionError,'Native target'):env._check_termination()
    def test_reward_mismatch(self):
        env,sink,r=compact_fixture();a=torch.zeros(2,2)
        r.original_reward=lambda:None
        r.begin_step('policy_mean',0,torch.ones(2,5),a,a)
        env._check_termination();env.rew_buf.fill_(42)
        with self.assertRaisesRegex(AssertionError,'contribution mismatch'):env._compute_reward()
    def test_nonfinite_rejected(self):
        env,sink,r=compact_fixture();a=torch.zeros(2,2);a[0,0]=float('nan')
        with self.assertRaises(AssertionError):r.begin_step('policy_sample',0,torch.ones(2,5),a,a)

class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/workspace');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/'out';self.out.mkdir();self.zip=self.root/'bundle.zip'
    def test_over_old_raw_cap_compresses_and_roundtrips(self):
        # Regress the old40MiB raw-size gate without any outbound upload.
        payload=b'0'*(41*1024*1024)
        (self.out/'telemetry.jsonl').write_bytes(payload)
        manifest=build_bundle(self.out,self.zip)
        self.assertGreater(manifest['raw_bytes'],40*1024*1024)
        self.assertLess(manifest['compressed_bytes'],32*1024*1024)
        with zipfile.ZipFile(self.zip) as bundle:self.assertEqual(bundle.read('telemetry.jsonl'),payload)
    def test_excludes_inputs_and_subdirs(self):
        (self.out/'summary.json').write_text('{}');(self.out/'input.zip').write_bytes(b'secret_input')
        (self.out/'dataset').mkdir();(self.out/'dataset'/'large.json').write_text('{}')
        result=build_bundle(self.out,self.zip)
        self.assertEqual(result['files'],['summary.json'])
    def test_raw_cap_enforced(self):
        (self.out/'x.json').write_text('123456')
        with self.assertRaisesRegex(AssertionError,'raw-data budget'):build_bundle(self.out,self.zip,max_raw=5)
        self.assertFalse(self.zip.exists())
    def test_compressed_cap_enforced(self):
        (self.out/'x.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError,'compressed-transfer budget'):build_bundle(self.out,self.zip,max_bundle=1)
    def test_symlink_rejected(self):
        (self.root/'real').write_text('{}');(self.out/'x.json').symlink_to(self.root/'real')
        with self.assertRaisesRegex(AssertionError,'Unsafe'):build_bundle(self.out,self.zip)
    def test_empty_rejected(self):
        with self.assertRaises(AssertionError):build_bundle(self.out,self.zip)
    def test_no_existing_bundle_overwrite(self):
        (self.out/'x.json').write_text('{}');self.zip.write_bytes(b'original')
        with self.assertRaisesRegex(AssertionError,'Fresh bundle'):build_bundle(self.out,self.zip)
        self.assertEqual(self.zip.read_bytes(),b'original')

if __name__=='__main__':unittest.main(verbosity=2)
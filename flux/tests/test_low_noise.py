import dataclasses,importlib.util,itertools,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch.distributions import Normal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_low_noise as entry
from low_noise_tools import set_initial_noise,distribution_metrics,stop_reason,EnvironmentStats,evaluate,train

class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.std=torch.nn.Parameter(torch.full((29,),.8));self.mu=torch.nn.Parameter(torch.zeros(29))
        self.min_noise_std=self.min_mean_noise_std=0.;self.distribution=None
    def project_action_std(self):pass
    def act_inference(self,obs):return self.mu.expand(len(obs['actor_obs']),29)
    def act(self,obs):
        self.distribution=Normal(self.act_inference(obs),self.std.expand(len(obs['actor_obs']),29));return self.distribution.sample()
    def get_actions_log_prob(self,actions):return self.distribution.log_prob(actions).sum(-1)
    @property
    def action_mean(self):return self.distribution.mean
    @property
    def action_std(self):return self.distribution.stddev
    @property
    def entropy(self):return self.distribution.entropy().sum(-1)

@dataclasses.dataclass(frozen=True)
class MotionCfg:use_adaptive_timesteps_sampler:bool=True

class FakeEnv:
    def __init__(self):
        self.num_envs=32;self.device='cpu';self.dt=.02;self.stepno=0;self.is_evaluating=False
        self.motion=NS(motion_cfg=MotionCfg(),time_steps=torch.arange(32))
        self.simulator=NS(dof_pos=torch.zeros(32,29))
        self.reward_manager=NS(_episode_sums={'tracking':torch.zeros(32),'cost':torch.zeros(32)},
                              cfg=NS(terms={'tracking':NS(weight=1),'cost':NS(weight=-1)}))
        self.termination_manager=NS(_term_funcs={'failure':lambda env:env.stepno>=torch.tensor([3]*8+[8]*8+[128]*8+[129]*8)},_term_instances={})
    def reset_all(self):
        self.stepno=0
        for v in self.reward_manager._episode_sums.values():v.zero_()
        return {'actor_obs':torch.zeros(32,1)}
    def _check_termination(self):
        self.reset_buf=self.termination_manager._term_funcs['failure'](self)
    def _compute_reward(self):
        self.reward_manager._episode_sums['tracking']+=.1
        self.reward_manager._episode_sums['cost']-=.02
        self.rew_buf=torch.full((32,),.08)
    def step(self,data):
        self.stepno+=1;self._check_termination();self._compute_reward()
        return {'actor_obs':torch.zeros(32,1)},self.rew_buf,self.reset_buf,{}

class FakeAlgo:
    def __init__(self):
        self.env=FakeEnv();self.actor=Actor();self.actor_obs_keys=['actor_obs'];self.device='cpu'
        self.actor_learning_rate=self.critic_learning_rate=1e-5
        self.config=NS(clip_param=.2);self.current_learning_iteration=791;self.native_logs=0;self.native_steps=0
        self.actor_optimizer=NS(param_groups=[dict(lr=1e-5,params=list(self.actor.parameters()))],step=self.opt_step)
        self.critic_optimizer=NS(param_groups=[dict(lr=1e-5,params=[torch.nn.Parameter(torch.zeros(1))])],step=self.opt_step)
    def opt_step(self):self.native_steps+=1
    def _eval_mode(self):self.actor.eval()
    def _normalize_actor_obs(self,x,update=False):return x
    def _compute_ppo_loss(self,batch):return {'Value':torch.tensor(0.)}
    def _post_epoch_logging(self,it,losses):self.native_logs+=1
    def learn(self):
        for it in range(791,1100):
            self.current_learning_iteration=it
            for i in range(20):
                actions=self.actor.act({'actor_obs':torch.zeros(2,1)})
                batch=dict(actions=actions,action_mean=self.actor.action_mean.detach().clone(),
                           action_sigma=self.actor.action_std.detach().clone(),actions_log_prob=self.actor.get_actions_log_prob(actions).detach().clone())
                losses=self._compute_ppo_loss(batch);self.actor_optimizer.step();self.critic_optimizer.step()
            self._post_epoch_logging(it,losses)

class LowNoiseTests(unittest.TestCase):
    def test_budget_constants(self):
        self.assertEqual((entry.WORKER_SECONDS,entry.ARM_SECONDS),(1800,825))
        self.assertEqual(entry.ARMS,(('resume_control',1.),('resume_low_noise',.25)))
    def test_bootstrap_stdlib_only(self):
        code=f'import sys;sys.path.insert(0,{str(Path(entry.__file__).parent)!r});import train_low_noise;assert "torch" not in sys.modules'
        p=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_stop_iteration_boundaries(self):
        self.assertIsNone(stop_reason(199,299.99));self.assertEqual(stop_reason(200,0),'iteration_budget')
        self.assertEqual(stop_reason(1,300),'time_budget')
    def test_quarter_std_and_mean_unchanged(self):
        a=Actor();before=a.mu.clone();info=set_initial_noise(a,.25,completed_rollouts=0)
        self.assertTrue(torch.equal(a.mu,before));self.assertAlmostEqual(info['new_std_mean'],.2,places=6)
    def test_control_identity(self):
        a=Actor();before={k:v.clone() for k,v in a.state_dict().items()};set_initial_noise(a,1.,completed_rollouts=0)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in a.state_dict().items()))
    def test_no_stale_distribution(self):
        a=Actor();a.act({'actor_obs':torch.zeros(2,1)})
        with self.assertRaises(ValueError):set_initial_noise(a,.25,completed_rollouts=0)
    def test_no_old_rollout(self):
        with self.assertRaises(ValueError):set_initial_noise(Actor(),.25,completed_rollouts=1)
    def test_no_repeated_intervention(self):
        a=Actor();set_initial_noise(a,.25,completed_rollouts=0)
        with self.assertRaises(ValueError):set_initial_noise(a,.25,completed_rollouts=0)
    def test_first_episode_survival_not_completed_mean(self):
        a=FakeAlgo();stats=EnvironmentStats(a.env);result=evaluate(a,a.env.motion,stats)
        self.assertEqual(result['first_episode_steps'],[3]*8+[8]*8+[128]*16)
        self.assertEqual(result['survivors_at_horizon'],8)
        self.assertAlmostEqual(result['restricted_mean_survival_seconds'],1.335)
        self.assertAlmostEqual(result['tracking_positive_mean_reward'],.1,places=5)
        self.assertEqual(result['termination_flags']['failure'],24)
    def test_evaluation_restores_sampler_and_disables_capture(self):
        a=FakeAlgo();stats=EnvironmentStats(a.env);old=a.env.motion.motion_cfg;evaluate(a,a.env.motion,stats)
        self.assertIs(a.env.motion.motion_cfg,old);self.assertFalse(stats.active);self.assertIsNone(stats.mask)
        self.assertFalse(a.env.is_evaluating)
    def test_evaluation_is_deterministic_and_never_creates_sampling_distribution(self):
        a=FakeAlgo();stats=EnvironmentStats(a.env)
        self.assertEqual(evaluate(a,a.env.motion,stats),evaluate(a,a.env.motion,stats))
        self.assertIsNone(a.actor.distribution)
    def test_native_train_iteration_cap_and_telemetry(self):
        a=FakeAlgo();stats=EnvironmentStats(a.env);messages=[];original=a._compute_ppo_loss
        with tempfile.TemporaryDirectory(dir='/workspace') as tmp:
            result=train(a,stats,tmp,lambda **kw:messages.append(kw),clock=lambda:0)
            self.assertEqual(result['completed_updates'],200);self.assertEqual(result['last_iteration'],990)
            self.assertEqual(result['minibatches'],4000);self.assertEqual(a.native_steps,8000)
            self.assertEqual(result['transitions'],153600);self.assertEqual(len(messages),200)
            rows=[json.loads(x) for x in (Path(tmp)/'minibatches.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4000);self.assertTrue(all(r['kl']==0 and r['clip_fraction']==0 for r in rows))
        self.assertEqual(a._compute_ppo_loss,original);self.assertFalse(stats.active)
    def test_native_train_time_cap(self):
        a=FakeAlgo();stats=EnvironmentStats(a.env);ticks=itertools.count()
        with tempfile.TemporaryDirectory(dir='/workspace') as tmp:
            result=train(a,stats,tmp,lambda **kw:None,clock=lambda:next(ticks)*150)
        self.assertEqual(result['completed_updates'],1);self.assertEqual(result['stop_reason'],'time_budget')
    def test_nonfinite_gradient_rejected_and_hooks_restored(self):
        a=FakeAlgo();stats=EnvironmentStats(a.env);original=a._compute_ppo_loss
        a.actor.mu.grad=torch.full((29,),float('nan'))
        with tempfile.TemporaryDirectory(dir='/workspace') as tmp:
            with self.assertRaises(FloatingPointError):train(a,stats,tmp,lambda **kw:None,clock=lambda:0)
        self.assertEqual(a._compute_ppo_loss,original);self.assertFalse(stats.active)
    def test_native_checkpoint_saved_before_post_eval(self):
        text=Path(entry.__file__).read_text();self.assertLess(text.index('algo.save(str(final))'),text.index('after=evaluate'))
        self.assertEqual(text.count('algo.load(str(checkpoint))'),2)
    def test_no_credential_forward_and_hard_deadline(self):
        text=Path(entry.__file__).read_text();self.assertIn('child_env.pop(n,None)',text)
        self.assertIn('os._exit(124)',text);self.assertIn('signal.alarm(WORKER_SECONDS)',text)
    def test_output_upload_rejects_unapproved_hosts(self):
        for url in ['http://storage.googleapis.com/x','https://example.com/x','https://storage.googleapis.com.attacker.test/x','https://user@storage.googleapis.com/x']:
            with self.assertRaises(ValueError):entry.seal(url,{})
    def test_output_upload_rejects_api_credentials(self):
        with self.assertRaises(ValueError):entry.seal('https://storage.googleapis.com/output',{'Authorization':'Bearer prohibited'})
    def test_no_original_ppo_source_changes(self):
        import hashlib
        source=Path(entry.__file__).parents[1]/'thirdparty/holosoma/src/holosoma/holosoma/agents/ppo/ppo.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),'0899268dff8e9f8def38db6b76d2e4afd193c3d7a4ef28550217dc142d6898f6')
    def test_native_termination_manager_values_captured_without_extra_call(self):
        import ast,types
        source=Path(entry.__file__).parents[1]/'thirdparty/holosoma/src/holosoma/holosoma/managers/termination/manager.py'
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef))
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check')
        ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        class Stateful:
            def __init__(self):self.calls=0;self.resets=0
            def __call__(self,env):self.calls+=1;return torch.ones(32,dtype=torch.bool)
            def reset(self):self.resets+=1
        env=FakeEnv();term=Stateful()
        manager=NS(env=env,device='cpu',_term_names=['failure','timeout'],
                   _term_cfgs=[NS(params={},is_timeout=False),NS(params={},is_timeout=True)],
                   _term_instances={'failure':term},_term_funcs={'timeout':lambda e:torch.zeros(32,dtype=torch.bool)})
        manager.check=types.MethodType(ns['check'],manager);env.termination_manager=manager
        def check():env.reset_buf,env.time_out_buf=manager.check()
        env._check_termination=check
        stats=EnvironmentStats(env);stats.active=True;env._check_termination()
        self.assertEqual(stats.terminations,{'failure':32,'timeout':0});self.assertEqual(term.calls,1)
        manager._term_instances['failure'].reset();self.assertEqual(term.resets,1)
    def test_evaluation_uses_real_motion_configuration_not_unused_params(self):
        import inspect
        text=inspect.getsource(evaluate)
        self.assertIn('motion.motion_cfg',text);self.assertIn('use_adaptive_timesteps_sampler=False',text)
        self.assertNotIn('start_time_sampler=',text)

if __name__=='__main__':unittest.main()
"""Real native PPO update on synthetic CPU batches, not simulator training."""
import ast,copy,hashlib,io,json,sys,types,unittest,zipfile,__future__,logging
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime_snapshot/bundle/runtime/holosoma/src/holosoma'))
try: import loguru
except ModuleNotFoundError:
    shim=types.ModuleType('loguru');shim.logger=logging.getLogger('cpu-warning-shim');sys.modules['loguru']=shim
from three_adapter.ppo_routing import StrictRoutedTeacherCritic
from three_adapter.ppo_mask import ExplicitRoutePPOMixin
from three_adapter.ppo_config import build_ppo_smoke_dict,ppo_checkpoint_contract,validate_ppo_checkpoint
from holosoma.agents.modules.depth_backbone import DepthOnlyFCBackbone58x87Small
torch.set_num_threads(2)

MEMBER='bundle/runtime/holosoma/src/holosoma/holosoma/agents/distillation_ppo/distillation_ppo.py'
with zipfile.ZipFile('/workspace/public-core-check/core.zip') as archive:
    source=archive.read(MEMBER)
    pin=next(x for x in json.loads(archive.read('payload-manifest.json'))['files'] if x['path']==MEMBER)
    assert len(source)==pin['bytes'] and hashlib.sha256(source).hexdigest()==pin['sha256']
tree=ast.parse(source);definition=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DistillationPPO')
METHODS={'_build_optimizer','adjust_ppo_dagger_coeff','_set_std_lr_from_ppo_coef','_update_step','_dagger_loss_per_sample','_dagger_loss','_optimizer_step','_dagger_only_update','_compute_kl_div','_update_learning_rate'}
body=[x for x in definition.body if isinstance(x,ast.FunctionDef) and x.name in METHODS]
assert {x.name for x in body}==METHODS
definition.body=body;definition.bases=[];definition.decorator_list=[]
module=ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[]))
scope={'torch':torch,'nn':nn,'Any':object,'DistillationMinibatch':dict}
exec(compile(module,'hash-verified-native-distillation-ppo','exec',flags=__future__.annotations.compiler_flag),scope)
Native=scope['DistillationPPO']

def fixture():
    class Trainer(ExplicitRoutePPOMixin,Native): pass
    t=Trainer();t.is_multi_gpu=False;t.device='cpu';t.ppo_coef=0.;t.learning_rate=3e-4;t._warmup_active=False
    t.config=types.SimpleNamespace(learning_rate=3e-4,weight_decay=0.,depth_weight_decay=0.,
        max_grad_norm=1.,ppo_start_epoch=0,dagger_end_epoch=10000,desired_kl=.01,
        schedule='adaptive',clip_param=.2,value_loss_coef=1.,entropy_coef=.001,dagger_loss_coef=10.)
    t.policy=StrictRoutedTeacherCritic(num_actor_obs=96,num_teacher_obs=450,num_critic_obs=575,num_actions=29,
        depth_backbone=DepthOnlyFCBackbone58x87Small(32),depth_output_dim=32,
        student_hidden_dims=[2048,1024,512,256,128],teacher_hidden_dims=[512,256,128],
        critic_hidden_dims=[512,256,128],num_teachers=3,routing_column=449,activation='ELU',init_noise_std=.01)
    for i in range(3):
        c=torch.load(ROOT/'data/teachers'/f'{i}.pt',map_location='cpu',weights_only=True)
        t.policy.load_teacher_state_dict({'model_state_dict':c['actor_model_state_dict'],'actor_obs_normalizer_state_dict':None},teacher_index=i)
    t._build_optimizer()
    actor=torch.randn(6,96);depth=torch.randn(6,58,87);critic=torch.randn(6,575)
    teacher=torch.randn(6,450);teacher[:,449]=torch.tensor([0.,1.,2.,0.,1.,2.])
    with torch.no_grad():
        actions=t.policy.act({'actor_obs':actor,'depth_obs':depth})
        values=t.policy.evaluate({'critic_obs':critic})
        batch=dict(actor_obs=actor,depth_obs=depth,critic_obs=critic,teacher_obs=teacher,actions=actions.detach(),
            teacher_actions=t.policy.teacher_act(teacher),values=values.detach(),returns=values.detach()+torch.randn(6,1),
            advantages=torch.randn(6,1),actions_log_prob=t.policy.get_actions_log_prob(actions).detach().unsqueeze(1),
            action_mean=t.policy.action_mean.detach().clone(),action_sigma=t.policy.action_std.detach().clone(),ppo_gate=torch.ones(6,1))
    return t,batch

class PPOTests(unittest.TestCase):
    def test_01_real_native_combined_update_and_roundtrip(self):
        torch.manual_seed(42);t,b=fixture()
        frozen=copy.deepcopy(t.policy.teachers.state_dict())
        before={n:copy.deepcopy(getattr(t.policy,n).state_dict()) for n in ('student','critic','depth_backbone')}
        for iteration in range(5):
            t.adjust_ppo_dagger_coeff(iteration);t._set_std_lr_from_ppo_coef();metrics=t._update_step(b)
            self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        self.assertEqual(t.ppo_coef,.0004)
        for n in before:self.assertTrue(any(not torch.equal(v,getattr(t.policy,n).state_dict()[k]) for k,v in before[n].items()),n)
        self.assertTrue(all(torch.equal(v,t.policy.teachers.state_dict()[k]) for k,v in frozen.items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in t.policy.teachers.parameters()))
        state=dict(model_state_dict=t.policy.state_dict(),optimizer_state_dict=t.optimizer.state_dict(),
            three_teacher_contract=ppo_checkpoint_contract(),num_teachers=3,teacher_loaded=[True]*3,ppo_coef=t.ppo_coef,learning_rate=t.learning_rate,iter=4)
        stream=io.BytesIO();torch.save(state,stream);stream.seek(0);saved=torch.load(stream,map_location='cpu',weights_only=True);validate_ppo_checkpoint(saved)
        fresh,_=fixture();fresh.policy.load_training_state_dict(saved['model_state_dict']);fresh.optimizer.load_state_dict(saved['optimizer_state_dict'])
        self.assertTrue(all(torch.equal(v,fresh.policy.state_dict()[k]) for k,v in t.policy.state_dict().items()))
        for key,value in t.optimizer.state_dict()['state'].items():
            for field,v in value.items():
                self.assertTrue(torch.equal(v,fresh.optimizer.state_dict()['state'][key][field]))

    def test_02_explicit_mask_keeps_valid_zero_labels(self):
        t,b=fixture();t._dagger_route_valid=torch.tensor([True,False])
        p=torch.ones(2,29,requires_grad=True)
        loss=t._dagger_loss_per_sample(p,torch.zeros_like(p));self.assertEqual(loss.tolist(),[1.,0.]);loss.mean().backward()
        self.assertGreater(float(p.grad[0].abs().sum()),0);self.assertEqual(float(p.grad[1].abs().sum()),0)

    def test_03_all_lost_rows_still_train_critic(self):
        t,b=fixture();b['teacher_obs'][:,449]=-1.;b['teacher_actions'].zero_()
        before=copy.deepcopy(t.policy.critic.state_dict());t.ppo_coef=.5
        metrics=t._update_step(b);self.assertEqual(float(metrics['behavior']),0.)
        self.assertTrue(any(not torch.equal(v,t.policy.critic.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(hasattr(t,'_dagger_route_valid'))

    def test_04_unlabeled_warmup_skips_optimizer(self):
        t,b=fixture();t._warmup_active=True;b['teacher_obs'][:,449]=-1.
        before=copy.deepcopy(t.policy.state_dict());t._update_step(b)
        self.assertEqual(len(t.optimizer.state),0)
        self.assertTrue(all(torch.equal(v,t.policy.state_dict()[k]) for k,v in before.items()))

    def test_05_g1_schedule_not_accelerated(self):
        t,_=fixture()
        for iteration,expected in [(0,0.),(1,.0001),(1000,.1),(9000,.9),(10000,.9)]:
            t.adjust_ppo_dagger_coeff(iteration);self.assertEqual(t.ppo_coef,expected)

    def test_06_ppo_config_preserves_hud04_critic_and_rejects_pure(self):
        template=json.loads((ROOT/'data/locomotion-teacher-config.json').read_text())
        c=build_ppo_smoke_dict(template,bundle=Path('/verified/bundle'),smoke=ROOT/'data',teachers=[ROOT/'data/teachers'/f'{i}.pt' for i in range(3)],output=Path('/bounded/output'),mesh_vertices=[],mesh_faces=[])
        self.assertEqual(c['observation']['groups']['critic_obs'],template['observation']['groups']['critic_obs'])
        self.assertEqual(c['algo']['config']['dagger_loss_coef'],10.)
        self.assertEqual(c['algo']['config']['distillation_warmup_steps'],0)
        self.assertEqual(c['training']['num_envs'],6)
        with self.assertRaises(ValueError):validate_ppo_checkpoint({'three_teacher_contract':{'algorithm':'pure_dagger'}})

if __name__=='__main__':unittest.main(verbosity=2)
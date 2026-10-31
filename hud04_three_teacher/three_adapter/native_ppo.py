"""Native DAgger+PPO validation adapter. Requires a separately bounded worker.

Uses the SAME native loss/schedule/GAE/storage as G1. Not a submission entry.
"""
import copy,functools,hashlib,json,os,time
from pathlib import Path
import torch
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from .contract import BoundedUpdatesMixin, UpdateBudget, load_verified
from .native_algo import cpu_state, require_equal
from .routing import routing_column_index
from .ppo_routing import StrictRoutedTeacherCritic
from .ppo_mask import ExplicitRoutePPOMixin
from .ppo_config import ppo_checkpoint_contract, validate_ppo_checkpoint
from .smoke_config import check_live_observations

class HuD04ThreeTeacherPPO(BoundedUpdatesMixin, ExplicitRoutePPOMixin, DistillationPPO):
    def _checkpoint_metadata(self, iteration=None):
        metadata = super()._checkpoint_metadata(iteration=iteration)
        metadata.update(three_teacher_contract=ppo_checkpoint_contract(),
                        record_role='bounded_dagger_ppo_validation_not_skill_evaluation',
                        _wb_agent={k:os.environ[v] for k,v in [('thread_id','HUD04_ARIA_THREAD_ID'),('turn_id','HUD04_ARIA_TURN_ID')]})
        return metadata

    def _resolve_obs_dims(self):
        super()._resolve_obs_dims()
        self.live_contract = check_live_observations(self.env.observation_manager)
        if (self.actor_obs_dim,self.teacher_obs_dim,self.critic_obs_dim,self.num_act,self.depth_shape) != (96,450,575,29,(58,87)):
            raise ValueError('Hu-D04 actor/teacher/critic/action/depth contract changed')

    def _build_policy(self):
        self.policy = setup_student_teacher_module(num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_act,module_config=self.config.module,device=self.device,
            init_noise_std=self.config.init_noise_std,num_teachers=self.num_teachers,
            module_cls=functools.partial(StrictRoutedTeacherCritic,routing_column=routing_column_index(self.env)))

    def load_teacher(self, paths):
        if isinstance(paths,str): paths=[paths]
        load_verified(paths)
        return super().load_teacher(paths)

    def load(self, path):
        if path is not None: validate_ppo_checkpoint(torch.load(path,map_location='cpu',weights_only=True))
        return super().load(path)

    def save(self,path=None,name='last.ckpt'):
        if any(not torch.isfinite(t).all() for t in self.policy.state_dict().values()):
            raise FloatingPointError('Nonfinite policy checkpoint')
        return super().save(path=path,name=name)

    def learn(self,num_learning_iterations=None):
        if os.environ.get('HUD04_THREE_GPU_EXECUTION_APPROVED') != '1':
            raise PermissionError('GPU budget and bounded parent worker required')
        remaining=float(os.environ['HUD04_TRAIN_DEADLINE'])-time.monotonic()
        if not 0 < remaining <= 600 or hasattr(self,'update_budget'):
            raise ValueError('Invalid or repeated validation budget')
        if self.is_multi_gpu or self.env.num_envs!=6 or self.num_teachers!=3 or self.current_learning_iteration!=0:
            raise ValueError('Fresh six-env single-GPU validation only')
        c=self.config
        if (c.num_learning_iterations,c.num_steps_per_env,c.num_learning_epochs,c.num_mini_batches,c.distillation_warmup_steps,c.ppo_start_epoch,c.dagger_end_epoch)!=(5,24,2,2,0,0,10000):
            raise ValueError('Validation scope or G1 PPO ramp changed')
        self.update_budget=UpdateBudget(seconds=remaining)
        self.policy.require_loaded_teachers()
        frozen=cpu_state(self.policy.teachers)
        before={n:cpu_state(getattr(self.policy,n)) for n in ('student','critic','depth_backbone')}
        optimized={id(p) for g in self.optimizer.param_groups for p in g['params']}
        allowed={id(p) for n in ('student','critic','depth_backbone') for p in getattr(self.policy,n).parameters()}|{id(self.policy.std)}
        if optimized!=allowed or len(self.optimizer.param_groups)!=4:raise RuntimeError('PPO optimizer group mismatch')
        self._route_counts=[0]*3
        super().learn(num_learning_iterations)
        require_equal(frozen,cpu_state(self.policy.teachers))
        changed={n:any(not torch.equal(v,cpu_state(getattr(self.policy,n))[k]) for k,v in state.items()) for n,state in before.items()}
        if not all(changed.values()) or not all(self._route_counts):raise RuntimeError('Missing per-skill labels or student/critic/depth learning')
        final=Path(self.log_dir)/'student_dagger_ppo_validation.pt';self.save(str(final))
        state=cpu_state(self.policy);optimizer=copy.deepcopy(self.optimizer.state_dict());self.load(str(final))
        require_equal(state,cpu_state(self.policy));require_equal(optimizer,self.optimizer.state_dict())
        report=dict(algorithm='dagger_ppo',completed_updates=self.update_budget.completed,modules_changed=changed,
                    valid_labels_by_skill=self._route_counts,teachers_frozen=True,native_reload_verified=True,
                    checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),skill_success_verified=False)
        (Path(os.environ['HUD04_SMOKE_OUTPUT'])/'distillation-result.json').write_text(json.dumps(report,indent=2))

    def _update_ppo_gate_stats(self,student_mean,teacher_actions):
        # The three-skill command has no residual gate: native all-ones gate
        # retains the scalar G1 loss balance. Never derive masks from actions.
        obs=self.env.observation_manager.compute_group('teacher_obs',modify_history=False)
        route=obs[:,self.policy.routing_column]
        for i in range(3):self._route_counts[i]+=int((route==i).sum())

    def _post_epoch_logging(self,it,loss_dict):
        super()._post_epoch_logging(it,loss_dict)
        row=dict(iteration=int(it),completed_updates=self.update_budget.completed,
                 losses=loss_dict,valid_labels_by_skill=self._route_counts)
        (Path(os.environ['HUD04_SMOKE_OUTPUT'])/'progress.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(dict(event='hud04_dagger_ppo_progress',**row)),flush=True)
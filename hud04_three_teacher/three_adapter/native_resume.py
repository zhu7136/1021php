"""One bounded continuation of the hash-pinned three-teacher student."""
import hashlib,json,os,time
from pathlib import Path
import torch
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from .native_ppo import HuD04ThreeTeacherPPO
from .native_algo import cpu_state,require_equal
from .checkpoint_export import finalize_native_checkpoint
from .ppo_config import validate_ppo_checkpoint
from .resume_contract import (SOURCE_SHA,SOURCE_BYTES,SOURCE_ITER,MAX_UPDATES,TRAIN_SECONDS,
    ResumeBudget,ResumeBudgetExpired,verify_loaded,next_iteration)

class HuD04ThreeTeacherResume(HuD04ThreeTeacherPPO):
    def _checkpoint_metadata(self,iteration=None):
        m=super()._checkpoint_metadata(iteration=iteration)
        m.update(record_role='bounded_student_continuation_not_skill_evaluation',
            resume_source_sha256=SOURCE_SHA,resume_source_iter=SOURCE_ITER,
            trajectory_exact_resume=False)
        return m

    def load(self,path):
        if path is None:raise ValueError('Resume checkpoint required')
        p=Path(path)
        first=not hasattr(self,'resume_verified')
        if first and (p.is_symlink() or p.stat().st_size!=SOURCE_BYTES or hashlib.sha256(p.read_bytes()).hexdigest()!=SOURCE_SHA):
            raise ValueError('Resume source identity mismatch')
        saved=validate_ppo_checkpoint(torch.load(p,map_location='cpu',weights_only=True))
        result=super().load(path)
        # Native loader warns rather than fails on optimizer mismatch: enforce equality.
        verify_loaded(self,saved)
        if first:
            self.resume_next_iteration=next_iteration(saved)
            self.resume_verified=True
            report=dict(source_sha256=SOURCE_SHA,source_iteration=saved['iter'],
                next_iteration=self.resume_next_iteration,model_optimizer_scalars_readiness_equal=True,
                optimizer_groups=len(self.optimizer.param_groups),optimizer_slots=len(saved['optimizer_state_dict']['state']),
                fresh_process_resume_verified=True,trajectory_exact_resume=False)
            (Path(os.environ['HUD04_SMOKE_OUTPUT'])/'resume-verification.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(event='hud04_resume_verified',**report)),flush=True)
        return result

    def learn(self,num_learning_iterations=None):
        if os.environ.get('HUD04_RESUME_APPROVED')!='1000_updates_7200_seconds_once' or not getattr(self,'resume_verified',False):
            raise PermissionError('Explicit bound and full-state restore required')
        remaining=float(os.environ['HUD04_TRAIN_DEADLINE'])-time.monotonic()
        if not 0<remaining<=TRAIN_SECONDS or hasattr(self,'update_budget'):
            raise ValueError('Repeated or expired resume budget')
        c=self.config
        if self.is_multi_gpu or self.env.num_envs!=4096 or self.num_teachers!=3 or self.current_learning_iteration!=SOURCE_ITER:
            raise ValueError('Resume scope changed')
        if (c.num_learning_iterations,c.num_steps_per_env,c.num_learning_epochs,c.num_mini_batches,c.distillation_warmup_steps,c.ppo_start_epoch,c.dagger_end_epoch)!=(MAX_UPDATES,24,2,2,0,0,10000):
            raise ValueError('Recipe or update bound changed')
        if num_learning_iterations is not None and num_learning_iterations!=MAX_UPDATES:
            raise ValueError('Unexpected caller iteration override')
        self.update_budget=ResumeBudget(seconds=remaining)
        self.policy.require_loaded_teachers()
        frozen=cpu_state(self.policy.teachers)
        before={n:cpu_state(getattr(self.policy,n)) for n in ('student','critic','depth_backbone')}
        optimized={id(p) for g in self.optimizer.param_groups for p in g['params']}
        allowed={id(p) for n in before for p in getattr(self.policy,n).parameters()}|{id(self.policy.std)}
        if optimized!=allowed or len(self.optimizer.param_groups)!=4:raise ValueError('Optimizer parameter groups changed')
        self._route_counts=[0]*3
        torch.cuda.synchronize()
        print(json.dumps(dict(event='capacity_initialized',num_envs=self.env.num_envs,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            device_total_bytes=torch.cuda.get_device_properties(self.device).total_memory)),flush=True)
        self.current_learning_iteration=self.resume_next_iteration
        stop_reason='update_limit'
        try:
            # Bypass the validation-only learn guard, preserving native PPO loop/loss.
            DistillationPPO.learn(self,num_learning_iterations=MAX_UPDATES)
        except ResumeBudgetExpired:
            stop_reason='training_time_limit'
        if not self.update_budget.completed:raise ValueError('No completed resumed update')
        # If time expired during an unfinished rollout, label the last COMPLETED index.
        self.current_learning_iteration=self.resume_next_iteration+self.update_budget.completed-1
        require_equal(frozen,cpu_state(self.policy.teachers))
        if any(p.requires_grad or p.grad is not None for p in self.policy.teachers.parameters()):
            raise ValueError('Teacher freeze changed')
        changed={n:any(not torch.equal(v,cpu_state(getattr(self.policy,n))[k]) for k,v in state.items()) for n,state in before.items()}
        if not all(changed.values()) or not all(self._route_counts):raise ValueError('Missing learning or skill labels')
        output=Path(os.environ['HUD04_SMOKE_OUTPUT'])
        export=finalize_native_checkpoint(self,output,cpu_state,require_equal)
        (output/'checkpoint-export.json').write_text(json.dumps(export,indent=2))
        report=dict(algorithm='dagger_ppo',completed_updates=self.update_budget.completed,
            source_iteration=SOURCE_ITER,first_iteration=self.resume_next_iteration,
            last_completed_iteration=self.current_learning_iteration,total_completed_updates=self.current_learning_iteration+1,
            stop_reason=stop_reason,modules_changed=changed,valid_labels_by_skill=self._route_counts,
            teachers_frozen=True,native_reload_verified=True,fresh_process_resume_verified=True,
            checkpoint_sha256=export['checkpoint_sha256'],skill_success_verified=False)
        (output/'distillation-result.json').write_text(json.dumps(report,indent=2))

    def _training_step(self):
        if not self.update_budget.active or self.update_budget.clock()>=self.update_budget.deadline:
            raise ResumeBudgetExpired('Training time limit before optimizer update')
        result=DistillationPPO._training_step(self)
        self.update_budget.complete()
        return result

    def _post_epoch_logging(self,it,loss_dict):
        super()._post_epoch_logging(it,loss_dict)
        print(json.dumps(dict(event='formal_update',iteration=int(it),completed_updates=self.update_budget.completed,
            total_completed_updates=int(it)+1,behavior_loss=float(loss_dict['behavior']),total_loss=float(loss_dict['total_loss']))),flush=True)
"""Low-climb-76 model/optimizer continuation guard; bounded parent required."""
import hashlib,json,os,time
from pathlib import Path
import torch
from teacher_algo import FullTeacherPPO
from flux.teacher_algo import BoundedPPO,_require_same_state

SOURCE_SHA='9a0b61cdb689886c1171f77eb725ff85f4910573d7fa6376fdf4594415e45711'
SOURCE_ITERATION=660
NEXT_ITERATION=661

class LowClimbResumePPO(FullTeacherPPO):
    def load(self,path):
        # Native BoundedPPO final reloads must not be mistaken for source loads.
        source=os.environ.get('PHP_LOW_STEP_SOURCE_CHECKPOINT')
        if not source:
            raise ValueError('Explicit source checkpoint binding required')
        initial=path is not None and Path(path).resolve()==Path(source).resolve()
        if not initial:
            if not getattr(self,'_source_resume_verified',False):
                raise ValueError('First load must be the bound source checkpoint')
            return super().load(path)
        if getattr(self,'_source_resume_verified',False):
            raise ValueError('Source checkpoint already consumed')
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_SHA:
            raise ValueError('Source checkpoint SHA256 mismatch')
        expected=torch.load(path,map_location='cpu',weights_only=True)
        if expected.get('iter')!=SOURCE_ITERATION or not self.config.load_optimizer:
            raise ValueError('Source iteration or optimizer policy mismatch')
        result=super().load(path)
        for name,key in [('actor','actor_model_state_dict'),('critic','critic_model_state_dict'),('actor_optimizer','actor_optimizer_state_dict'),('critic_optimizer','critic_optimizer_state_dict')]:
            _require_same_state(expected[key],getattr(self,name).state_dict(),name)
        for name in ('actor_obs_normalizer','critic_obs_normalizer'):
            saved=expected.get(name+'_state_dict')
            if saved is not None:
                if getattr(self,name,None) is None:raise ValueError('Missing saved normalizer')
                _require_same_state(saved,getattr(self,name).state_dict(),name)
        for name in ('actor','critic'):
            lr=expected[name+'_optimizer_state_dict']['param_groups'][0]['lr']
            if getattr(self,name+'_learning_rate')!=lr:raise ValueError('Learning rate changed')
        if self.current_learning_iteration!=SOURCE_ITERATION:raise ValueError('Native iteration load mismatch')
        # Upstream saves the last COMPLETED loop index and learns from the
        # current index inclusive. Advance only the counter, not model state.
        self.current_learning_iteration=NEXT_ITERATION
        self._source_resume_verified=True
        row={'source_iteration':SOURCE_ITERATION,'next_iteration':NEXT_ITERATION,'source_sha256':SOURCE_SHA,'model_and_optimizers_exact':True,'learning_rates_exact':True,'simulator_rng_exact_replay':False}
        out=Path(os.environ['PHP_FLUX_OUTPUT']);out.mkdir(exist_ok=True,parents=True)
        (out/'source_resume_verified.json').write_text(json.dumps(row,indent=2))
        return result

    def learn(self):
        if not getattr(self,'_source_resume_verified',False) or self.current_learning_iteration!=NEXT_ITERATION:
            raise ValueError('Verified source restoration required before learning')
        if self.env.num_envs!=4096 or self.config.num_steps_per_env!=24:
            raise ValueError('Original environment/rollout scope required')
        out=Path(os.environ['PHP_FLUX_OUTPUT'])
        stop=os.environ.get('PHP_FLUX_STOP_FILE')
        if not stop or Path(stop).resolve()!=(out/'STOP_AFTER_UPDATE').resolve():
            raise ValueError('Bounded parent STOP_AFTER_UPDATE control required')
        from learning_algo import snapshot
        before,old=snapshot(self)
        self._started=time.monotonic()
        # Bypass FullTeacherPPO.learn, whose final evidence claims fresh training.
        result=BoundedPPO.learn(self)
        after,new=snapshot(self)
        checkpoint=Path(self.log_dir)/'model_smoke_final.pt'
        record={'completed_updates':int(self._flux_completed_updates),'actor_changed':any(not torch.equal(v,after['actor'][k]) for k,v in before['actor'].items()),'critic_changed':any(not torch.equal(v,after['critic'][k]) for k,v in before['critic'].items()),'actor_optimizer_steps_increased':new['actor_optimizer']>old['actor_optimizer'],'critic_optimizer_steps_increased':new['critic_optimizer']>old['critic_optimizer'],'parameters_finite':all(torch.isfinite(v).all().item() for a in after.values() for v in a.values()),'optimizer_step_sums_before':old,'optimizer_step_sums_after':new,'teacher_converged':False,'fresh_training':False,'new_process_resume_verified':True,'checkpoint_relative_path':str(checkpoint.resolve().relative_to(out.resolve())),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        with (out/'learning_state.json').open('x') as f:json.dump(record,f,indent=2)
        required=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite']
        if not 1<=record['completed_updates']<=100000 or not all(record[k] is True for k in required):
            raise ValueError('Continuation learning evidence incomplete')
        (out/'continuation_state.json').write_text(json.dumps({'fresh_training':False,'source_iteration':SOURCE_ITERATION,'first_new_iteration':NEXT_ITERATION,'last_iteration':int(self.current_learning_iteration),'new_completed_updates':int(self._flux_completed_updates),'teacher_converged':False,'model_optimizer_resume':True,'simulator_rng_exact_replay':False},indent=2))
        return result
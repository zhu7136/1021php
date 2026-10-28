"""4096-environment time-bounded continuation; never fallback or retry."""
import hashlib,json,os,time,math
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO,_StopAfterUpdate
from learning_algo import snapshot

class FormalPPO(BoundedPPO):
    def _post_epoch_logging(self,it,loss_dict):
        stop=False
        try:super()._post_epoch_logging(it,loss_dict)
        except _StopAfterUpdate:stop=True
        out=Path(os.environ['PHP_FLUX_OUTPUT'])
        elapsed=time.monotonic()-self._formal_start
        row={'update':self._flux_completed_updates,'iteration':int(it),'num_envs':int(self.env.num_envs),
             'training_elapsed_seconds':elapsed,'transitions':self._flux_completed_updates*4096*self.config.num_steps_per_env,
             'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved(),
             **{'Loss/'+k:float(v) for k,v in loss_dict.items()}}
        assert row['num_envs']==4096 and all(math.isfinite(v) for k,v in row.items() if k.startswith('Loss/'))
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps({'event':'formal_update',**row}),flush=True)
        if stop:raise _StopAfterUpdate()

    def learn(self):
        assert self.env.num_envs==4096 and self.config.save_interval==100
        assert self.config.num_steps_per_env==24 and self.config.load_optimizer
        out=Path(os.environ['PHP_FLUX_OUTPUT']);self._formal_start=time.monotonic()
        free,total=torch.cuda.mem_get_info()
        capacity={'event':'capacity_initialized','effective_num_envs':int(self.env.num_envs),
                  'num_steps_per_env':24,'samples_per_update':4096*24,'cuda_free_bytes':free,
                  'cuda_total_bytes':total,'cuda_allocated_bytes':torch.cuda.memory_allocated(),
                  'cuda_reserved_bytes':torch.cuda.memory_reserved(),'gpu':torch.cuda.get_device_name(0)}
        (out/'capacity.json').write_text(json.dumps(capacity,indent=2));print(json.dumps(capacity),flush=True)
        before,old_steps=snapshot(self)
        result=super().learn()
        after,new_steps=snapshot(self)
        record={'completed_updates':self._flux_completed_updates,
            'actor_changed':any(not torch.equal(before['actor'][k],after['actor'][k]) for k in before['actor']),
            'critic_changed':any(not torch.equal(before['critic'][k],after['critic'][k]) for k in before['critic']),
            'actor_optimizer_steps_increased':new_steps['actor_optimizer']>old_steps['actor_optimizer'],
            'critic_optimizer_steps_increased':new_steps['critic_optimizer']>old_steps['critic_optimizer'],
            'parameters_finite':all(torch.isfinite(v).all().item() for values in after.values() for v in values.values()),
            'same_learner_reload':True,'new_process_resume_verified':getattr(self,'_resume_verified',False),
            'teacher_converged':False,'effective_num_envs':4096}
        final=Path(self.log_dir)/'model_smoke_final.pt'
        record['checkpoint_sha256']=hashlib.sha256(final.read_bytes()).hexdigest()
        (out/'learning_state.json').write_text(json.dumps(record,indent=2))
        flags=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite','new_process_resume_verified']
        if not 1<=record['completed_updates']<=100000 or not all(record[k] for k in flags):
            raise ValueError('Missing/invalid formal learning evidence')
        from formal_closure import write_learning_receipt
        write_learning_receipt(out,os.environ['PHP_FLUX_TASK_ID'])
        (out/'algo-learn-returning.json').write_text(json.dumps({'algo_learn_returning':True,'completed_updates':self._flux_completed_updates}))
        return result
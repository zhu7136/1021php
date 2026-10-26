"""4096-env fresh G1 teacher: validate 20 updates, then one bounded 30m segment."""
import copy,json,os,time
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO,_require_same_state

def budget_times(validation_deadline,gate):
    if not gate or gate.get('passed') is not True:
        return validation_deadline-30,validation_deadline
    return gate['formal_started_monotonic']+1680,gate['formal_started_monotonic']+1800

class G1LowstepPPO(BoundedPPO):
    def learn(self):
        if (self.env.num_envs!=4096 or self.config.num_steps_per_env!=24 or
            self.config.num_learning_epochs!=5 or self.config.num_mini_batches!=4 or
            self.config.num_learning_iterations!=20020 or self.config.save_interval!=500):
            raise ValueError('Low-step 4096 recipe changed')
        if not torch.cuda.is_available():raise ValueError('Low-step trial requires CUDA')
        self._g1_peak=0;self._g1_gate=False
        torch.cuda.reset_peak_memory_stats(self.device)
        result=super().learn()
        output=Path(os.environ['PHP_FLUX_OUTPUT'])
        if not self._g1_gate or self._flux_completed_updates<=20:
            raise ValueError('Validation or formal updates incomplete')
        (output/'formal_result.json').write_text(json.dumps({
            'validation_passed':True,'validation_updates':20,'formal_updates':self._flux_completed_updates-20,
            'total_updates':self._flux_completed_updates,'teacher_converged':False,
            'formal_worker_limit_seconds':1800,'dataset':'motions-low-step',
            'device_used_peak_boundary_sampled_bytes':self._g1_peak,
            'memory_caveat':'Boundary sampled device memory is not continuous peak; Torch peaks exclude simulator allocations'}))
        return result

    def _post_epoch_logging(self,it,loss_dict):
        super()._post_epoch_logging(it,loss_dict)
        free,total=torch.cuda.mem_get_info(self.device)
        self._g1_peak=max(self._g1_peak,total-free)
        output=Path(os.environ['PHP_FLUX_OUTPUT'])
        if self._flux_completed_updates==20:
            # Strict native checkpoint save/load gate before any formal updates.
            path=Path(self.log_dir)/'model_validation20.pt'
            self.save(str(path))
            names=('actor','critic','actor_obs_normalizer','critic_obs_normalizer','actor_optimizer','critic_optimizer')
            before={n:copy.deepcopy(getattr(self,n).state_dict()) for n in names if getattr(self,n,None) is not None}
            scalars={n:copy.deepcopy(getattr(self,n)) for n in ('current_learning_iteration','actor_learning_rate','critic_learning_rate')}
            self.load(str(path))
            for n,value in before.items():_require_same_state(value,getattr(self,n).state_dict(),n)
            for n,value in scalars.items():_require_same_state(value,getattr(self,n),n)
            self._g1_gate=True
            report={'passed':True,'num_envs':4096,'completed_updates':20,'native_reload_passed':True,
                    'dataset':'motions-low-step','checkpoint':str(path.relative_to(output)),
                    'formal_started_monotonic':time.monotonic(),'formal_worker_limit_seconds':1800,
                    'device_used_peak_boundary_sampled_bytes':self._g1_peak,
                    'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
                    'checked_state':list(before),'checked_scalars':list(scalars)}
            (output/'capacity_gate.json').write_text(json.dumps(report))
            print(json.dumps({'event':'lowstep_validation_passed_formal_started',**report}),flush=True)
        if self._flux_completed_updates%100==0:
            print(json.dumps({'event':'lowstep_progress','total_updates':self._flux_completed_updates,
                              'formal_updates':max(0,self._flux_completed_updates-20),
                              'device_used_peak_boundary_sampled_bytes':self._g1_peak}),flush=True)
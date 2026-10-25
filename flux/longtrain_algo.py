"""Native continuation plus memory observations; no PPO/physics modifications."""
import json,os,time
from pathlib import Path
import torch
_text=Path(__file__).with_name('formal_algo.py').read_text()
if _text.count('from flux.formal_input import')!=1:raise ValueError('Native base changed')
_text=_text.replace('from flux.formal_input import','from flux.longtrain_input import',1)
_namespace={'__name__':__name__+'.native_base','__file__':__file__}
exec(compile(_text,__file__,'exec'),_namespace)

class LongtrainPPO(_namespace['FormalContinuationPPO']):
    def learn(self):
        if self.env.num_envs!=4096 or self.config.num_steps_per_env!=24:
            raise ValueError('Fullscale must use4096env24steps')
        if (self.config.num_learning_iterations!=20000 or self.config.save_interval!=500
                or self.config.num_learning_epochs!=5 or self.config.num_mini_batches!=4):
            raise ValueError('Fullscale native update/save recipe changed')
        if not torch.cuda.is_available():raise ValueError('Fullscale requires CUDA')
        self._resource_started=time.monotonic()
        self._resource_peak=0
        self._resource_observations=0
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._write_resource(False,'before_learning')
        result=super().learn()
        self._write_resource(True,'segment_after_native_reload')
        return result

    def _write_resource(self,segment_passed,stage):
        free,total=torch.cuda.mem_get_info(self.device)
        used=total-free
        self._resource_peak=max(self._resource_peak,used)
        report={'segment_passed':segment_passed,'stage':stage,'num_envs':self.env.num_envs,
                'num_steps_per_env':self.config.num_steps_per_env,'rollout_samples_per_update':98304,
                'completed_updates':getattr(self,'_flux_completed_updates',0),
                'target_updates':20000,'num_learning_epochs':self.config.num_learning_epochs,
                'num_mini_batches':self.config.num_mini_batches,'save_interval':500,
                'elapsed_learning_and_reload_seconds':round(time.monotonic()-self._resource_started,3),
                'device_used_peak_sampled_at_update_boundaries_bytes':self._resource_peak,
                'device_memory_observations':self._resource_observations,
                'cuda':{'device_used_bytes':used,'device_free_bytes':free,'device_total_bytes':total,
                        'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
                        'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device)},
                'gpu_memory_caveat':'Global used is boundary-sampled,not continuous;Torch peaks exclude simulator allocations',
                'teacher_convergence_claimed':False}
        (Path(os.environ['PHP_FLUX_OUTPUT'])/'resources.json').write_text(json.dumps(report,indent=2))
        if segment_passed or self._resource_observations%100==0:
            print(json.dumps({'event':'longtrain_resource_observation',**report}),flush=True)

    def _post_epoch_logging(self,it,loss_dict):
        try:return super()._post_epoch_logging(it,loss_dict)
        finally:
            self._resource_observations+=1
            self._write_resource(False,'update_boundary')
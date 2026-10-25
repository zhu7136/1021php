"""Native PPO capacity verification, with bounded20updates and CUDA evidence."""
import json,os,time
from pathlib import Path
import torch

# Isolate new source pins/update ceiling; leave formal_algo module unchanged.
_text=Path(__file__).with_name('formal_algo.py').read_text()
if _text.count('from flux.formal_input import')!=1:raise ValueError('Capacity algorithm contract changed')
_text=_text.replace('from flux.formal_input import','from flux.capacity_input import',1)
_namespace={'__name__':__name__+'.native_base','__file__':__file__}
exec(compile(_text,__file__,'exec'),_namespace)

class CapacityPPO(_namespace['FormalContinuationPPO']):
    def _capacity_gpu(self):
        free,total=torch.cuda.mem_get_info(self.device)
        return {'torch_allocated_bytes':torch.cuda.memory_allocated(self.device),
                'torch_reserved_bytes':torch.cuda.memory_reserved(self.device),
                'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
                'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device),
                'device_free_bytes':free,'device_total_bytes':total,
                'device_used_bytes':total-free}

    def learn(self):
        if self.env.num_envs!=4096 or self.config.num_steps_per_env!=24:
            raise ValueError('Capacity must use4096env and24step full rollout')
        if self.config.num_learning_iterations!=20:
            raise ValueError('Capacity must perform20updates')
        if not torch.cuda.is_available():raise ValueError('GPU capacity check requires CUDA')
        self._capacity_started=time.monotonic()
        self._capacity_device_peak=0
        self._capacity_observations=0
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._write_capacity(False,'before_learning')
        result=super().learn()
        progress=json.loads((Path(os.environ['PHP_FLUX_OUTPUT'])/'progress.json').read_text())
        if self._flux_completed_updates!=20 or progress['stopped_early']:
            raise ValueError('Capacity incomplete:20fullupdates and native reload required')
        self._write_capacity(True,'after_native_reload')
        return result

    def _write_capacity(self,passed,stage):
        gpu=self._capacity_gpu()
        self._capacity_device_peak=max(self._capacity_device_peak,gpu['device_used_bytes'])
        report={'passed':passed,'stage':stage,'num_envs':self.env.num_envs,
                'num_steps_per_env':self.config.num_steps_per_env,
                'rollout_samples_per_update':self.env.num_envs*self.config.num_steps_per_env,
                'completed_updates':getattr(self,'_flux_completed_updates',0),
                'num_learning_epochs':self.config.num_learning_epochs,
                'num_mini_batches':self.config.num_mini_batches,
                'elapsed_learning_and_reload_seconds':round(time.monotonic()-self._capacity_started,3),
                'device_used_peak_sampled_at_update_boundaries_bytes':self._capacity_device_peak,
                'device_memory_observations':self._capacity_observations,
                'cuda':gpu,'gpu_memory_caveat':'Torch peaks exclude non-Torch allocations; device peak is boundary-sampled,not continuous',
                'full_scale_training_completed':False,'teacher_convergence_claimed':False}
        (Path(os.environ['PHP_FLUX_OUTPUT'])/'capacity.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'event':'capacity_observation',**report}),flush=True)

    def _post_epoch_logging(self,it,loss_dict):
        try:return super()._post_epoch_logging(it,loss_dict)
        finally:
            self._capacity_observations+=1
            self._write_capacity(False,'update_boundary')
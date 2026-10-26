"""CUDA capacity check only; refuses formal training and changed recipes."""
import json
import os
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO


class G1ValidationPPO(BoundedPPO):
    def learn(self):
        if (self.env.num_envs != 4096 or self.config.num_steps_per_env != 24 or
                self.config.num_learning_epochs != 5 or self.config.num_mini_batches != 4 or
                self.config.num_learning_iterations != 20 or self.config.save_interval != 5 or
                self.config.load_optimizer is not True):
            raise ValueError('G1 validation recipe changed')
        if not torch.cuda.is_available():
            raise ValueError('G1 validation requires CUDA')
        self._g1_peak = 0
        torch.cuda.reset_peak_memory_stats(self.device)
        result = super().learn()
        output = Path(os.environ['PHP_FLUX_OUTPUT'])
        native = json.loads((output/'native_reload.json').read_text())
        if (self._flux_completed_updates != 20 or native.get('native_load_passed') is not True or
                set(native.get('checked_optimizers', [])) != {'actor_optimizer', 'critic_optimizer'} or
                not {'actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer'}.issubset(native.get('checked_modules', []))):
            raise ValueError('G1 validation incomplete')
        report = {'passed': True, 'num_envs': self.env.num_envs, 'completed_updates': 20,
                  'formal_updates': 0, 'native_reload_passed': True, 'teacher_converged': False,
                  'dataset': 'motions-low-step', 'device_used_peak_boundary_sampled_bytes': self._g1_peak,
                  'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
                  'memory_caveat': 'Boundary samples are not a continuous peak; Torch excludes simulator allocations'}
        (output/'validation_result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'event': 'g1_validation_result', **report}), flush=True)
        return result

    def _post_epoch_logging(self, it, loss_dict):
        super()._post_epoch_logging(it, loss_dict)
        free, total = torch.cuda.mem_get_info(self.device)
        self._g1_peak = max(self._g1_peak, total-free)
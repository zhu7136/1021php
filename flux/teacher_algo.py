"""PPO instrumentation: preserve upstream physics/rewards and stop at update boundaries."""
import json
import math
import os
from pathlib import Path

import torch
from holosoma.agents.ppo.ppo import PPO


class _StopAfterUpdate(BaseException):
    pass


class BoundedPPO(PPO):
    def learn(self):
        self._flux_completed_updates = 0
        try:
            result = super().learn()
        except _StopAfterUpdate:
            # A final native checkpoint has already been saved. The parent
            # marks this as an incomplete smoke if fewer updates were done.
            result = None
        # Use the native load method on the exact saved actor/critic/optimizer.
        # Fake-PPO CPU tests do not implement native load and do not verify it.
        if hasattr(self, 'load'):
            output = Path(os.environ['PHP_FLUX_OUTPUT'])
            final = output / 'model_smoke_final.pt'
            self.save(str(final))
            before = {name: {key: value.detach().clone() for key, value in getattr(self, name).state_dict().items()}
                      for name in ('actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer')
                      if getattr(self, name, None) is not None}
            self.load(str(final))
            for name, values in before.items():
                for key, expected in values.items():
                    if not torch.equal(getattr(self, name).state_dict()[key], expected):
                        raise ValueError('Native checkpoint roundtrip changed model state')
            (output / 'native_reload.json').write_text(json.dumps({
                'native_load_passed': True, 'path': final.name,
                'completed_updates': self._flux_completed_updates,
                'checked_modules': list(before)}))
        return result

    def export(self, onnx_file_path):
        # This smoke validates PPO/native checkpoints, not ONNX deployment.
        print(json.dumps({'event': 'onnx_export_skipped_for_checkpoint_smoke'}), flush=True)

    def _post_epoch_logging(self, it, loss_dict):
        for name, value in loss_dict.items():
            if not math.isfinite(float(value)):
                raise FloatingPointError(f'Nonfinite training statistic: {name}')
        super()._post_epoch_logging(it, loss_dict)
        self._flux_completed_updates += 1
        output = Path(os.environ['PHP_FLUX_OUTPUT'])
        stop = (output / 'STOP_AFTER_UPDATE').exists()
        progress = {'completed_updates': self._flux_completed_updates,
                    'last_iteration_index': int(it), 'stopped_early': stop,
                    'losses': {key: float(value) for key, value in loss_dict.items()}}
        (output / 'progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps({'event': 'complete_update', **progress}), flush=True)
        if stop:
            self.save(str(Path(self.log_dir) / f'model_{int(it):05d}.pt'))
            raise _StopAfterUpdate()

    def save(self, path, infos=None):
        for name in ('actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer'):
            module = getattr(self, name, None)
            if module is None:
                continue
            for key, value in module.state_dict().items():
                if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                    raise FloatingPointError(f'Nonfinite checkpoint parameter: {name}.{key}')
        return super().save(path, infos)
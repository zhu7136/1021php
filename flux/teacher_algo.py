"""PPO instrumentation: preserve upstream physics/rewards and stop at update boundaries."""
import copy
import json
import math
import os
from pathlib import Path

import torch
from holosoma.agents.ppo.ppo import PPO


class _StopAfterUpdate(BaseException):
    pass


def _require_same_state(expected, actual, name):
    """Exact recursive comparison, including optimizer buffers and scalars."""
    message = f'Native checkpoint roundtrip changed state: {name}'
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or expected.shape != actual.shape
                or expected.dtype != actual.dtype or not torch.equal(expected, actual)):
            raise ValueError(message)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(message)
        for key, value in expected.items():
            _require_same_state(value, actual[key], f'{name}.{key}')
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(message)
        for index, value in enumerate(expected):
            _require_same_state(value, actual[index], f'{name}[{index}]')
    elif type(actual) is not type(expected) or expected != actual:
        raise ValueError(message)


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
        # CPU contract fixtures exercise this branch, but are NOT real native
        # Holosoma/IsaacSim reload evidence.
        if hasattr(self, 'load'):
            output = Path(os.environ['PHP_FLUX_OUTPUT'])
            # Upstream save_checkpoint_artifact requires the actual run log_dir,
            # not the parent output directory used for our control reports.
            final = Path(self.log_dir) / 'model_smoke_final.pt'
            relative_path = final.resolve().relative_to(output.resolve()).as_posix()
            self.save(str(final))
            if not final.is_file() or final.stat().st_size == 0:
                raise ValueError('Native final checkpoint missing or empty')
            before = {name: copy.deepcopy(getattr(self, name).state_dict())
                      for name in ('actor', 'critic', 'actor_obs_normalizer', 'critic_obs_normalizer')
                      if getattr(self, name, None) is not None}
            optimizers = {name: copy.deepcopy(getattr(self, name).state_dict())
                          for name in ('actor_optimizer', 'critic_optimizer')
                          if self.config.load_optimizer and getattr(self, name, None) is not None}
            scalars = {name: copy.deepcopy(getattr(self, name))
                       for name in ('current_learning_iteration', 'actor_learning_rate', 'critic_learning_rate')
                       if hasattr(self, name)}
            self.load(str(final))
            for name, values in before.items():
                _require_same_state(values, getattr(self, name).state_dict(), name)
            for name, values in optimizers.items():
                _require_same_state(values, getattr(self, name).state_dict(), name)
            for name, value in scalars.items():
                _require_same_state(value, getattr(self, name), name)
            (output / 'native_reload.json').write_text(json.dumps({
                'native_load_passed': True, 'path': relative_path,
                'completed_updates': self._flux_completed_updates,
                'iteration': self.current_learning_iteration,
                'checked_modules': list(before), 'checked_optimizers': list(optimizers),
                'checked_scalars': list(scalars)}))
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
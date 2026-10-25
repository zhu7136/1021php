"""Native PPO continuation from task048; relative-500 checkpoint cadence."""
import json
import os
from pathlib import Path
import re
import time
import wandb
from flux.live_monitor import relative_checkpoint
_text = Path(__file__).with_name('formal_algo.py').read_text()
if _text.count('from flux.formal_input import') != 1:
    raise ValueError('Pinned native continuation changed')
_text = _text.replace('from flux.formal_input import', 'from flux.continue1000_input import', 1)
_namespace = {'__name__': __name__+'.native_base', '__file__': __file__}
exec(compile(_text, __file__, 'exec'), _namespace)

class Continue1000PPO(_namespace['FormalContinuationPPO']):
    def learn(self):
        if (self.env.num_envs != 4096 or self.config.num_steps_per_env != 24
                or self.config.num_learning_iterations != 1000 or self.config.save_interval != 500
                or self.config.num_learning_epochs != 5 or self.config.num_mini_batches != 4):
            raise ValueError('Unexpected continuation recipe')
        if wandb.run is None or wandb.run.settings.mode != 'online':
            raise ValueError('An online native training run must exist before PPO starts')
        self._live_started = time.monotonic()
        result = super().learn()
        validation = json.loads((Path(os.environ['PHP_FLUX_OUTPUT'])/'resume_validation.json').read_text())
        validation['requested_updates_completed'] = self._flux_completed_updates == 1000
        (Path(os.environ['PHP_FLUX_OUTPUT'])/'resume_validation.json').write_text(json.dumps(validation,indent=2))
        return result

    def _post_epoch_logging(self, it, loss_dict):
        try:
            return super()._post_epoch_logging(it, loss_dict)
        finally:
            completed = self._flux_completed_updates
            if completed > 0:
                elapsed = time.monotonic()-self._live_started
                print(json.dumps({'event':'continuation_progress','progress':f'{completed}/1000',
                    'completed_updates':completed,'native_iteration':int(it),
                    'rollout_samples_per_second':round(completed*98304/max(elapsed,.001),2),
                    'elapsed_training_seconds':round(elapsed,2)}),flush=True)
                if relative_checkpoint(completed):
                    self.save(str(Path(self.log_dir)/f'model_{int(it):05d}.pt'))

    def save(self, path, infos=None):
        # Native PPO also requests absolute-index saves (e.g.4000,4500). Suppress
        # those so the user's cadence is NEW updates500/1000, not global index.
        if re.fullmatch(r'model_\d+\.pt',Path(path).name):
            if not relative_checkpoint(getattr(self,'_flux_completed_updates',0)):
                return None
        result = super().save(path,infos)
        if wandb.run is None or wandb.run.settings.mode != 'online':
            raise ValueError('Checkpoint saved locally but online run is unavailable')
        # Native helper uses policy=live(default); request immediate upload too.
        # This queues a transfer, NOT proof of server-side backup completion.
        wandb.save(path,base_path=self.log_dir,policy='now')
        print(json.dumps({'event':'checkpoint_upload_requested','file':Path(path).name,
            'completed_updates':self._flux_completed_updates,'remote_verified':False}),flush=True)
        return result
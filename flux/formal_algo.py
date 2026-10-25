"""Bounded native PPO continuation; never changes physics or the PPO update."""
import json
import os
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO, _require_same_state
from flux.resume_algo import MODULES, OPTIMIZERS, SCALARS, snapshot, finite, require_loaded_state, require_updated
from flux.formal_input import CHECKPOINT_SHA256, SOURCE_ITERATION, MAX_UPDATES, digest


class FormalContinuationPPO(BoundedPPO):
    def learn(self):
        output = Path(os.environ['PHP_FLUX_OUTPUT'])
        source = Path(os.environ['PHP_FLUX_RESUME_CHECKPOINT'])
        if digest(source) != CHECKPOINT_SHA256 or self.config.num_learning_iterations != MAX_UPDATES:
            raise ValueError('Unexpected formal source or update ceiling')
        if not self.config.load_optimizer or self.current_learning_iteration != SOURCE_ITERATION:
            raise ValueError('Native continuation state was not loaded')
        saved = torch.load(source, weights_only=True, map_location='cpu')
        before = snapshot(self)
        finite(before)
        require_loaded_state(self, saved, before)
        for name, optimizer in (('actor_learning_rate','actor_optimizer_state_dict'),
                                ('critic_learning_rate','critic_optimizer_state_dict')):
            _require_same_state(saved[optimizer]['param_groups'][0]['lr'], before[name], name)
        _require_same_state(saved['iter'], before['current_learning_iteration'], 'loaded.iteration')
        (output/'resume_loaded.json').write_text(json.dumps({
            'native_loaded_state_matches_source':True,'source_sha256':CHECKPOINT_SHA256,
            'source_iteration':SOURCE_ITERATION,'checked_modules':list(MODULES),
            'checked_optimizers':list(OPTIMIZERS),'checked_scalars':list(SCALARS),
            'empirical_normalization':self.empirical_normalization}))
        print(json.dumps({'event':'formal_source_loaded','source_iteration':SOURCE_ITERATION}),flush=True)
        # Replay already verified on task031. Start native training directly;
        # native PPO handles train mode and reset, with no new inference context.
        result = super().learn()
        after = snapshot(self)
        finite(after)
        changes, steps = require_updated(before, after)
        completed = self._flux_completed_updates
        if not 1 <= completed <= MAX_UPDATES:
            raise ValueError('Formal continuation made no updates or exceeded ceiling')
        if self.current_learning_iteration != SOURCE_ITERATION + completed - 1:
            raise ValueError('Unexpected native continuation indexing')
        if digest(source) != CHECKPOINT_SHA256:
            raise ValueError('Formal source checkpoint was modified')
        native = json.loads((output/'native_reload.json').read_text())
        if native['native_load_passed'] is not True or native['completed_updates'] != completed:
            raise ValueError('Formal final checkpoint roundtrip failed')
        progress = json.loads((output/'progress.json').read_text())
        if not progress['stopped_early'] and completed != MAX_UPDATES:
            raise ValueError('Unexpected early termination without bounded stop')
        report = {'passed':True,'source_sha256':CHECKPOINT_SHA256,'source_iteration':SOURCE_ITERATION,
                  'completed_updates':completed,'native_final_iteration':self.current_learning_iteration,
                  'parameters_changed':changes,'optimizer_step_advancement':steps,
                  'source_checkpoint_unchanged':True,'final_checkpoint_native_reload_passed':True,
                  'termination_reason':'worker_time_budget' if progress['stopped_early'] else 'update_ceiling',
                  'exact_environment_rng_resume_claimed':False,'teacher_convergence_claimed':False}
        (output/'resume_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'event':'formal_continuation_passed',**report}),flush=True)
        return result

    def _post_epoch_logging(self, it, loss_dict):
        # Keep native logs and validated boundary stop. Append small scalar-only
        # history even when the parent raises its intentional soft-stop signal.
        try:
            return super()._post_epoch_logging(it,loss_dict)
        finally:
            output = Path(os.environ['PHP_FLUX_OUTPUT'])
            progress_path = output/'progress.json'
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text())
                if progress['last_iteration_index'] == int(it):
                    with (output/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(progress)+'\n')
"""Native PPO continuation with exact-source audit and bounded recovery saves."""
import hashlib
import json
import math
import os
import time
from pathlib import Path
import torch
from flux.teacher_algo import BoundedPPO, _StopAfterUpdate
from learning_algo import snapshot
from speed10_audit import install_native_audit
from resume_audit import verify_loaded_state, SOURCE_UPDATES


class FullTeacherPPO(BoundedPPO):
    def load(self, checkpoint):
        result = super().load(checkpoint)
        if checkpoint is not None and Path(checkpoint).resolve() == Path(os.environ['PHP_RESUME_CHECKPOINT']).resolve():
            self._resume_evidence = verify_loaded_state(self, checkpoint, os.environ['PHP_FLUX_OUTPUT'])
        return result

    def _recovery_checkpoint(self):
        root = Path(self.log_dir)
        pending = root / 'model_pending.pt'
        latest = root / 'model_latest.pt'
        previous = root / 'model_previous.pt'
        self.save(str(pending))
        if latest.exists():
            os.replace(latest, previous)
        os.replace(pending, latest)

    def _post_epoch_logging(self, iteration, losses):
        stop = False
        try:
            super()._post_epoch_logging(iteration, losses)
        except _StopAfterUpdate:
            stop = True
        out = Path(os.environ['PHP_FLUX_OUTPUT'])
        row = dict(update=self._flux_completed_updates, iteration=int(iteration), num_envs=int(self.env.num_envs),
                   elapsed_seconds=time.monotonic() - self._started, transitions=self._flux_completed_updates * 4096 * 24,
                   source_iteration=1361, source_completed_updates=SOURCE_UPDATES,
                   cumulative_completed_updates=SOURCE_UPDATES + self._flux_completed_updates,
                   **{'Loss/' + k: float(v) for k, v in losses.items()})
        assert int(iteration) == SOURCE_UPDATES + self._flux_completed_updates - 1
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError('nonfinite_update_metric')
        with (out / 'updates.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
            f.flush()
        print(json.dumps(dict(event='formal_update', **row)), flush=True)
        if self._flux_completed_updates == 1 or self._flux_completed_updates % 100 == 0 or stop:
            self._recovery_checkpoint()
            self._speed_flush(self._flux_completed_updates)
        if stop:
            raise _StopAfterUpdate()

    def learn(self):
        out = Path(os.environ['PHP_FLUX_OUTPUT'])
        self._started = time.monotonic()
        assert self.env.num_envs == 4096 and self.config.num_steps_per_env == 24
        assert self._resume_evidence['exact_training_state_loaded'] and self.current_learning_iteration == SOURCE_UPDATES - 1
        # This checkpoint stores the last zero-based completed iteration.
        self.current_learning_iteration = SOURCE_UPDATES
        self._speed_flush = install_native_audit(self.env, out)
        before, old_steps = snapshot(self)
        capacity = dict(effective_num_envs=int(self.env.num_envs), samples_per_update=4096 * 24,
                        gpu=torch.cuda.get_device_name(0), cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        (out / 'capacity.json').write_text(json.dumps(capacity, indent=2))
        print(json.dumps(dict(event='capacity_initialized', **capacity)), flush=True)
        result = super().learn()  # original rollouts, optimizer, native save/reload
        after, new_steps = snapshot(self)
        checkpoint = Path(self.log_dir) / 'model_smoke_final.pt'
        record = dict(completed_updates=self._flux_completed_updates,
                      actor_changed=any(not torch.equal(v, after['actor'][k]) for k, v in before['actor'].items()),
                      critic_changed=any(not torch.equal(v, after['critic'][k]) for k, v in before['critic'].items()),
                      actor_optimizer_steps_increased=new_steps['actor_optimizer'] > old_steps['actor_optimizer'],
                      critic_optimizer_steps_increased=new_steps['critic_optimizer'] > old_steps['critic_optimizer'],
                      parameters_finite=all(torch.isfinite(v).all().item() for values in after.values() for v in values.values()),
                      teacher_converged=False, skills_evaluated=False, fresh_training=False,
                      new_process_resume_verified=True, trajectory_exact_resume=False,
                      source_completed_updates=SOURCE_UPDATES, source_iteration=1361,
                      cumulative_completed_updates=SOURCE_UPDATES + self._flux_completed_updates,
                      source_task='TASK_20261029_011', checkpoint_relative_path=str(checkpoint.relative_to(out)),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (out / 'learning_state.json').write_text(json.dumps(record, indent=2))
        flags = ['actor_changed', 'critic_changed', 'actor_optimizer_steps_increased', 'critic_optimizer_steps_increased', 'parameters_finite']
        if not 1 <= record['completed_updates'] <= 2000 or not all(record[k] for k in flags):
            raise ValueError('native_learning_evidence_failed')
        return result
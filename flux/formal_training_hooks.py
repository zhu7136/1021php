"""Stage-one time budget, verified resume and credential-free scalar capture.

Does not replace PPO rollout/update, rewards, termination or physical parameters.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

CHECKPOINT_SHA = '5c8573b1a954f3e58a4242f3a1858aa5de06c20c860465dddf7d373250ff2489'
ITERATION_CEILING = 1000000  # Safety ceiling; wall clock, not this number, ends stage.


class StageBudgetReached(Exception):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = Path(str(path) + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class ScalarMirror:
    def __init__(self, original, path):
        self.original, self.path = original, Path(path)
        self.values = {}
    def __getattr__(self, name):
        return getattr(self.original, name)
    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        value = float(scalar_value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite training metric: ' + tag)
        self.values[str(tag)] = value
        with self.path.open('a') as target:
            target.write(json.dumps(dict(step=int(global_step), key=str(tag), value=value)) + '\n')
        return self.original.add_scalar(tag, scalar_value, global_step, *args, **kwargs)


def instrument(cls, out, checkpoint, soft_deadline, clock=time.monotonic):
    if cls.__dict__.get('_hud04_formal_wrapped', False):
        return cls
    out, checkpoint = Path(out), Path(checkpoint)
    learn, post = cls.learn, cls._post_epoch_logging

    def epoch(self, it, losses):
        result = post(self, it, losses)
        self._hud04_completed_updates += 1
        values = self.logging_helper.writer.values
        keep = ('Loss/Value', 'Loss/Surrogate', 'Train/mean_reward',
                'Train/mean_episode_length', 'Perf/total_fps')
        event = dict(event='formal_training_progress', iteration=int(it),
            completed_updates=self._hud04_completed_updates,
            metrics={key: values[key] for key in keep if key in values})
        print(json.dumps(event, allow_nan=False), flush=True)
        if clock() >= soft_deadline:
            raise StageBudgetReached()
        return result

    def bounded_learn(self):
        if digest(checkpoint) != CHECKPOINT_SHA:
            raise ValueError('Resume checkpoint hash mismatch')
        if (self.env.num_envs != 32 or self.config.num_learning_iterations != ITERATION_CEILING
                or self.config.save_interval != 10 or not self.config.load_optimizer):
            raise ValueError('Formal parsed training bounds mismatch')
        if clock() >= soft_deadline:
            raise TimeoutError('No remaining stage budget for PPO')
        self.load(str(checkpoint))  # Native actor/critic/normalizer/optimizer/env resume.
        if self.current_learning_iteration != 1:
            raise ValueError('Unexpected resume iteration')
        self.current_learning_iteration += 1  # Saved iter 1 is already complete.
        self._hud04_completed_updates = 0
        self.logging_helper.writer = ScalarMirror(self.logging_helper.writer, out / 'metrics.jsonl')
        print(json.dumps(dict(event='formal_resume_verified', source_iteration=1,
            next_iteration=2, checkpoint_sha256=CHECKPOINT_SHA, optimizer_restored=True)), flush=True)
        reason = 'iteration_ceiling'
        try:
            learn(self)
        except StageBudgetReached:
            reason = 'approved_time_budget'
        except BaseException as exc:
            atomic_json(out / 'failure.json', dict(exception_type=type(exc).__name__,
                completed_updates=self._hud04_completed_updates))
            raise
        if self._hud04_completed_updates < 1:
            raise RuntimeError('No PPO updates completed')
        final = Path(self.log_dir) / 'model_stage1_final.pt'
        self.save(str(final))
        if not final.is_file() or final.stat().st_size == 0:
            raise RuntimeError('Final checkpoint missing')
        if digest(checkpoint) != CHECKPOINT_SHA:
            raise RuntimeError('Read-only resume checkpoint changed')
        atomic_json(out / 'training_stage_completed.json', dict(version=1,
            stop_reason=reason, completed_updates=self._hud04_completed_updates,
            last_iteration=int(self.current_learning_iteration), num_envs=32,
            motion_variants=32, checkpoint_input_sha256=CHECKPOINT_SHA,
            checkpoint_input_unchanged=True, final_checkpoint=str(final),
            final_checkpoint_sha256=digest(final), final_checkpoint_bytes=final.stat().st_size,
            log_dir=str(self.log_dir), worker_limit_seconds=3600, automatic_retries=0))
        print(json.dumps(dict(event='formal_training_stage_saved', stop_reason=reason,
            completed_updates=self._hud04_completed_updates,
            last_iteration=int(self.current_learning_iteration))), flush=True)

    cls.learn, cls._post_epoch_logging = bounded_learn, epoch
    cls._hud04_formal_wrapped = True
    return cls


def install(out, checkpoint, soft_deadline):
    import holosoma.train_agent as training
    original = training.get_class
    def resolve(target):
        cls = original(target)
        # Instrument only the actual PPO class, after its normal import phase.
        if cls.__module__ == 'holosoma.agents.ppo.ppo' and cls.__name__ == 'PPO':
            return instrument(cls, out, checkpoint, soft_deadline)
        return cls
    training.get_class = resolve
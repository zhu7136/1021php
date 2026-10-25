# Continuous-segment evaluation accounting (CPU-tested repair)

## Scope

The protected frozen-model evaluator now opts into continuous-segment accounting
via `evaluate(..., continuous_segments=True)`. Other callers retain the old
default; training paths are unchanged. Native episode metrics are kept under
their original names. They are NOT renamed into continuous-clip survival.

The existing native MotionCommand performs internal clip-end robot/root/dof
resets without ending the environment episode. The observer wraps the existing
`env._update_tasks_callback` once, invokes it exactly once, then copies the
explicit native `motion_end_reset` boolean flag and reference frames. It does
not invoke a reset, a termination term, RNG, simulator write, or physics step.
The callback is restored in `evaluate`'s finally block, including error paths.

Only the supported WBT order (termination before command update) is accepted.
Missing native flags or unexpected callback counts fail closed. Frame jumps
alone are NOT presented as explicit native reset evidence. Freeze-at-start and
a reset that coincidentally samples a contiguous frame do not confuse the flag.

## Boundary semantics and outputs

If native command update resets after control step k:

- Physics, termination and pre-reset telemetry at step k belong to the old
  segment. Its observed duration ends at k*dt, not (k-1)*dt.
- `internal_reset_events` records `boundary_after_step=k`, frame before/after,
  environment ID, segment ID, explicit flag evidence, and any same-step native
  termination. Post-first-environment-episode resets and warmup are excluded.
- The new segment starts at boundary k. Its first physical control sample is
  k+1. Do NOT compute displacement, velocity or uninterrupted contact runs across
  the telemetry edge k -> k+1. Join telemetry to these event boundaries first.
- A reset truncates the old segment as `censored_internal_reset`; it is neither
  failure nor success. A same-step native termination takes precedence because
  it occurs before command update. No segment restarts after first native done.
- Repeated internal resets create separate segments. A horizon-step reset never
  produces a zero-duration segment or a first-segment horizon-survival claim.

Each model/seed writes `continuous_segments.json`, also embedded in
`evaluation.json`. Pair summaries include separate first-segment counts/durations.

Fields include `first_segment_native_terminations`,
`first_segment_internal_reset_censored`, `first_segment_horizon_survivors`, and
`first_segment_mean_observed_seconds`. The last is a descriptive observed
duration, NOT a corrected/unbiased survival estimate: clip-end censoring can
depend on the initial motion phase. Later segments are not independent resets
and must not be pooled as independent model trials. Pair comparisons should use
the common uninterrupted prefix and disclose censoring in either arm.

Historical TASK_20261025_005 artifacts are NOT rewritten. Their frame-jump
inferences remain retrospective evidence; this new explicit flag was not
retroactively sampled. Historical episode improvement remains episode-scoped.

## CPU verification and limits

`python -m unittest discover -s flux/tests -p test_continuous_segments.py`
has 19 dedicated regressions. They cover no-reset equivalence, internal/reset
and terminal precedence, repeated/final-step resets, post-failure exclusion,
invalid accounting, restoration, explicit flags even with contiguous frames,
RNG/native-write equivalence, a real native RootStatesProxy plus full mocked
128-step evaluation, and error cleanup. The pinned native MotionCommand.step
AST prefix is executed through the actual reset and simulator-flush branch;
its SHA256 is checked before extraction. Tests do NOT import or run IsaacSim
physics, nor execute the subsequent native alignment math on CPU.

No GPU task, new W&B run, training update or checkpoint is produced by this
repair. Physical settings, rewards, targets, PD, limits, native reset logic and
native termination thresholds are unchanged. A future GPU smoke check needs
separate approval; CPU success is not GPU validation.
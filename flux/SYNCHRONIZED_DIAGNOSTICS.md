# HU_D04 frozen-evaluation synchronized diagnostics

This is a CPU-tested, opt-in instrumentation change based on the completed
TASK_20261024_103 source commit `27471cb5f34e503562c01b6a31676b8006eaa27c`.
It does not backfill missing measurements into TASK103 and does not authorize
or submit another GPU task. No controller/state from TASK099/103 may be reused.

## Scope and unchanged behavior

`evaluate_protected_multiseed.py` accepts `--synchronized-diagnostics`; the
default remains OFF. The parent forwards the option to each independent worker.
The TASK097 iteration-0/200 checkpoints, three evaluation seeds, 32 environments,
128 control steps, target guard, physical/control/randomization parameters,
BadTracking thresholds, no-training/no-optimizer-step prohibitions and unchanged
weight checks are unchanged. No PD gains, effort limits or termination changes.
No action substitution/reference-PD intervention is introduced.

Diagnostics wrap the native physics and push calls exactly once, consume no RNG,
and copy tensors before serialization. Their CPU overhead can increase wall
time; no increase of the existing 1800s worker budget, automatic retry, or GPU
approval is implied. Output exceeding a budget is an error, not silent truncation.

## Files included in each arm's existing result bundle

- `sync_schema.json`: names, coordinate frame, sign, clocks, torque semantics.
- `sync_static_state.json`: configured effort/velocity limits, default target
  pose and base pose. Native `setup_dof_pos_bias` shifts `default_dof_pos`; it
  is a target/rest-pose offset, NOT a measured observation/sensor bias.
- `sync_control_steps.jsonl`: aligned/reference and actual tracked-body XYZ,
  signed XYZ error (`actual-reference`) and 3D norm; torso XYZ error; motion
  ID/global frame; four ankle reference q/qd, requested/applied target, actual
  q/qd and both actual-target and actual-reference differences; ankle-roll-link
  net world contact force, velocity/horizontal speed; native termination flags.
- `sync_physics_substeps.jsonl`: `(step,substep,env)` join key, start/end time,
  ankle q/qd before/after native physics, requested/applied targets, actuator
  reported computed/applied torque estimates, per-foot net world contact force.
- `sync_push_events.jsonl`: actual native `_push_robots` invocation/return,
  affected first-episode environment IDs, before/after root linear/angular
  velocity and delta, motion frame/ID, boundary time. A push after control step
  k is assigned to input of k+1, never retrospectively to failure at k. An event
  after the final horizon has no observed response in this evaluation.
- `sync_coverage.json`: emitted only after all 128 control steps and the expected
  physics-substep coverage; includes missing optional actuator reports.

No nested folders are used: the existing bounded bundle packager includes these
JSON/JSONL files automatically. Per-arm JSONL bound is 16MiB; original aggregate
raw/compressed bundle bounds remain 128MiB/32MiB. Numeric telemetry is rounded
to six decimals in native units; validity/mapping/reference/target comparisons
use the original unrounded tensors. Existing exact initial-state pairing files
remain unchanged and unrounded.

## Timing and non-claims

Only the first episode is included; its failing step is INCLUDED, while warmup
and auto-reset trajectories are EXCLUDED using the existing alive mask. Control
sampling is after native termination evaluation and before reward/reset/command
advance. The reference at physics entry must equal the reference at this
boundary; otherwise fail closed instead of comparing different motion frames.
Native WBT advances command and applies pushes after termination, not before.
The adapter requires this IsaacSim ordering and verifies DOF name/index mapping.
Net-force sensor ordering is mapped by body NAME, independently of DOF order.
Sensor update period must not exceed the physics substep period; no sensor
configuration is changed to make this check pass.

**Torque limitation:** in implicit-position mode, IsaacLab
`Articulation.data.applied_torque` / `computed_torque` are actuator reports and
may be PD estimates, not the actual joint effort produced by the implicit PhysX
solver. Fields explicitly say `*_torque_estimate_nm`.
`measured_solver_joint_effort_nm` is NULL and metadata says unavailable. Missing
reports remain NULL, never synthesized from Kp*q_error or filled with zero.
These estimates alone do NOT prove actuator saturation or spare real torque
capacity. A validated solver effort sensor/API is still needed for that claim.

**Contact limitation:** net force on ankle-roll links is NOT a contact-point
force or proof of terrain-only support. The `>1N` flag is only a diagnostic
convention, not a policy/termination change. Link horizontal speed under that
flag is only a slip proxy: foot rotation, other contacts and contact geometry
can confound it. Reference stance/contact labels and contact-point slip are
unavailable; do not invent them from XYZ error or force alone.

## Diagnosis rules for a future separately approved evaluation

1. Compare ankle XYZ growth against torso error and the native failing step.
   Separate horizontal from vertical error; neither alone establishes slip.
2. If commanded q diverges from reference q while actual q follows commanded q,
   prioritize policy-target/reference mismatch. Compare at matched times/frames.
3. If commanded q is near reference but actual q lags, inspect qd, contact and
   actuator reports. Do NOT conclude torque saturation from an implicit estimate.
4. Join actual push events to subsequent steps, and inspect rest-pose offset.
   A configured push interval or noise range is not an observed event.
5. Report unavailable quantities explicitly and retain native safety gates.

## CPU verification (not GPU physics validation)

Run `python -m unittest discover -s flux/tests` from the repository root.
New tests check timing/counts for 128 steps, XYZ sign/norm, reversed native DOF
and contact ordering, missing torque, warmup and post-failure exclusion,
pre-reset copies, reference advance detection, target mismatch, finite values,
native push event timing/exclusion, unchanged RNG/targets, flat packaging,
serialization bounds, slow sensor rejection and wrapper restoration.

Push tests execute the actual pinned Holosoma WBT `_push_robots` AST, SHA256
`5bd1b8626781773d5368a14fefb16e35e8ffdd85728bcd389d345da0fce87735`, using
the actual hash-checked native RootStatesProxy/converters already covered by
the TASK099 regression tests. Holosoma pin:
`4a0bf04cde5930af5ed59da797568d32be972525`.
Physics/sensor data in these tests is mocked. Real GPU smoke/evaluation remains
unexecuted and requires a fresh explicit compute approval and submission state.
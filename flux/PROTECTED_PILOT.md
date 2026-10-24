 # HU_D04 opt-in protected-target pilot

Experimental safety/observability recipe, not a proven tracking or convergence fix.
Entrypoint: `python flux/train_protected_pilot.py` with the same scoped output-upload
and W&B-run arguments as `train_fresh_pilot.py`. Do not submit until compute bounds
are approved. Historical presets and previous checkpoints are not overwritten.

## Exact changes relative to TASK_20261024_083

- Fresh random initialization; do NOT resume a checkpoint trained under unit action
  scale. Same network, PPO optimizer/learning-rate schedule, exploration std,
  positive tracking rewards, physical-joint-limit penalty and motion data recipe.
- Native implicit-PD target mapping `q_default + 0.25 * raw_action` instead of
  `q_default + 1.0 * raw_action`.
- Before native actuator application, clamp each target to its own simulator hard
  limits with 0.02 rad inward margin. Reject nonfinite values, invalid ranges and
  default poses outside these guarded ranges. Keep PPO raw actions/log-probability
  semantics unchanged; do not insert clipped targets into PPO rollout storage.
- Replace raw-action rate cost by rate cost in physical target-delta units,
  `sum(((a_t-a_prev)*action_scales)^2)`, weight remains -0.1.
- Add pre-protection request cost `sum((requested_target-applied_target)^2)`,
  weight -1.0 (reward manager multiplies by control dt). This exposes persistent
  policy drift even if all applied targets are safe.
- Add failure cost -1.0 per non-timeout termination, not per simulated second:
  native termination flag divided by dt, reward-manager weight -1.0. Preserve
  existing termination rules; do not invoke stateful termination functions twice.
- Keep simulator reset/step buffer creation in normal-tensor/no-grad mode, even
  when native PPO calls them within inference_mode. Clone incoming actions at this
  boundary. Leave native policy inference and PPO gradient updates intact. This
  addresses the TASK083 training-to-evaluation inference-tensor write exception.

## Proposed bounded validation (awaiting fresh approval)

### TASK095 initialization-order repair

TASK095 reached IsaacSim scene creation but failed before PPO setup: native
BaseTask constructs ActionManager before log buffers, and WBT creates its
default pose only after BaseTask initialization returns. The guarded action
constructor must not read that pose. It now defers validation until the first
process_actions call, before native action processing; apply_actions fails closed
until validation succeeds. No action/reward/PPO settings change in this repair.
Six new CPU regressions cover late/missing/invalid/nonfinite poses, application
before validation, and early reset. All 324 CPU tests pass; GPU proof is pending.

One Flux task, one 4090D 24GB, 32 environments, fresh initialization, at most
200 completed updates OR 300 seconds training; worker maximum 1800 seconds,
excluding platform container startup/clone; no automatic retries. Includes native
preflight save/load, deterministic pre/post evaluations (seed1402, 32 environments,
128 control steps, uniform reference frames and native observation/reset noise),
then native final checkpoint save/load and bundle upload. A hard training deadline
may stop an incomplete update without a final checkpoint; never present that as a
completed pilot. Results can demonstrate plumbing/safety and early behavior only.

## Readback and decision rules

- Confirm selected task resource, exact code revision and no resume/checkpoint
  mount before start. Verify native processed/applied-target readback agreement.
- Applied-target hard-limit violations must be zero. Also report PRE-protection
  request violation fraction, guard activation, left-hip policy mean/std and
  requested mean; safe clipped targets alone do not demonstrate a learned fix.
- Require finite telemetry, a final checkpoint with native actor/critic/optimizer/
  iteration roundtrip, and completed post-training evaluation.
- Compare deterministic pre/post first-episode restricted survival and failure
  reasons at identical seeds/reset protocol. This is one small, noisy comparison,
  not a success-rate estimate. Report reward components separately: rewards under
  this recipe cannot be compared numerically with historical total rewards.
- If failure or persistent request drift remains, stop and report it. Do not
  silently launch another task, tune penalties, extend training or retry.

CPU tests exercise the pinned native action class, tensor lifecycle regression,
reward blind-spot protection and opt-in configuration. These are NOT IsaacSim GPU
validation. New coefficients are initial test values, not calibrated guarantees.
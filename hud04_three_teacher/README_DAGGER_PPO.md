# Hu-D04 three-skill DAgger + PPO update

Status: CPU implementation and 27 regression tests passed. NO GPU submitted.
This adds a separate combined algorithm; the historical pure-DAgger adapter,
its checkpoint schema and all prior teacher tasks remain unchanged.

## Training contract

- Skills, in order: locomotion1361, low-step1272, high-step1373. No low-climb.
- Reuse verified `hud04-three-skill-data:v0`, 162 train/54 held-out clips.
- Preserve Hu-D04 31 physical joints/29 actions, 96 student proprio features,
  449 privileged teacher features + routing, 575 critic features, 58x87 depth.
- Use native Holosoma `DistillationPPO`, not renamed behavior cloning:
  clipped PPO surrogate + entropy, always-active value loss, DAgger MSE.
- G1 scalar recipe: PPO coefficient ramps from0 to0.9 over10000 iterations;
  DAgger coefficient10, LR3e-4, clip0.2, gamma0.99, GAE lambda0.95,
  entropy0.001, value coefficient1. No accelerated ramp in validation.
- Retain explicit failed-route masking, rather than interpreting a valid zero
  action as an invalid label. DAgger uses full batch-mean denominator like G1;
  unlabeled samples can still receive PPO/value supervision.
- Native four-group AdamW: student, critic, depth backbone, action std.
- The three-skill command does not implement G1's optional residual gate.
  Native all-ones gating preserves the scalar combined recipe; no claim of
  equivalent residual-gated motion curriculum or matched closed-loop success.

## Added modules

`ppo_config.py`, `ppo_routing.py`, `ppo_mask.py`, `native_ppo.py` under
`three_adapter/`. Use `build_ppo_smoke_dict`, not the old pure-DAgger CPU entry.
The native target is `three_adapter.native_ppo.HuD04ThreeTeacherPPO`.
Validation is intentionally restricted to a fresh6-env,5-update student,
single GPU, <=600 training seconds. It is not a production job launcher.

Before GPU: separately approve resource/time budget, prepare a bounded parent
with whole-job watchdog (including input/install/startup/shutdown/upload), bind
runtime/data hashes, and verify Flux configuration and owner. Native runtime
import, depth rendering, simulator rollout, clean process exit, GPU save/load
and per-skill closed-loop performance remain unvalidated. Historical teacher
jobs segfaulted at interpreter shutdown; do not mask or ignore that risk.

## CPU evidence

6 new tests execute hash-verified native `_update_step`, optimizer and schedule
method bodies on real native networks and actual frozen Hu-D04 teacher states,
with synthetic observations/advantages. They check three trainable modules,
teacher freezing, four-group model/optimizer roundtrip, valid-zero-label mask,
PPO/value update with invalid teacher labels, warmup skip and schedule.
21 original data/route/backprop tests also pass. A stdlib warning logger shim
substitutes for unavailable loguru in CPU tests; networks/math are not mocked.
No synthetic test weights are published as a trained student or a metric result.

Restore original adapter artifact and dataset first, overlay this code archive.
Tests currently require the original authenticated core archive and runtime
snapshot at the recorded workspace locations. The artifact includes the exact
native PPO source used for tests and its source hash for audit.
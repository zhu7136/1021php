# PHP locomotion teacher on Flux — preparation, not a running experiment

Prepared on 2026-10-24 for `HuD04_WBT` (`PRO_20261023_011`).

## Status

- CPU tests: six tests passed, including seven rejection cases in one test.
- Simulator/GPU initialization: NOT tested.
- Learning updates and real checkpoint reload: NOT tested.
- Data download/delivery and robot mesh closure: NOT yet verified.
- Target-worker dependencies and output retrieval: NOT yet verified.
- GPU spending: NOT authorized by `smoke-plan.json`; no task submitted.

Tests use a fake PPO implementation with CPU tensors. They establish wrapper
behavior only, not training quality or simulator compatibility.

## Fixed inputs and deliberate differences

- Upstream PHP base: `c7c9ce3fc75a8754a7e403c58de3b6933254d01b`.
- Holosoma submodule: `4a0bf04cde5930af5ed59da797568d32be972525`.
- Dataset: full `motions-locomotion` bundle; 201 declared files, 282340220
  extracted bytes; no data has been bundled or uploaded by this preparation.
- Candidate Flux image: `BJX00000170` / `isaac-lab2.3.1-v3`.
- Candidate runtime: Python 3.11, Torch 2.7.0, IsaacSim 5.1.0, IsaacLab 2.3.1.
  PHP's own installer uses IsaacLab 2.3.0; 2.3.1 is an UNVERIFIED adaptation.
- Preset: `g1_wbt_terrain_ref_noscale`; rewards, termination, action scaling,
  motion set and obstacle variant settings are unchanged.
- Proposed smoke: one GPU, 128 environments, 20 updates, save interval five,
  worker deadline 1200 seconds, no automatic retry. These are reduced plumbing
  settings, not the 4096-environment / 20000-update formal recipe.
- Calls the existing image Python directly; does not require the upstream
  launcher's `$HOME/.holosoma_deps/miniconda3/envs/php` path.
- W&B is offline; TensorBoard/config/native checkpoints remain in output.
  Retrieve and sync the offline run later. No sandbox account key is supplied.

## Preparation checks (CPU only)

From the checkout root:

```bash
python flux/bootstrap.py --check
python flux/test_preparation.py
```

## Remaining preparation

1. Fetch the pinned Holosoma submodule and validate all URDF/mesh references.
2. Download the official locomotion bundle in an approved delivery environment;
   run `scripts/download_assets.py motions-locomotion --verify --destination DIR`.
3. Audit preprocessing memory use at 128 environments with all motions and the
   preset's original obstacle variants. Do not silently reduce motion coverage.
4. Prepare compatible offline increments for the actual worker interpreter.
   Install Holosoma from this checkout; do not replace image Torch or fetch
   unpinned packages while paying for GPU time.
5. Establish a platform-supported persistent output path / retrieval mechanism.
   A local output directory alone does not prove results can be recovered.
6. Confirm the actual currency, billing unit and user budget. API resource price
   `5.4` provides neither currency nor unit. Platform startup/clone/release may
   be billable outside the worker deadline.
7. Set approval fields only after approval; pin the final branch commit.
   Create a unique Flux draft once, read it back, then submit once.

## Entry after those gates are satisfied

Use `gm-run php_parkour/flux/bootstrap.py` with explicit `--approval-file`,
`--data`, and `--output` arguments pointing to verified worker paths. No example
worker paths or approvals are invented here. The current smoke-plan deliberately
fails closed and is not a request to launch.

The parent starts a dedicated process group. It requests stop after an update
at worker deadline minus 120 seconds, sends TERM at minus 30 seconds, and uses
KILL if needed. Soft stopping preserves a native checkpoint only if an update
boundary is reached in time. A hang during initialization or hard timeout can
prevent a usable final checkpoint; it is reported as failure, not success.

`completion.json` reports updates, exit status and checkpoint hashes. It always
leaves `acceptance_passed=false` until checkpoint retrieval, hash verification
and native reload are independently verified. W&B offline data does not appear
on the server until it has been retrieved and synced.

No Flux API key, GitHub token, W&B API key, signed URL or approval value is
stored in these files.
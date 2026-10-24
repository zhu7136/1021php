# PHP locomotion teacher on Flux — preparation, not a running experiment

Prepared on 2026-10-24 for `HuD04_WBT` (`PRO_20261023_011`).

## Status

- CPU tests: six tests passed, including seven rejection cases in one test.
- Simulator/GPU initialization: NOT tested.
- Learning updates and real checkpoint reload: NOT tested.
- Official data downloaded locally and all manifest hashes verified; NOT yet
  delivered to the Flux worker. Pinned Holosoma checkout and 34 unique G1
  URDF mesh references verified as actual files, not LFS pointers.
- Full-data CPU preprocessing: passed with ONE audit layout, preserving all
  100 clips / 71,042 frames / 100 motion boundaries. Simulator NOT started.
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

After downloading the official bundle, run the reproducible data audit:

```bash
python flux/audit_preprocess.py --data /absolute/path/to/motions-locomotion \
  --scratch /absolute/path/to/audit-output
```

The scratch path holds generated arrays and a JSON report. This CPU audit
requires numpy, torch and trimesh; it never starts the simulator. It checks
the full source hashes, finite numeric arrays, root DOF layout, consistent
body/joint names, and actual URDF mesh files before preprocessing. The audit
uses ONE layout solely for validation; the training preset remains at 64.

Observed CPU result on 2026-10-24: 100 clips, 71,042 frames, 50 fps, 29 joints,
32 bodies, 50 nonempty terrain arrays, 4,088 mesh vertices / 6,132 faces.
The audit sandbox was Python 3.13 / NumPy 2.4.4 / trimesh 4.7.4, NOT the target
Python 3.11 worker. Target Holosoma requires NumPy <2. CPU success does not
establish target interpreter, IsaacLab, GPU or training compatibility.

From the actual array shapes, 64-layout source-format storage is estimated at
16.751 GiB and the core float32 MotionLib tensors at 8.028 GiB of GPU memory.
These are NOT measured peaks and exclude temporary copies, index tensors,
simulator state, terrain geometry, PPO storage and optimizer. The preprocessing
converts body positions to float32, so generated NPZ sizes can be smaller than
the source-format estimate. Reducing simulation environment count does not
remove the preset's 64-fold motion expansion.

## Remaining preparation

1. Deliver the verified full bundle to an accessible worker path and reverify
   its hashes there. Local sandbox data is not automatically available to Flux.
2. Run the CPU audit under the actual target interpreter/dependencies.
3. Validate peak memory with the preset's original 64 obstacle variants on the
   target runtime; the 1-layout CPU audit does not measure that peak. Do not
   silently reduce motion coverage or terrain variants.
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
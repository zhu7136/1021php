# PHP locomotion teacher on Flux — preparation, not a running experiment

Prepared on 2026-10-24 for `HuD04_WBT` (`PRO_20261023_011`).

## Status

### Latest local correction — 2026-10-25 cross-device comparison (CPU-only)

Task `TASK_20261025_026` at published commit
`42f3c51a6ef46a5e5bf0dd3091734f7589eff6e7` completed 20 updates, saved the
final checkpoint inside the real log directory, and returned from upstream
`PPO.load` (including optimizer restoration). It still FAILED acceptance:
the adapter's subsequent optimizer-state comparison called `torch.equal`
across CPU and CUDA storage. No complete `native_reload.json` was produced.
The exact offending optimizer key was not recorded, so it is not established.

This local repair compares tensors using detached CPU views/copies. It retains
exact tensor shape, dtype, finite-value and value checks, along with the
recursive key, sequence, scalar, optimizer and iteration checks. It does not
move or replace live optimizer buffers, change their device-placement policy,
cast dtype, introduce approximate tolerances, skip optimizer verification, or
change native save/load and training parameters. Transfer errors still fail.

New CPU tests use CPU storage with explicit MOCK device labels and an equal
operation that reproduces the original cross-device error. They check both
directions, nested optimizer buffers, corruption and nonfinite rejection,
unchanged source state/gradients, and copy failures without success markers.
An integrated CPU fixture includes mixed-label optimizer state after load.
These are NOT real CUDA tensors or tests of live CUDA transfer/continuation.
The original worker acceptance remains failed until a real-runtime test proves
the full roundtrip. Neither CPU success nor native load returning alone proves
that a subsequent optimizer update or simulator replay is valid.

CPU regression: 91/91 tests passed on Python 3.13/Torch CPU, including 12 new
comparison tests; Python 3.11 pure-Python subset 47/47 plus syntax passed.
Python 3.11 has no Torch locally, so this is not target CUDA verification.
At 2026-10-25T02:34:19Z the user authorized publishing this three-file repair
and ONE bounded Flux validation: 1x4090D24GB,128env,20updates,worker<=1200s,
automatic retries=0. This note is authorization, NOT evidence of submission or
success. No formal training or further paid retries are covered.
Earlier sections below describe historical preparation.

### Latest local correction — 2026-10-25 final checkpoint directory (CPU-only)

Task `TASK_20261025_023`, at published commit
`88703e5953b48fb1b250e58be214e80b25489b4b`, passed version/source preflight and
completed all 20 requested PPO updates. Five distinct iteration checkpoints
were retrieved with verified hashes. The last checkpoint passed a safe CPU
`torch.load(weights_only=True)` and finite-tensor inspection. The smoke still
FAILED: the added adapter tried to save `training/model_smoke_final.pt` outside
Holosoma's timestamped run log directory, and the native logger rejected it.
The native reload call was never reached. RabbitMQ errors were not the final
exception, and the worker did not reach its hard timeout.

This local-only repair changes final checkpoint placement to
`Path(self.log_dir) / 'model_smoke_final.pt'`. The control report remains at
`PHP_FLUX_OUTPUT/native_reload.json`; its `path` is relative to that outer
output directory and now includes the real nested run directory. Paths outside
the outer output (including symlink escapes), missing/empty checkpoint files,
save/load failures and state mismatches fail without a success report.

The adapter invokes the unchanged native save/load methods, compares actor,
critic and enabled observation-normalizer states exactly, and also checks
optimizer buffers/parameter groups when `config.load_optimizer` is enabled,
plus the iteration and learning-rate scalars. It does NOT force optimizer
loading or alter the training configuration, physics, rewards, input data,
image, GPU count, environment count, update budget or runtime budget.

`test_checkpoint_roundtrip.py` uses a CPU contract fixture with real small
Torch modules and optimizer buffers, but NOT Holosoma's real PPO or IsaacSim.
It reproduces task023's old failure before the repair and exercises nested
logging, soft-stop boundaries, state corruption and failure propagation.
Passing these unit tests does NOT establish real-worker native reload,
simulator replay, successful smoke acceptance or trained policy quality.

Run the local CPU suite with:

```bash
python -m unittest discover -s flux -p 'test_*.py'
```

This repair has not been published, and authorizes no GPU start or retry.
The older status sections below are historical; task023 supersedes their
claims that version recognition, simulator startup or PPO updates are untested.

### Latest local correction — 2026-10-25 runtime version identity (CPU-only)

Task `TASK_20261025_021` passed the shallow-source check, then failed because
`metadata.version('isaacsim')` is unavailable in the standalone installation.
The old check also incorrectly compared IsaacLab's extension version with its
product release. This local correction does not authorize another GPU start.

- New stdlib-only `runtime_versions.py` resolves evidence from the actual
  interpreter/module locations, without importing Torch, IsaacSim or IsaacLab.
- IsaacSim requires 5.1.0 evidence: importable pip distribution or standalone
  `VERSION` anchored to the interpreter/import and a `python.sh` + `kit` layout.
  Missing files, shadowed imports and contradictory pip/file evidence fail.
  Official IsaacSim tag `v5.1.0` stores `5.1.0-rc.19` in `VERSION`; that exact
  marker is explicitly recognized, not arbitrary prereleases. Build suffixes
  are retained in raw evidence. No version override or silent skip exists.
- IsaacLab requires the resolved editable source root's `VERSION=2.3.1` and
  `source/isaaclab/config/extension.toml` package version `0.48.0` together.
  If distribution metadata exists it must match the extension, not the release.
  Other layouts are rejected until independently supported; cwd is not searched.
- The worker probes versions using the real image Python BEFORE wheel downloads
  and installation. `runtime-versions.log` is sealed into the existing result
  ZIP, including the traceback on failure. Teacher preflight checks again after
  dependency preparation, so an introduced conflicting/shadowing package fails.
- No upstream code, physics, data, image, training scope or budget was changed.
  CPU fixtures establish detection/rejection behavior only. The real worker's
  VERSION contents, CUDA/simulator startup, updates and reload remain unverified.

Authoritative fixed-tag references (read during this correction):

- https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/VERSION
- https://github.com/isaac-sim/IsaacLab/blob/v2.3.1/VERSION
- https://github.com/isaac-sim/IsaacLab/blob/v2.3.1/source/isaaclab/config/extension.toml
- https://github.com/isaac-sim/IsaacLab/blob/v2.3.1/source/isaaclab/isaaclab/app/app_launcher.py

### Latest correction — 2026-10-25 shallow-clone preflight

This section supersedes historical preparation status below. Two separately
authorized smoke attempts terminated before teacher/simulator startup. The
second task (`TASK_20261025_020`) failed because Flux clones with `--depth 1`
and the old ancestor check required an absent historical commit object.
No further GPU launch is authorized by this correction.

The corrected source preflight runs BEFORE input download/installation:

- Exact approved worker HEAD is required for a shallow checkout.
- The committed tree outside the added `flux/` directory must match the official
  PHP base byte-for-byte at the Git path/mode/blob-ID/gitlink level. All 156
  entries are covered. Canonical NUL-delimited tree SHA-256:
  `80aac341646b1a5f5016334ac0530d98ebd309694e97b1a76082da1364f5642e`.
- Preparation checks real full-history ancestry to PHP base
  `c7c9ce3fc75a8754a7e403c58de3b6933254d01b`; a full-history worker continues
  checking ancestry. Shallow workers verify pinned contents instead of fetching
  history or suppressing an error. Unapproved HEAD or tracked source changes fail.
- Holosoma gitlink and runtime checkout checks remain in place.
- `prepare.log` and `bootstrap.log` preserve child stdout AND stderr and are
  included in the existing result ZIP/manifest, even on nonzero child exit.
- `failure.json` records stack locations without exception values or arguments
  that could expose scoped upload URLs.

CPU tests cover the original exit-128 reproduction, approved shallow checkout,
full-history ancestry, dirty/changed/extra source rejection, gitlink changes,
early failure before downloads, and child traceback retention. These tests do
not establish IsaacSim, CUDA, native checkpoint reload or training compatibility.
No physics, motion coverage, terrain variants, environment count, update target,
runtime image or GPU budget was changed by this fix.

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
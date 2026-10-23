# HU_D04 WBT teacher on LimX Flux

## Status and provenance

This branch restores code, robot assets and 56 low-step motion/terrain pairs
from the user's existing Flux upload associated with `TASK_20261023_097`.
It does **not** claim that GPU simulation or training has succeeded.
The earlier task ran data/CPU checks and training `--help` only.

The original GitHub main branch imports a missing
`holosoma.config_values.wbt.hu_d04.experiment`. The restored upload instead
defines that experiment in `wbt_training/config_values/hu_d04/experiment.py`.
Three Holosoma Python changes plus HU_D04 robot assets are preserved under
`holosoma_overlay/`, without changing the upstream submodule pointer.
The launcher copies the pinned Holosoma package into `.flux-runtime/` and
applies this overlay there; the source submodule is not modified.

`recovered-manifest.json` records SHA-256 hashes of all recovered files.
`holosoma-base.json` checks selected source hashes from the pinned revision
`4a0bf04cde5930af5ed59da797568d32be972525`.

### Motion-data caveat

The recovered directory is labelled HU_D04 and its 29 joint / 32 body names
match the restored robot configuration. All 56 pairs have finite numeric arrays.
However, the original dataset README still describes **G1 source data**.
Names and shapes alone do not prove correct HU_D04 retargeting. Check reference
poses, joint limits and forward kinematics before interpreting training results.
The recovered preset also notes estimated/G1-derived armatures; these are not
validated HU_D04 physical parameters. Do not deploy this policy on a real robot
without separate validation.

## Read-only local source check

From the checkout root:

```bash
git submodule update --init thirdparty/holosoma
python flux/train_hud04.py --inspect
```

This validates source hashes and paired filenames only: no GPU, dependency
installation, task creation or training. Runtime data and config checks are
separate. Tests: `python -m unittest discover -s flux/tests -v`.

## Flux task configuration (not submitted by this branch)

- Code: GitHub repository `https://github.com/zhu7136/1021php.git`, this branch.
- Enable submodule fetching.
- Project: `HuD04_WBT` (`PRO_20261023_011`).
- Candidate GPU: 1 x 4090D 24 GB (`ESKU000001`). Availability and cost must
  be checked again before submission.
- Candidate image: IsaacSim 5.1 / IsaacLab 2.3.1, image `BJX00000170`,
  version `V000212`; previously used for the CPU preflight, **not** proven for training.
- Launcher expects `/isaac-sim/python.sh` in the container and refuses to
  silently fall back to system Python.

Flux startup scripts must begin with `gm-run`. Assuming Flux clones into
`1021php/`, use the following first for a configuration check:

```text
gm-run 1021php/flux/train_hud04.py --check-config --install-deps --time-limit-seconds 900
```

This step can incur GPU-instance charges even though it does not train.
`--install-deps` requires container access to the Python package registry and
working `python -m pip`. Packages are installed into `.flux-deps/`, with
`--no-deps` to avoid replacing the image's Torch/CUDA installation. The lock
reproduces the previous CPU preflight packages, not a GPU-tested environment;
runtime-only missing packages or IsaacSim conflicts may still need adjustment.

Only after explicit approval, a bounded smoke-training command would be:

```text
gm-run 1021php/flux/train_hud04.py --train --install-deps --iterations 10 --num-envs 128 --time-limit-seconds 900
```

The process limit includes setup. The launcher terminates its own child process
group on timeout. This is **not a platform billing guarantee**: Flux container
termination and task state must be verified afterwards. No automatic relaunch.
Smoke-test metrics are not a convergence result. Default logging is disabled;
checkpoints and training logs follow the framework's normal local output paths.
Confirm platform persistence/download before a long run.

For online W&B logging, add `--logger wandb` and securely inject the user's
`WANDB_API_KEY` via a supported Flux mechanism. The launcher records to
`zhul136-university-of-science-and-technology-of-china/HuD04_WBT` and never
embeds credentials. When provided by the runner, `WANDB_ARIA_THREAD_ID` and
`WANDB_ARIA_TURN_ID` are copied into run config `_wb_agent`.
The ARIA `github` and `key` secrets are not automatically injected into Flux.

## Security and next gate

No API keys, signed download URLs, preinstalled wheel trees, logs or trained
checkpoints are part of this change. The included motion bundle and meshes are
recovered input assets, not generated model weights. Original notices remain.
Before a GPU submission: review the branch, verify data provenance/kinematics,
confirm the image, resource, logging/persistence and a spend/time bound.
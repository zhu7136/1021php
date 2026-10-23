# HuD04 v8: ordinary Git data parts (no LFS)

This branch carries the exact approved 218231385-byte repair ZIP as five
ordinary Git blobs in `flux/data/hud04-v8-parts`. Four parts are 48 MiB and
the last is 16904793 bytes. Each is below GitHub's ordinary-file limit.

Select the dedicated `aria/hud04-git-data-v8-20261023` branch for a future,
separately approved Flux task. The command is:

```sh
gm-run 1021php/flux/git_smoke_v8.py --data-parts-dir flux/data/hud04-v8-parts
```

Relative parts-directory paths are relative to the repository root, not the
platform's current working directory. No personal storage mount is required.
The code and data must come from the same verified commit.

The bootstrap performs only local reads for the DATA stage: validate the
manifest, verify all five part sizes and SHA256 hashes, stream them in order
to a fresh archive, check the original immutable SHA256, and run the existing
all-56-motion manifest checks. It rejects LFS pointers, missing/reordered or
corrupted parts, unsafe names and symlinks. It does not overwrite existing
outputs or fall back to Drive/Release when validation fails.

Original ZIP SHA256:
`99c8225859128d27ec94dcc9ea7176e1257257926ae19900be40bebe4208d4e4`

The worker remains limited to 900 seconds and requests 32 environments and
two PPO iterations. Git cloning happens during platform startup, outside
that worker limit; slow cloning can still add time and platform charges.
No new paid GPU attempt is authorized by preparing or publishing this branch.
The prior Drive task remains terminated and must not be restarted.

Data no longer requires a worker-time network download, but this is NOT a
fully offline training environment: the existing pinned dependency-install
step can still need network access unless packages are already available.
Small-code cloning worked previously; large-data clone speed on Flux is not
yet proven. No PPO/simulator success is asserted by CPU preparation tests.

Binary data stays in Git history and adds roughly 208 MiB. Keep this branch
isolated; do not routinely update binary datasets through ordinary Git.
The local attributes explicitly disable LFS and text conversion for parts.

CPU regression suite (from `flux/tests`):

```sh
python -m unittest test_git_parts_source test_git_smoke test_download_diagnostics test_local_archive test_github_release_source test_google_drive_source -v
```
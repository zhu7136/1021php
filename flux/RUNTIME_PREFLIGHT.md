# HU_D04 runtime preflight — 2026-10-24

This is plumbing validation, not evidence of successful simulation or PPO.

## Verified without GPU

- The IsaacSim preset uses `hu_d04/HU_D04_01_train.urdf` from the runtime overlay.
- Its 32 unique mesh references exist as ordinary binary STL files, not LFS
  pointer stubs. Total mesh bytes: 17,968,138. Each binary STL byte count matches
  its triangle header. This does not certify geometry, collisions, or dynamics.
- `logger:disabled` exists in the pinned Holosoma logger registry.
- All five recursive submodules initialize at their pinned revisions after
  repairing `.gitmodules`; no submodule or pinned revision was removed.

## Launcher resolution

The old entrypoint required only `/isaac-sim/python.sh`. The failed remote run
reported Python packages under `/workspace/isaaclab/_isaac_sim/kit/python/`.
That observation does not establish which launcher scripts or aliases exist.

The entrypoint now checks these vendor launchers in order:

1. `/isaac-sim/python.sh`
2. `/workspace/isaaclab/_isaac_sim/python.sh`

`HUD04_ISAAC_PYTHON_SH` can specify one absolute vendor `python.sh` instead.
An invalid explicit override fails rather than silently selecting another.
The script must be executable and have a sibling `kit/python/bin/python3`.
There is no PATH/system-Python fallback. Resolution is logged when selected.

The smoke worker checks source integrity, required training meshes, and vendor
launcher existence before assembling candidates or installing wheels. This
preflight does not import IsaacSim, check CUDA, or launch the simulator.

## Remaining runtime boundaries

- Actual launcher existence in the remote image is not yet read back. The
  preflight will establish it on the next separately authorized task.
- CUDA, simulator imports, URDF conversion, collisions, PPO iterations, and
  persisted checkpoints still require a bounded remote test.
- LFS assets elsewhere in the repository were not globally recovered. The
  active HU_D04 overlay meshes above are present without LFS downloads.
- The alternate MuJoCo XML specifies `../meshes/HU_D04_01/`, which does not
  resolve its 44 mesh entries in this overlay layout. This audit did not alter
  that XML because the approved training path uses IsaacSim and its URDF.
- No new paid task is authorized by CPU validation or by this document.
# CPU-verified nested package path fix

## Evidence

Task149 passed Git input SHA256 checks, then failed before learner training in
the observation manager: wbt_training.config_values.motion_command:which_motion
could not be imported. The module exists in the pinned runtime. The overlay's
top-level package extends its path; its config_values subpackage did not, so
the regular subpackage hid the runtime module.

## Minimal correction

Before any wbt_training imports, extend the nested overlay package path with
pkgutil.extend_path. Only the exact known initializer is accepted; unexpected
versions, symlinks, missing upstream files and pre-imported packages fail closed.
Record before/after/upstream hashes in overlay-package-repair.json, which the
existing result packer includes. Do not rewrite the immutable input archives.

## CPU verification

17 tests passed: 8 package-repair regressions and 9 existing shard tests.
The fixture comes from the original pinned base archive. An isolated Python
process reproduces failed module discovery before repair and resolves the
runtime module after repair, while keeping both Hu-D04 overlay config modules.
AST confirms which_motion exists; the module body/IsaacSim was NOT executed.
Only the root and config_values packages overlap in this overlay.

No GPU execution or skill success is claimed. The simulator-close segfault in
task149 is NOT proven fixed. Clean native exit remains mandatory for acceptance.

## Unchanged scope

- Base archive SHA256: 490d7d1b8b927f42303816c1a40f440c05ef5393aec950724ab897fad45c643f.
- Data archive SHA256: f2e3c5de404b68b9ac1e6849e9d4f02aa5fcf88bc378cc45dc2b8865ac3aea79.
- 1x4090D24GB, 6 environments, 5 updates, native phase<=600s, task<=1800s,
  automatic retries=0. Any further GPU execution requires separate confirmation.
- Original failed task/run and current-page hud04-low-climb unchanged.
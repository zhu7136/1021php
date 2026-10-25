## TASK022 import-order repair

TASK_20261025_022 stopped before environment construction, checkpoint load,
PPO updates or evaluation. The native checkpoint config reader imported
Pydantic Core, which required Sentinel. IsaacSim's pip_prebundle version of
typing_extensions had already been cached; adding .flux-deps to sys.path did
not replace that cached module. The uploaded model files were copied inputs.

The continuation training child, continuation evaluation child, and standalone
paired-evaluation child now activate runtime_import_boundary before heavy
imports. It verifies typing_extensions.py byte-for-byte against the SHA256-pinned
4.16.0 offline wheel, selects that precise module even if an older module is
cached, verifies Sentinel and records provenance. Repeated activation before
other dependencies preserves the selected module's identity. Incompatible
already-loaded Pydantic trees are rejected rather than broadly reloaded.
Torch/CUDA/IsaacSim are never reinstalled or reloaded. No network fallback.

CPU regressions cover the original stale-cache ImportError, verified replacement,
Sentinel/source checks, idempotence, subsequent real CPU Torch/Pydantic imports,
preservation of already-loaded Torch, incompatible Pydantic rejection, corrupt
wheel/installation, missing installation, symlink rejection, and entrypoint order.
All 411 tests pass (12 new). Tests run on sandbox CPython3.13 with its installed
Torch/Pydantic, not IsaacSim CPython3.11: GPU startup compatibility still requires
the single separately authorized retry, and no GPU success is implied here.

The parent continuation's source model/optimizer, saved experiment recipe,
iteration201 start, 200-new-update OR300-second training limit, 1800-second worker
limit, three paired before/after seeds, continuous-reset censoring and physical
control/reward/termination settings are unchanged. A retry controller must use
new independent state and fresh signed capabilities, never replay TASK022.
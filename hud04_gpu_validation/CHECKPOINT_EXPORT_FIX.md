# CPU-tested native checkpoint export fix (no GPU run)

Task151 completed five native updates after the import repair, then failed
because final save targeted the output root outside logging_helper.log_dir.
The process later segfaulted. No valid final checkpoint was recovered.

## Correction

Use native algo.save(path=...) strictly inside the timestamped native log
directory, then native algo.load(...) from that file. Require exact equality
of model tensors, all four optimizer groups/slots, iteration, learning rate,
PPO coefficient and teacher readiness. Only after successful reload, copy the
verified bytes via an exclusive pending file into the bounded output and
verify SHA256/readback before exposing the final checkpoint name. Existing
destinations, linked paths, oversized files, changed data and state mismatches
fail closed. Immutable input archives and native logging path guard unchanged.

The native adapter records checkpoint-export.json; existing worker packing
already includes that JSON and the final validation-only checkpoint. Model,
optimizer, teacher-freezing, three-route, finite-state and clean-exit acceptance
requirements are NOT weakened. Training scope and algorithm are unchanged.

## CPU tests and limits of evidence

- 14 new persistence tests use real CPU tensors and Adam with four parameter
  groups. The actual pinned logging helper method is extracted and compiled
  alone, preserving its path guard without importing the simulator.
- 4 Python-only lifecycle tests ensure SystemExit and ordinary/close exceptions
  propagate and cannot set the child-success marker after a failure.
- The previous 8 overlay and 9 shard regressions are rerun (35 total).
- The worker now records train_exception_type and SystemExit code before real
  cleanup. No forced exit, skipped cleanup or segfault suppression is added.
- These tests do NOT execute native IsaacSim save/load/teardown on GPU and do
  NOT prove the observed segmentation fault fixed. Native clean process exit
  remains a mandatory check in any future separately authorized validation.

## Preserved boundaries

No new GPU task or retry. Task151's five-update evidence and failed validation
remain intact. Current-page hud04-low-climb checkpoint and all teacher weights
unchanged. Proposed future validation, only after explicit approval: one
4090D24GB,6env,5updates,native<=600s,total<=1800s,retries0.
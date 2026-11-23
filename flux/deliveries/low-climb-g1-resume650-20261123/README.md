Hu-D04 low-climb G1 continuation, authorized 2026-11-23T00:58:12Z.

Resume TASK_20261122_086 iteration649 / 650 completed PPO updates.
Source checkpoint SHA256 ec7d0fb9e7e10e8758a0d8936680a28508e9cda778157904b8f56b34c6f4e18e.
One 4090D24GB, 4096 environments, seed42, same40clips/32814frames.
Unchanged all29 G1 simulation velocity limits, head13rad/s, PD/effort/geometry,
observations, actions, rewards, termination and randomization configuration.
Exact actor/critic/optimizer/learning-rate verification; no random-init fallback.
Restarts simulator and RNG; not a bitwise trajectory continuation.

Worker cap1800s includes input reconstruction/dependencies, save and result seal.
Soft stop after1500s, native child cutoff1670s, at most2000 additional updates.
Queue, clone/allocation and release may add billable time outside worker cap.
Zero automatic retries; no other GPU tasks authorized by this package.

Public code/input/robot assets/checkpoint delivery authorized for this trial.
No account credentials, signed URLs or raw logs are included in Git or results.
Original full training data/runtime/checkpoint shards are reused from the pinned
parent commit. The single-clip replay input is NOT used for training.
Task-private log carries a checksummed selected-result archive; readback is
required before claiming recovered weights. W&B status record completion is
not evidence of external training completion, convergence or task success.

Default entry only checks package hashes/config on CPU; --execute starts GPU.
Native lifecycle close may still fail as observed in source training; report
learning and lifecycle evidence separately. Preserve main and existing tasks.
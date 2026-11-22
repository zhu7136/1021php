## Frozen low-climb teacher 660 native/MuJoCo comparison

Authorized delivery: original iteration-660 checkpoint and the first failing
training clip, as checksummed Git shards. No Flux private storage is required.
Public publication was authorized on 2026-11-22 at 11:57:44 UTC.

Scope: one task, one 4090D 24GB, worker at most 900 seconds, zero retries,
zero training updates. One environment, reference frame zero, no initial
settling step, no observation noise, no domain randomization or auto-reset.
Uses deterministic mean actor inference, original robot/drive configuration,
translated source terrain, policy 50 Hz / physics 200 Hz. At most 799 control
intervals spanning the 800-frame, 15.98-second reference clip.

CPU preflight does not validate native physics or certify a locomotion skill.
The child writes state/action/observation/contact evidence before any reset.
On termination, the reset is intercepted and the original terminal state kept.
Lifecycle failure is reported separately from policy/trajectory evidence.

The worker reuses the pinned sibling runtime shards from
`locomotion-g1all-fresh-20261122/inputs`, but never executes its training code.
It does not accept provider keys, signed URLs or private mount paths.
Results are checksummed and mirrored into the task log. Actual download,
archive reconstruction and member-hash verification are required before
claiming that remote results were retrieved.

The worker's 900-second wall limit begins at entry; platform queueing, Git
checkout, allocation and release are outside that limit. Child has an earlier
800-second deadline to leave time for output sealing.
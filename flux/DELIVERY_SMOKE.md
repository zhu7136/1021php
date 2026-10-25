## One-update GPU delivery smoke (2026-10-25)

### Shell-safe transport repair and separately authorized verification

TASK_20261025_030 terminated before the custom entrypoint: raw signed URL
query ampersands escaped the platform's nested shell quoting and START_SCRIPT
became empty. No PPO update or output checkpoint was observed. At
2026-10-25T03:24:08Z the user authorized publishing this repair and ONE new
verification with the same bounds,zero evaluations,and no automatic task retry.

Use `launch_envelope.start_script` to build the new controller payload. It
passes one URL-safe base64 token containing freshly minted signed capabilities.
The whole command has no quotes,ampersands,dollar signs,semicolons,or newlines.
The worker decodes directly into Python argv;no decoded data enters a shell.
Encoding is NOT encryption: the token remains confidential,expires,may appear
in platform logs,and must never be committed or displayed in shared output.
Neither W&B nor Flux nor GitHub API credentials are included in the envelope.

477 CPU tests pass,including9 new nested-shell transport tests. Single and
double quoted START_SCRIPT assignment and multiple shell passes preserve every
synthetic signed URL byte. This is CPU evidence,not proof of Flux execution,
remote network connectivity,or a new checkpoint. New GPU validation must use
a new W&B bookkeeping run/task/state;never replay TASK030 capabilities.

### Original single-update contract (unchanged)

The user authorized starting this single bounded verification directly at
2026-10-25T03:07:28Z. This is not authorization for 200-update continuation.

Use `train_delivery_smoke.py` as BOTH the gm-run parent entrypoint and the
IsaacSim runtime-child entrypoint. It configures one training arm, one update,
zero evaluation arms,32 environments,source iteration200,next iteration201,
300s training failsafe,1800s worker failsafe,no automatic task retry. Clone and
platform initialization occur before the worker timer and may add billed time.
The generic native PPO wrapper also stops after its first completed epoch,so
the cap does not depend only on the native loop configuration. PPO math,
rollout length24,source weights/optimizer,physical/reward/termination recipe
are unchanged. Exact simulator/RNG trajectory continuation is not claimed.

Before simulation, the signed JSON probe must PUT/GET with matching SHA256.
After one update, native model/optimizer/config save+reload must be exact,
and the final native iteration must be201. Upload the new checkpoint before
summaries or the ZIP. When GCS confirms storage MD5 but no new-file signed GET
exists yet,explicitly require the controller's independent W&B download SHA
and native checkpoint iteration/optimizer check. Source input files are never
accepted as newly generated output. See OUTPUT_DELIVERY_REPAIR.md for protocol.

468 CPU tests pass,including9 new one-update wiring/stop-hook/import-boundary/
summary-gate tests. A deliberately hundreds-of-updates fake native loop is
stopped after exactly1 completed update. CPU tests are not GPU validation.
This smoke tests plumbing only;no reward/quality/deployment conclusion follows.
## One-update GPU delivery smoke (2026-10-25)

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
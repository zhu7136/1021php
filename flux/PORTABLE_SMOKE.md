# Portable checkpoint delivery smoke (one authorized GPU verification)

- Approval: 2026-10-25T04:51:59Z;one new Flux task only,no automatic task/GPU retries.
- 1x4090D24GB,32 environments,exactly one PPO update200→201,zero evaluations,300s training and1800s worker failsafes.
- Uses `train_portable_smoke.py` through the existing URL-base64 shell-safe launch envelope.
- Re-generates from the previously validated iteration200 native checkpoint;does NOT recover the old TASK_20261025_032 iteration201 weights,and does not claim policy quality or long-run continuation success.
- Produces the normal native checkpoint first,then preserves every native field and adds `_aria_delivery` provenance in `outputs/portable_smoke/platform_exports/model_00201.pt`.
- Uses the platform-observed `model_<iteration>.pt` convention,distinct from input `model_final.pt` and `model_preflight.pt`. Exact task ID and embedded W&B run path bind the export to this run.
- Saves a non-`.pt` partial,CPU-loads and compares all native fields,then invokes the native live algorithm loader and checks exact actor,critic,optimizer and iteration before atomically linking the visible `.pt`. Refuses overwriting any destination.
- Controller retrieves only this exact task's `model_00201.pt`,not `loadRun=inputs`,through its documented platform model-download URL. Validates native iteration201,exact run/source/one-update provenance,populated optimizer moments and finite tensors. Any worker-reported export digest,when obtainable,must also match.
- Verified output is uploaded by the controller to the same new W&B run and independently downloaded/rehashed before declaring checkpoint delivery passed. Worker-direct signed PUT receipts alone are NOT proof of retrievability.
- API credentials never enter the worker payload. Signed URLs/envelopes are capabilities and must be redacted from all public logs.
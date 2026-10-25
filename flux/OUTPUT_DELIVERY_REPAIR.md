## TASK025 output-delivery repair — CPU validation only

The previous continuation's iteration400 checkpoint is not recoverable from the
verified outputs. Its final bundle PUT failed with URLError, and the legacy
logger did not retain the underlying reason. That is not enough evidence to
identify the Flux worker's network root cause. Do not claim it is fixed remotely.
Use only the preserved TASK097 iteration200 source checkpoint for a future retry.

### What changed

- Before dependency bootstrap, simulator startup or PPO, the worker PUTs a small
  JSON probe and downloads it through a distinct signed GET capability. Byte
  length and SHA256 must match, otherwise execution stops without GPU simulation.
  An allocated Flux GPU instance may still incur startup/minimum charges.
- On training-stage completion, upload the NEW checkpoint independently as
  `resume_trained_checkpoint.pt` BEFORE any paired evaluation or final ZIP.
  Upload training summary, each evaluation summary, and final summary separately.
- Signed PUT and GET URLs are strictly scoped to the exact run and output path.
  No API key is forwarded; redirects and credential-bearing headers are rejected.
  Output files cannot be symlinks or escape the output root. Upload/download byte
  bounds apply. Failures retain local checkpoint files; no file deletion.
- Log HTTP status, exception/reason types, errno, phase, output and attempt. Do
  not print arbitrary exception messages or signed URLs. Only a transient 5xx,
  timeout or connection failure gets one upload retry; NEVER retry training.
- A new object does not yet have a signed W&B GET URL. The controller seeds ONLY
  the JSON probe through the SDK (not a dummy `.pt`), then obtains its signed GET.
  Unwritten checkpoint/summary destinations use GCS's returned MD5/ETag to confirm
  stored bytes, and explicitly log `controller_download_required: true` and
  `download_sha256_verified: false`. An upload ACK alone is NOT full verification.
- The controller MUST then download `resume_trained_checkpoint.pt` with W&B,
  compare SHA256 to the worker receipt, and check native iteration, actor/critic,
  populated optimizer state and saved config. `download_checkpoint` implements
  this read-back gate. Presence checks do not imply optimizer equivalence; the
  existing native save/reload roundtrip checks exact state before uploading.
  Never authorize a larger training run until that gate succeeds.

### Controller integration for a future authorized task

Use a NEW run, independent task state and fresh signed capabilities. Call
`prepare_delivery_manifest.prepare(api, run_path)` in the trusted controller.
Keep the returned manifest private and pass it as `--delivery-manifest-json` to
the worker along with the existing input capability and `--wandb-run-path`.
The previous upload-only CLI is disabled. Do NOT replay an old Flux payload or
call `submit_resume.py` unchanged: it does not pass the new manifest.

No checkpoint, reward, physical control, termination or PPO hyperparameters were
changed here. The worker still has its previous 200-update/300-second/1800-second
upper bounds; **this is not yet the one-update smoke-test controller**. Before
any GPU submission, prepare and CPU-test that narrower entrypoint/controller,
explicitly authorize one Flux task (1x4090D24GB,32env,one update from iter200,
no full paired evaluation,no automatic task retry), and verify the output gate.
No task has been submitted by this patch.

### Evidence and limitations

- 459 Flux CPU regressions passed, including 42 new delivery/controller tests.
- Real W&B/GCS probe run: `HuD04_WBT/67b925f6`, 125 bytes,
  SHA256 `97432381153f241f96845225b338449fbeeb727f9f8ecb805889cdb1c5733129`.
  Signed PUT, returned storage MD5, signed GET and independent SDK download
  matched; run summary read back from the server. No model was generated.
- This proves the sandbox-to-W&B path, NOT the Flux-to-GCS path. No Flux task,
  GPU simulator, PPO update, iteration201 checkpoint or GPU evaluation ran here.
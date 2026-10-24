## HU_D04 bounded startup diagnosis

This revision adds observability, not a claimed simulator fix. The remote job
keeps the verified v8 candidates, 32 environments, two PPO iterations, existing
Isaac image, CUDA device selection, and 900-second outer worker limit.

`train_hud04.py` installs `startup_diagnostics.py` before entering training.
The actual Holosoma Isaac launcher is wrapped with a 300-second CPython native
faulthandler watchdog. A timeout prints all Python stacks and exits that runtime
process even if a native extension holds the GIL. Success cancels the watchdog
and proceeds directly into the same training execution; no second paid task is
created. Cold initialization exceeding five minutes can therefore fail this
diagnostic even if it might eventually succeed.

Structured console events mark launcher completion, environment/algorithm
construction, first reset/step, learn entry/return, and checkpoint-save return.
A checkpoint event reports actual file existence and size, not training quality.
The runtime emits heartbeats every 15 seconds and stack snapshots every minute
while its Python thread can run. An independent parent emits heartbeats and
mirrors bounded chunks of current-job Kit and training logs even if the runtime
child stalls. Runtime and parent mirrors can duplicate a small amount of log
text. URL strings and credential assignments are redacted. No secrets are
injected into this training job by these changes.

The platform SDK's own global file-watcher output is outside this module's
control, so platform log truncation remains possible. Missing phase markers in
a truncated export are not proof of failed phases. No automatic retry, renderer
change, driver change, or CPU simulator fallback is introduced.

The 900-second worker deadline does not include platform startup and Git clone
time and is not a hard billing limit. These smoke results do not validate
collisions, torques, dynamics, or policy quality.

CPU regression command:
`python -m unittest discover -s flux/tests -v`
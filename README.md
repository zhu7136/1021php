# HuD04 r28: corrected bounded empty-application close comparison

This is the isolated r28 follow-up, not a relaunch of the r27 task. The runtime
event prefix remains HUD04_R27_EVENT for parser compatibility; exact task ID,
commit and ARIA provenance identify the new execution. Fixes the event(name)
collision with GPU field name and records bounded first-error messages and
source-frame metadata (no local variables). These are diagnostic fixes, not a
shutdown repair. Twenty-five CPU protocol tests passed before this submission;
fake CUDA/simulator objects do not establish a real runtime result.

One necessary GPU lifecycle diagnostic under the project's standing validation
authorization. Uses the previously verified Flux image BJX00000170/V000212,
one catalog 4090D24GB, worker360s from entry, real close30s, zero retries.
Queue/clone/pull/release time is not included in the worker bound; catalog
price alone does not establish currency, billing unit or a total bill cap.

No robot, dataset, wheel, checkpoint, PPO, account key, private input,
signed storage ticket, installation or remote-model call is included.
The default entry is read-only. Execution requires exact commit/task/ARIA
provenance. The fresh child receives only an allowlist of runtime settings.

Starts a headless AppLauncher, fingerprints the actual runtime close function
and SIGSEGV callback source, checks the reconciled RTX4090-family device
contract, executes tiny CUDA arithmetic and eight empty-app updates, and
calls the real app.close(wait_for_replicator=False). Disables and reads back
fastShutdown as in the r5 baseline. No Holosoma USD/STOP patch is installed
in this empty-app comparison. The same close_observer interface now wraps
the r26 close_profile candidate. Neither signal handlers nor app.close are
replaced, and no signal is sent to the simulator for the experiment.

The parent requires all lifecycle markers, real close return, natural child
exit0, completed profiling and a recorded actual close call. Forced cleanup
does not pass. Only the newly started process group is managed. Five-second
Python stack samples are not native C/CUDA stacks. At most64 call events
are recorded in the current Python thread; missing events cannot exclude
native failure. The journal is read after the child exits or times out.

Empty-app success does not repair or verify the HuD04 robot/PPO workload.
Outputs are this task's logs, with bounded JSON events mirrored for retrieval.
There is no model and no W&B training history to sync.

supervisor_base.py and runtime_versions.py are unchanged from the earlier
authorized r5 diagnostic. PHP source notices and license are retained.
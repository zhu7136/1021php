# HuD04 r29: pinned shutdown-helper-only comparison

One bounded diagnostic, not PPO training. Relative to successful r28, retain
image BJX00000170/V000212, catalog single 4090D-family GPU, eight empty-app
updates, tiny CUDA check, fastShutdown=false/readback, the same observer,
profile, supervisor, runtime versions, and natural-exit requirements.
Worker360s, native close30s, automatic retries0. Queue/clone/pull/release are
not within worker bound; catalog5.4 has no independently confirmed unit,
currency or total bill cap. Standing necessary-GPU validation scope only.

Replace the final direct close adapter with the exact frozen Holosoma
close_simulation_app function text. The full Holosoma module is NOT imported:
its unrelated imports are deliberately avoided. A standard-library logger
and an ISAACSIM-only enum/getter are explicit minimal global bindings.
Function-text SHA256 is16c60deb4d5fb77a7f524e02085039e15b95209c56dbcfd1a722f6ada9eb9801.
The source function installs its original USD close_stage no-op workaround,
optionally disables SimulationContext STOP handling if an instance exists,
and still calls the real app.close. Actual function source is fingerprinted;
context presence and post-return patch effects are recorded.

An empty app may have no SimulationContext; in that case this does NOT test
the STOP-callback workaround in a training scene. No claim of full-module,
robot, environment/PPO cleanup or native-crash root-cause equivalence.
Thirty-three CPU tests passed with mocks; they are not GPU evidence.

No robot, motion, checkpoint, wheel, credential, signed URL, installation,
PPO, remote model call or W&B training history is included. Public isolated
branch does not change main. Task logs are the owned result channel.
Default entry is read-only; execution requires task/commit/ARIA provenance.
HUD04_R27_EVENT prefix retained for parsing compatibility; identify by task
and commit, not this prefix. Original Amazon license/notice retained.
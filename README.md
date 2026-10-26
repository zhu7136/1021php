# r13: same-image physics context plus procedural ground, NO ROBOT

One diagnostic task on existing image BJX00000170/V000212. Not PPO, not an
IsaacLab PHP environment, not proof that the current 31-joint robot trains.
One GPU, 64 steps dt=.005, worker360s from Python entry; shared cleanup+GC+close
deadline30s. Queue/image pull/Git clone/resource release billing lies outside
the Python bound. Zero automatic retries. No install or dependency overlay.

Before constructing physics, derive actual module files from the executing
SimulationContext/AssetBase/SimulationManager classes and verify all four
file hashes against reviewed official IsaacLab-v2.3.1/IsaacSim-v5.1.0 sources.
No argv-selected source substitute; no source mismatch waiver.

Relative to failed r12, remove all external robot assets and Articulation,
retain a SimulationContext, procedural flat ground, reset,64 steps, the
previous exact-source-gated temporary STOP guard (restore finally), real
stop/clear callbacks/clear instance. Read public instance() afterward.
Unwind physics scope and collect cycles; observe sim weakref only, not an
unsubstantiated requirement that this reference be dead. Call true app.close
with fastShutdown=False, require close return AND natural exit0, no segfault.
No private singleton overwrite, callback suppression, skip-close or exit(0).

Pre-registered interpretation:
- Crash after context and 64 steps: context/framework/image cleanup is also
  affected without the robot. This is not proof of driver or native root cause.
- Natural exit0 with full evidence: this baseline works; robot asset/Articulation
  lifecycle remains the next variable. Full robot is NOT retrospectively passed.
- Source mismatch/import/init failure: precondition failure, not a close test.

Resource diagnostic contract remains one NVIDIA4090-family23..25GiB CC8.9;
catalog4090D vs observed4090 identity discrepancy unresolved and not formal
training acceptance. No head/robot/tracking metric is produced in this control.
No wheels, robot assets, motions, weights, credentials, or signed links shipped.
Default command is read-only; execution requires fixed commit and task/provenance.

Official source references (reviewed fixed tags; raw files not executed here):
- github.com/isaac-sim/IsaacLab tag v2.3.1:
  source/isaaclab/isaaclab/sim/simulation_context.py
  source/isaaclab/isaaclab/assets/asset_base.py
- github.com/isaac-sim/IsaacSim tag v5.1.0:
  source/extensions/isaacsim.core.api/python/impl/simulation_context/simulation_context.py
  source/extensions/isaacsim.core.simulation_manager/python/impl/simulation_manager.py

CPU tests are protocol/API fixtures only, not native simulator or GPU evidence.
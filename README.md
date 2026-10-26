# HuD04 current31 source-pinned guarded-stop diagnostic (r11)

This branch is a plumbing diagnostic, NOT a PHP environment or training run.
Default `robot_probe.py` is read-only. Execution requires fixed commit, Flux task
and ARIA provenance binding; supervisor 360s from Python entry, shared physical
cleanup + close deadline 30s (not reset upon app.close),
one catalog 4090D24GB task, no automatic retry. Queue/pull/release billing is
outside the Python bound. No install or change to the target image is performed.

The previous r7 resource-name failure remains historical fact. This diagnostic
explicitly permits one RTX4090 or RTX4090D family device, 23..25GiB, CC8.9.
The actual name and catalog-name mismatch remain separately recorded; this does
NOT resolve catalog/billing identity and is not acceptance for formal training.

`input.zip` reuses, byte-for-byte, the already public Git blob
dfb868fc60267117c953ebf47223f4344bf65a35 from repository zhu7136/1021php commit
0da8a7f6024918073c8d37c9fa5e3e536d5e3054, flux/hud04-git-input/input.zip.
SHA256 ee5aeac12664ba16e7a5298642c88aeb9e544246331f3524f57d73f88d52c433.
Only 39 exact manifest robot assets are extracted and individually hashed.
No archive Python/overlay or motion/pickle is executed/loaded. No new wheels,
weights, credentials or previously private asset bytes are published.

`hud04_robot.py` is the existing current31 calibration module, unchanged from
the verified local source snapshot. Calibration is simulation-only, not a
certified hardware specification. The floating-base robot uses these existing
implicit PD gains/limits/armatures and initial joint angles. One robot, 64
physics steps at dt=.005 (0.32s), self-collision enabled, flat ground.
All 31 physical names must match exactly; 29 non-head policy indices are
resolved, but no policy is loaded and no policy actions are generated.
The head target remains zero, its observed maximum displacement is recorded
and checked against the preregistered 0.05rad short-probe threshold. All root,
joint-position and joint-velocity states must be finite. Record root height;
no claim of standing, reference feasibility, complete contact validation,
task success or PPO learning follows from this short diagnostic.

Success additionally requires genuine physics cleanup, app.close return,
natural child exit0 and parent evidence checks. Python error traceback is
flushed before cleanup. Native hang/crash is a failure even if partial checks
passed. NumPy<2 and complete PHP dependencies remain a separate unverified
target-overlay gate; this no-install diagnostic does not accept them.

Change vs r9: guard final sim.stop using the existing IsaacLab2.3.1
_disable_app_control_on_stop_handle guard, the same flag its reset uses.
Correction vs r10: ast.dump omits empty fields on local Python3.13 but not
target3.11. r10 exact source bytes matched upstream yet the AST dump hash
rejected, so the guard was never entered. Now derive actual source from
executing class and require FULL source SHA256 pinned to both official v2.3.1
and r10 actual runtime, plus reset guard pattern; unknown bytes/state fail
closed. No AST-dump dependency or fallback hash. Restore original guard after
stop, including exceptional paths.
This is a private version-specific adaptation, not a stable public API.
No callback/render monkeypatch, no skipped stop/cleanup/close, no exit(0)
fallback. Physics/assets/gains/64steps/thresholds/image/resource contract and
per-call events/5s Python stacks/shared30s cleanup+close are unchanged.
Natural childexit0 plus actual stop/clear/close returns and guard evidence
are required. Full31 robot physics/lifecycle only, NOT full PHP environment,
reference feasibility, dependency overlay acceptance, PPO or convergence.
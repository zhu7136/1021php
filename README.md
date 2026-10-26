# HuD04 r5: bounded empty-app GPU diagnostic

This isolated branch contains only a diagnostic, runtime-version checker,
CPU supervisor tests, source manifest and notices. It contains no robot,
motions, wheels, checkpoints, account credentials or signed storage URLs.

The diagnostic targets the already-used Flux image IsaacSim5.1/IsaacLab2.3.1,
Python3.11/Torch2.7 on one4090D24GB. One task, zero retries; worker360s from
entry, close30s, owned child termination reserved within that worker budget.
Clone/image-pull/queue/release time is outside the worker bound. The catalog
price field alone does not establish billing currency/unit or a full fee cap.

It checks anchored runtime versions and real CUDA arithmetic, starts a
headless AppLauncher, disables and reads back fastShutdown, runs eight empty
application updates and calls app.close(wait_for_replicator=False) directly.
No Holosoma USD/STOP monkeypatch and no robot SimulationContext are created.
Close is sampled by Python faulthandler every5s. The parent requires all
markers AND zero child exit; forced termination never passes. The application
may hang and be stopped; this is diagnostic evidence, not a repair claim.

Outputs are newly-owned child logs and JSON, mirrored in Flux task logs.
No result-upload ticket or private dependency delivery is needed for this
narrow test. That does NOT solve full HuD04 delivery, dependencies, PPO,
checkpoint transport, dynamic feasibility or original PHP task alignment.
An empty-app close pass would not prove the robot/physics workload closes.

Default invocation is read-only. Actual execution additionally binds the
exact Git commit, task ID and ARIA provenance. Secret names are not values;
only an allowlist of non-account runtime environment settings reaches the
child. Public startScript contains identifiers, never credentials.

Standing authorization: project user stated2026-10-26 that necessary GPU
validation may proceed without per-test approval. This is a single bounded
runtime test under that scope, not formal training authorization.

CPU tests use fresh owned subprocesses and synthetic markers; they do not
import IsaacSim, initialize CUDA or claim native training evidence.

runtime_versions.py is retained unchanged from PHP commit
0da8a7f6024918073c8d37c9fa5e3e536d5e3054,
blob16c310461178eb3a988c4a24dff8a3283ea6d2f4.
PHP LICENSE/NOTICE accompany the original source.

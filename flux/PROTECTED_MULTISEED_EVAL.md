High-level protocol: TASK097 frozen-model paired evaluation
=========================================================

This entrypoint performs no training and writes no new checkpoint. It compares
TASK_20261024_097 model_preflight.pt (iteration 0) with model_final.pt (iteration
200), identified by the SHA256 constants in evaluate_protected_multiseed.py.

One explicitly authorized Flux task, one 4090D 24GB, 32 environments, seeds
1402/2402/3402, 128 control steps per model/seed. All six model evaluations run
sequentially in fresh simulator processes to avoid carrying domain-randomization
episode counters or physics state between models. Environment construction seed
1401 and TASK097 physical configuration stay fixed; the three seeds govern
evaluation resets and observations. There are zero PPO updates, optimizer state
is not loaded, optimizer.step/learn/save are explicitly disabled, actor weights
and input checkpoint hashes are checked unchanged. Native checkpoint loading
still supplies iteration and actor/critic state; neither input has env_state.

Worker wall-clock cap is 1800 seconds INCLUDING bootstrap, six simulator
startups, evaluation, and output sealing. Platform startup and repository clone
are outside this cap. There is no retry, replacement task, or budget extension.
Any failed arm stops the task; diagnostics for completed arms are still sealed
if the hard deadline permits. This is not a model-training recipe.

Uses the already validated tensor-boundary wrapper and deterministic mean-policy
evaluate() implementation. Uniform reference-frame sampling; observation/reset
noise and existing physical randomization remain enabled. Each seed's initial
motion frames, joint positions/velocities, root states, domain-randomization
episode counters, and returned observations must match exactly across models
before an aggregate result is accepted. Non-matching pairs are not silently
treated as controlled experiments. Failing this check is an evaluation failure,
not evidence about either model's quality; individual arm files are retained.

Read-only diagnostics run after native termination evaluation but before
automatic resets. They retain only surviving first episodes and record each
first failure step/time, native termination flags, root-position/root-orientation
flags, and native 3-D tracked-body errors at the existing thresholds. Native
stateful termination __call__ methods are never invoked twice. Native applied
target readback and hard-limit assertions remain enforced. Reset warm-up and
later episodes cannot contaminate first-episode tracking summaries.

Inputs arrive via a fresh W&B-scoped signed GCS GET. The exact 13,513,018-byte
source ZIP hash and both model hashes are checked before loading. Only those
two exact archive members are materialized; no general extractall. Results go
to protected_multiseed_eval_bundle.zip via a scoped signed GCS PUT. Worker code
receives no W&B/GitHub/Flux API credentials, only file-scoped signed capabilities.

CPU tests exercise pairing failures, bounds, frozen-model calls, target guard
assertions, pre-reset attribution, snapshot copies, and warm-up/dead-env masks.
They do not prove actual GPU behavior. Three seeds and a 2.56-second horizon
are a small, censored evaluation, not convergence or deployment success proof.
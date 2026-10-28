# Hu-D04 four-skill validation via ordinary Git parts

This branch contains a **validation-only**, zero-PPO worker. Public package
publication was explicitly authorized. Publishing does **not** authorize a new
GPU task. Do not reuse previously consumed task approvals or dispatch scripts.

## Entry point

Use **`bootstrap_git.py`**, not the inherited HTTPS-input `bootstrap.py`.
The main-code name, main-code URI, and start script must all select this entry
point. Pin the exact new commit via `--commit`; the worker checks checkout HEAD.

```text
gm-run 1021php/bootstrap_git.py --execute --task-id <new-task-id> --commit <this-commit> --output-cap <fresh-scoped-output-capability> --thread-id <current-thread> --turn-id <current-turn>
```

Task IDs and output capabilities are fresh runtime parameters, never public
repository content. Only a single-object signed PUT capability is needed; no
input capability or account credential is forwarded. Avoid printing capability
arguments in shared logs; platform command visibility is a separate concern.

## Input verification

- Three **ordinary Git blobs**, not LFS pointers, under
  `git-input/hud04-validation-v3/`.
- Per-part size/order/SHA256, total size, and pinned whole-archive SHA256 checked
  before safe extraction and verification of **601 payload files**.
- Input archive: **116980683 bytes**; SHA256:
  `9eb6f16d5f5630f263d67d6d0e084724d7a2d3ac4836ce7b448c658c6d435e4f`.
- No HTTPS input download or fallback route. Output remains a bounded signed PUT.
- Offline dependency install remains `--no-index --no-deps --require-hashes`.

## Workload and bounds

Same unchanged four complete representative motion clips / **11 reset probes**,
at most25 control steps per probe, one environment; 31 physical joints,
29 actions, 42 reference bodies. No learner construction, PPO updates, training
or checkpoint. Native child still calls **unpatched `app.close()`** directly.
Pass requires all11 probes/four cases and normal close return.

Deadlines relative to worker Python entry: input assembly/extraction120s,
install210s, native child515s, output upload590s; watchdog exits598s. No automatic
retry. These bounds **do not include platform Git clone / image startup** before
Python runs; platform billing/time behavior for that overhead is not established.

## CPU evidence only

`git_worker_cpu_report.json` records14 passing CPU integration checks, including
an independent public sparse Git clone and actual archive assembly/extraction
with all601 payload hashes checked. Installer/native simulation stages in the
control-flow tests were **mocked**, not run. An additional prior11-case assembly
regression suite is recorded in `git_parts_cpu_report.json`.

This does **not** prove the GPU worker can clone in its network environment,
does not validate native physics, and does not certify all196 motions. Four
fresh teachers have not been trained by this validation workflow.
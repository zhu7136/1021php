# High-speed input delivery fix (CPU validated; not a GPU retry)

TASK296 failed before training: fetch_data premature EOF ('Truncated high-step input').
This isolated branch changes high-step data transfer and pinned292 checkpoint transfer only.
Low-step/old entry points, PPO recipe, source292 pins, 4096 environments, native restore gate,
3600-second TOTAL learning budget and5100-second worker envelope are unchanged.

Both transfers now publish a file atomically ONLY after exact expected size and SHA256.
Transient truncation/reset/timeout/HTTP429/5xx may retry the individual GET twice
(3total attempts), sharing a180-second transfer deadline and15-second socket timeout.
Deadline checked between reads; an in-progress read can last up to its socket timeout.
This never creates/retries/restarts a Flux task, increases a worker budget, or authorizes GPU spend.
Hash mismatch, oversize, untrusted host/redirect, auth failure and404 fail closed without retry.
Incomplete .part files are retained, not consumed/extracted; existing destinations cannot be overwritten.
Structured JSON reports contain attempt count, received/expected bytes, sanitized outcome and elapsed
time, but no URL/credential. They are included in existing result-manifest packaging.

Tests: `python -m unittest flux.test_verified_download` simulates the EXACT296 early-EOF
failure and verifies successful recovery, failure after3attempts, integrity/host/auth rejection,
deadline/no-clobber guards, actual high-step extraction and checkpoint download integration.
`python -m unittest discover -s flux -p 'test_*.py'` checks the full CPU suite after commit
(source preflight intentionally rejects uncommitted tracked worker changes).

No paid Flux task may be submitted from this branch without new explicit user authorization.
Old /workspace/g1_resume292_ops.py dispatch stays pinned to296's old commit and run_attempted guard;
it must NEVER be reset/reused. A newly approved attempt needs a new unique controller/state,
fresh input/output URLs and verified remote commit, while preserving the292 checkpoint hash.
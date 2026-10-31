# Hu-D04 three-teacher inputs: ordinary Git shards

The owner explicitly authorized public publication of both original input
archives, including their four weight files, on an independent branch on
2026-10-31. This authorization does NOT authorize another GPU run.

## Delivery

`git-inputs/manifest.json` binds 13 ordinary Git blobs, at most 48 MiB each:

| Input | Bytes | Parts | Complete SHA-256 |
|---|---:|---:|---|
| base.zip | 123410856 | 3 | 490d7d1b8b927f42303816c1a40f440c05ef5393aec950724ab897fad45c643f |
| data.zip | 470500407 | 10 | f2e3c5de404b68b9ac1e6849e9d4f02aa5fcf88bc378cc45dc2b8865ac3aea79 |

Use ordinary Git, not LFS pointers or GitHub Release asset downloads.
The existing `three_teacher_parent.py` entry now reassembles both inputs
locally and verifies part names, order, size, per-part SHA-256, complete input
SHA-256 and the fixed input identities before extracting or training. There is
no runtime input GET or silent alternate-data fallback.

CPU-only reconstruction (from the checkout root):

```bash
python hud04_gpu_validation/git_shards.py restore \
  --directory hud04_gpu_validation/git-inputs \
  --destination ./git-inputs-verified
```

The destination must not already exist. Damaged outputs are never accepted.
This reconstructs two archives; it does not start training or a Flux task.

## Scope and limitations

- Input identity, teacher order, data split, algorithm and acceptance are unchanged.
- Worker limits remain 1800 seconds total, 600 seconds native phase,
  six environments, at most five updates, no automatic retry.
- Input capability arguments are removed. A task-bound, limited result PUT
  capability is still required and must not be committed or made public.
- No account key is passed to the native child.
- Git clone remains a transfer. Platform clone/preparation time may be billable
  and is not covered by the worker's internal timer.
- CPU reconstruction/tests and GitHub readback do not prove Flux clone,
  simulator startup, CUDA execution or training success.
- The previous validation failed before training; no student checkpoint exists
  from it. A new GPU validation requires new explicit approval.
- Main/default branch and existing teacher tasks must remain unchanged.
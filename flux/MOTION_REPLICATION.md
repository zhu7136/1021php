# Bounded-memory motion replication (2026-10-24)

## What the real logs showed

Attempt 10 completed the Isaac launcher call in 5.413 seconds. Repeated CPU
stack snapshots then showed `offset_motions_to_file` building expanded arrays
with `np.concatenate([arr] * E)`. Later stacks showed the NPZ position array
being read. The worker reached its 900-second deadline before PPO iterations.
This is evidence of expensive preprocessing/loading, not evidence of a CUDA
OOM, disk fault, or Isaac initialization deadlock.

## Read-only size audit

```sh
python flux/audit_motion_replication.py --motions-dir /path/to/v8 --variants 32
```

The auditor reads NPY headers inside each ZIP, not complete arrays, and does
not modify files. For the verified 56 v8 clips it reports:

| Quantity | Value |
| --- | ---: |
| Original frames | 69,570 |
| Expanded frames (32 variants) | 2,226,240 |
| Combined array payload before replication | 271,677,074 bytes |
| Legacy output array payload | 7,838,597,264 bytes (7.30 GiB) |
| Float32 streamed output payload | 4,356,757,904 bytes (4.06 GiB) |
| Payload reduction | 44.42% |

These are derived array payload sizes, NOT measured runtime peak RAM or VRAM.
They exclude ZIP/NPY header overhead. Registry-added motion boundary/index
arrays are included. The old path already converted body positions to float32;
the other five kinematic fields remained float64 until the runtime loader.

## Patch

`obstacle_helpers.offset_motions_to_file` delegates to `motion_replication`:

1. Read one source field at a time, exactly once; do not iterate NPZ `.items()`.
2. For frame fields, write variants in the original variant-major/frame order,
   using 8 MiB chunks instead of constructing an E-times-sized array.
3. Preserve the old body-position arithmetic: cast to float32, add grid XY
   offset, then add per-variant XYZ origin. Do not reassociate the additions.
4. Narrow only the six kinematic fields to float32, matching the pinned
   Holosoma loader's explicit `torch.float32` conversions. Preserve metadata,
   integer motion indices, boolean boundaries, unknown-field dtypes, and names.
5. Write normal ZIP_STORED/ZIP64-compatible NPZ members; `np.load` needs no
   changes. Publish atomically only after the complete output closes.
6. Emit start/per-field/end events with payload sizes and elapsed times.

The lower-level writer accepts `narrow_float32=False` for exact legacy storage
dtypes while still avoiding expanded-array allocation. The integrated path uses
float32. This changes storage precision but is verified to preserve the CPU
float32 tensors consumed by the pinned loader. Source motion files are not
rewritten by the audit or tests; the generated temporary combined motion NPZ
is replaced just as in the old preprocess path. Failed writes retain a uniquely
named `.partial` scratch file for inspection and leave prior output untouched.

## Verification / boundaries

```sh
python -m unittest discover -s flux/tests -p test_motion_replication.py -v
python -m unittest discover -s flux/tests -v
```

12 new tests cover 1/3/32-variant parity with original Torch arithmetic, optional
legacy dtypes, one read per source field, no concatenate, bounded writes,
unchanged input arrays, same-path open-NPZ replacement, failure atomicity,
metadata/index dtype preservation, noncontiguous arrays, and header estimates.

CPU microbenchmark on one real 1,283-frame v8 clip, 32 variants, synthetic
offsets: exact loader-float32 parity passed; output file sizes were 144,567,200
bytes legacy and 80,355,616 bytes streamed. Timings are local one-shot CPU
microbenchmarks, NOT remote full-dataset performance predictions. Record:
https://wandb.ai/zhul136-university-of-science-and-technology-of-china/HuD04_WBT/runs/5dm7gjmc

No robot, obstacle sampling, variant count, PPO settings or budget is changed.
No paid task was launched to validate this patch. The expanded output is still
about 4.36 GB, so this does NOT prove it will load/train within 900 seconds on
Flux. A representative reduced-data smoke is an optional next trial, not an
automatically enabled setting. Longer training or further paid trials require
fresh user approval. Lazy indexed per-variant motion storage would reduce size
further but needs coordinated loader/index/coordinate changes and is not part
of this patch.
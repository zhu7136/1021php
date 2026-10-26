# HuD04 r6: image dependency inventory and bounded empty-app measurement

One necessary GPU-validation task under standing user authorization.
No package installation, model, robot, private input, PPO or checkpoint.
Catalog: one 4090D 24GB; image BJX00000170/V000212.
Worker limit 360s from Python entry; close 30s; automatic retries zero.
Queue, clone, pull and release are outside that worker timer, not a billing cap.

Delta from r5: read distribution versions for the target dependency candidate;
record exact Torch GPU properties before checking the resource contract.
If the GPU is capacity-qualified and 4090-family but fails the unchanged
4090D name contract, collect only CUDA arithmetic and eight empty-app updates,
then fail the task. This never authorizes training on mismatched resources.
Direct app.close(wait_for_replicator=False) must really return. Forced cleanup
is not success. No Holosoma close monkeypatch or forced-success exit.

Default command is read-only. Execution requires exact task/commit/provenance
binding and verifies the source manifest. The inventory reads package metadata,
not package import compatibility. The target dependency candidate is not
installed here. Empty-app update is not a physics, robot or PPO update.

Protocol tests use synthetic subprocess event streams, not GPU fixtures.
LICENSE/NOTICE retained from PHP. Source branch contains only diagnostic code,
package-name/version expectations and notices; never weights/wheels or secrets.
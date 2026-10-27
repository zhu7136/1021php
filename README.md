# HuD04 r30: original USD close_stage, all other helper logic retained

One necessary GPU diagnostic under standing bounded validation authorization.
Same r29 image BJX00000170/V000212, catalog one4090D-family GPU, headless
empty app, CUDA arithmetic, eight app updates, fastShutdown=false/readback,
worker360s, close30s, zero retries. Queue/clone/pull/release not included in
worker bound; catalog5.4 has unknown billing unit/currency and is not a cap.

Functional helper change: remove its USD Context.close_stage no-op definition,
assignment and success log, retain USD imports/context retrieval and original
SimulationContext import/STOP-flag handling and actual app.close. An AST
comparison against frozen r29 verifies this limited change after excluding
the new observational hook. Full Holosoma module is still NOT imported;
ISAACSIM-only bindings and stdlib logger match r29.

Before actual close, read back that USD close_stage is unchanged and record
SimulationContext presence/STOP flag. Fail closed if USD method changed.
Keep close observer/profile/supervisor/runtime versions byte-identical.
After the owned child exits/is recovered, mirror only a fixed allowlist of
owned close stack/settings/enter/return/error files. No symlink, nonregular
file, or file above64KiB is accepted. Base64 bytes/SHA256/length enable
independent exact retrieval; missing native-close evidence never passes.

35CPU protocol/mock tests passed, not GPU evidence. Runtime helper source
hash/task/commit/ARIA provenance are recorded. HUD04_R27_EVENT remains wire
prefix only, never task identity. Public isolated branch leaves main alone.

No robot, motion, PPO, checkpoint, wheel, install, credential, signed URL,
remote model call, or live training history. Default is read-only. Actual
normal close return, natural exit0 and complete journal are required.
No forced exit counts as success. Python stacks are not native C/CUDA stacks.
Different Flux tasks do not guarantee a fixed physical host/driver/cache.
Passing this probe does not establish HuD04 training lifecycle or root cause.
Amazon notices/license retained.
# Low-climb teacher: resume iteration660 with original dynamics
One 4090D24GB, worker <=1800s incl prep/save/seal; graceful stop1500s; child cutoff1670s; no automatic retry.
Restore actor, critic, optimizers, adaptive LR and iteration; CPU and remote checks reject actor-only fallback.
Same40clips,4096envs,seed42,original rewards/observation/action/safety/5rad-s leg limits.
Reuses the previously published exact teacher660 checkpoint and shared runtime. New complete training reference is explicitly authorized for publication2026-11-22T13:25:33Z.
New weights/results remain private in remote task outputs. Queue/clone/allocation/release excluded from worker bound. Not trajectory/RNG-exact resume, skill certification, or hardware validation.

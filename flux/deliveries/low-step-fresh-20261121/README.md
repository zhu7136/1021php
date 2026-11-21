# Hip/knee speed-limit trial

Fresh low-step teacher, random initialization, seed42, same56 clips/25676 frames,4096env.
Only dynamics change:8 hip/knee joints from5 to10rad/s; gains, effort, armature and other joints unchanged.
OneGPU,1800s worker wall cap including preparation and result sealing,2000update cap,zero retry.
Queue/clone/allocation/release outside worker cap; not convergence or hardware certification.
Native PhysX velocity limits must match config on all4096 environments before learning.
Read-only measured speeds after every physics step; checkpoint at672updates if reached.
New checkpoint and logs remain private; no historical weights loaded.
Publishing this entry is not evidence of GPU execution or successful training.

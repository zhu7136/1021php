# HuD04 locomotion: fresh all29 G1 simulation velocity limits
One 4090, worker<=1800s incl preparation/sealing, <=2000 updates, zero retries.
100 clips/71042frames;4096envs;seed42;no checkpoint load.
Only per-joint velocity limits change in robot dynamics; HuD04 geometry, PD, effort, armature and reward preserved.
Native PhysX readback of all31joints/all4096envs must match before learning.
Head limits retained13rad/s. Simulation limits are not guaranteed hard speed caps or hardware ratings.
New weights and raw logs stay private. Queue/clone/allocation/release time outside worker cap.

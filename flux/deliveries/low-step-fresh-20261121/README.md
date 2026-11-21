# HuD04 low-step teacher: G1-matched hip/knee SIMULATION limits

Fresh random initialization, seed42,4096envs,same56clips/25676frames.
Only dynamics change relative to baseline: left/right hip_pitch32,hip_roll20,
hip_yaw32,knee20 rad/s. Other joint limits, gains, damping, effort and armature unchanged.
Configuration vector, implicit actuator calibration and versioned URDF agree.
PhysX native limits must match all31joints/all4096envs before learning.
Read-only speed evidence after every physics step; no hard clipping added.
One4090D24GB; worker<=1800s including preparation/sealing, <=2000updates,
zero automatic retries. Queue/clone/allocation/release outside worker cap.
Not convergence, task-success, strict speed-cap or HuD04 hardware certification.
New weights/logs stay private; original baseline and speed10 branch remain unchanged.

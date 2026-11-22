# Fresh low-climb teacher with G1 simulation velocity limits
One Flux GPU, <=1800s worker including preparation/sealing, <=2000 updates, zero automatic retries.
40 original low-climb clips/32814frames,4096envs,seed42. No historical weight extraction or checkpoint load.
Only29 mapped velocity limits change; two head joints13rad/s. Other robot dynamics, geometry, PD, effort, reward, observations and actions preserved.
Read back all31 native PhysX limits in every environment before learning; log zero initial optimizer steps/iteration.
Reuses existing delivered data/runtime; no new data or trained weights published. New code on separate delivery branch; main unchanged.
Simulation experiment only, not hardware ratings or convergence certification. Queue/clone/allocation/release outside worker bound.

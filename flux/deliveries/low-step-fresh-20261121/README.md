# HuD04 low-step teacher: continue original 5rad/s policy for one bounded trial

Resume TASK_20261122_015 final native checkpoint after1363completed updates.
Exact SHA256 cf15fb0d86db22961a8774d0415286dd09f29e4fd278452172ddb81f8ba2c1a1.
No checkpoint bytes or private logs in Git. Flux private checkpoint directory:
1021php/flux/private-resume/. The worker selects exactly one file by pinned
size and SHA256,then supplies its actual mounted path to native load.
The config's /workspace/resume-input/model_smoke_final.pt is a logical contract,
replaced only by that verified private mount path. Missing/wrong bytes fail closed.
Actor,critic,optimizers,applicable normalizers,learning rates,and iteration are
compared exactly against the source before further learning. The source stores
last completed iteration1362; the next new optimizer update is iteration1363.
Source actor and critic optimizer step counters are27260each.
The simulator resets in a new process: this is not exact trajectory/RNG resume.

All8 left/right hip_pitch/hip_roll/hip_yaw/knee velocity limits stay5rad/s.
Configuration vector,implicit actuator calibration,and original URDF agree.
All31joints/all4096environments must pass real PhysX limit readback before PPO.
Same56clips/25676frames,4096environments,seed42,other dynamics/rewards unchanged.
Read-only speed measurements;no hard state clipping or real-robot claim.

ONE4090D24GB;worker<=1800seconds including preparation,checkpoint,cleanup,and
result sealing;<=2000ADDITIONAL updates;zero automatic retries. Soft stop
requested at worker1500seconds to reserve teardown/recovery time. Queue,
clone,allocation/release are outside the worker cap. Prior exit1 remains an
unresolved lifecycle issue,not evidence of no learning or an app.close diagnosis.
Not convergence or task-success certification. New weights/logs remain private.

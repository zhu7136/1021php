# HU_D04 repair report — 2026-10-23

## v8 continuation: all 56 clear the same kinematic gates

This section supersedes the **historical v7 partial-repair results below**.
No GPU job, training or simulator execution was started. Original source files
remain unchanged. This is still a candidate for simulation, not hardware use.

The failure diagnosis found that representative 74–83 mm hand residuals were
solvable within joint limits by bounded trust-region least squares. Independent
clipped DLS could settle on an unsuitable branch. `continuous_arms.py` now solves
frames sequentially using bounded TRF, the previous frame as the initial guess,
and a 0.003 continuity residual weight. The arm-only FK/Jacobian is tested against
the full URDF chain. Target positions, limb scaling and acceptance thresholds
are unchanged. This does not guarantee a globally optimal or collision-free pose.

`retarget_continuous.py` was piloted on 3 difficult clips, then all 56 source clips
were recomputed, not merely combined with the prior 29. CPU trial records are in
the same W&B project. `validate_hud04.py` independently reads the saved output.

| Full v8 read-back metric | Result |
|---|---:|
| Motion/terrain pairs | 56 / 56 |
| Passed original kinematic gates | 56 / 56 |
| Failed / rejected | 0 / 0 |
| Output frames at 50 FPS | 69,570 |
| Foot position mean / maximum | 0.854 / 10.165 mm |
| Scaled distal-hand position mean / maximum | 0.136 / 7.589 mm |
| Maximum foot orientation residual | 1.626 degrees |
| Maximum joint speed / URDF limit | 0.796153 |
| Uniform duration multiplier across clips | 2.0699–3.2246x |

All joint angles are within URDF limits. Full-body FK references, quaternion
normalization, joint/body velocity derivatives, root velocity fields and terrain
digests are also checked. Original thresholds remain: 20 mm endpoint position,
5 degree foot orientation, speed ratio <=0.81, plus strict numerical consistency.
The additional checks do not certify acceleration, torque, collision or balance.

Reproduce from the checkout root:

```bash
OPENBLAS_NUM_THREADS=1 python flux/retarget_continuous.py --all \
  --output outputs/hud04-candidates-v8
python flux/validate_hud04.py outputs/hud04-candidates-v8
python -m unittest discover -s flux/tests -v
```

Outputs retain 50 FPS, have more frames and genuinely run slower; these are not
original-speed demonstrations. Arms preserve only morphology-scaled distal-hand
position, not original wrist orientation or elbow pose. `vel_cmd` is omitted,
so these are only terrain-teacher inputs, not student/distillation inputs.

Delivery contains the complete 56-pair v8 kinematic candidate, diagnostics,
code patch, repaired source files and tests. The launcher still requires an
explicit separate dataset and acknowledgment; clearing its data gate is not
proof that GPU simulation or training has been validated. Next step is IsaacSim
visual/contact/collision/dynamics review, with separate compute approval.

## Historical v7 report (superseded by v8 above)

## Outcome

Configuration issues fixed; motion retargeting remains a **partial repair**.
CPU checks only. No Flux GPU task, GPU training, or robot execution was started.
Original 56 motion/terrain pairs are unchanged. Holosoma submodule remains at
`4a0bf04cde5930af5ed59da797568d32be972525`.

## Configuration fixes

- Right shoulder roll upper limit: 1.571 -> 0.087 rad.
- Left wrist yaw range: [-1.571, 1.571] -> [-2.356, 0.785] rad.
- Right wrist yaw range: [-1.571, 1.571] -> [-0.785, 2.356] rad.
- Default root height in both HU_D04 robot and experiment config: 0.80 -> 0.91 m.
  Default FK foot-center height becomes +0.033374 m rather than -0.076626 m.
  This is static center clearance, not a contact or mesh clearance certificate.

## Original-data findings

The 56 files contain 25,676 frames. Stored body references match G1 rather than
HU_D04: original HU_D04 nonroot mean position mismatch is about 0.146 m; maximum
about 0.387 m. 5,897 frames (22.97%) exceed the HU_D04 URDF speed limits; peak
knee speed is about 15.04 rad/s against a 5 rad/s limit.

## New offline pipeline

`kinematics.py` parses URDF transforms/limits and provides batched forward
kinematics and analytic geometric Jacobians. `retarget_hud04.py` preserves foot
world targets, lifts the root, performs bounded damped IK and projects distal
arms again on the dense resampled time grid. HU_D04 distal wrist **roll** maps to
G1 distal wrist **yaw**; same-name wrist links are not the same chain location.
Shoulder-relative endpoint targets are scaled by URDF chain length, not by the
folded zero-pose endpoint distance. Wrist orientation/elbow pose are not imposed.
Thus small hand-position error is not a full upper-body pose certificate.

PCHIP joint interpolation avoids angle overshoot; root quaternions use Slerp.
Actual trajectories are uniformly slowed while keeping 50 FPS; FPS metadata is
not changed to conceal excessive motion. Body poses, joint/body velocities and
world angular velocities are regenerated. Source terrains remain byte-identical.
Stale source `vel_cmd` is omitted; only the terrain teacher is supported.

## Full batch read-back results

All 56 source clips were attempted, with fixed gates (no threshold relaxation):

| Outcome | Clips |
|---|---:|
| Passed independent kinematic gates | 29 |
| Generated but failed wrist-position gate | 19 |
| Rejected because dense IK needs >10x time stretching | 8 |

All 48 generated trajectories have joint angles within URDF limits, stored
reference poses consistent with HU_D04 FK and unchanged terrain files. Their
maximum joint-speed/limit ratio is 0.795823. Those checks alone do not make the
19 failed trajectories usable; their worst hand target error reaches 82.83 mm.

For the **29 passing clips**:

- Maximum foot-position error: 10.17 mm (gate 20 mm).
- Maximum scaled distal-hand-position error: 1.86 mm (gate 20 mm).
- Time stretching: 2.283x–3.567x; these are not original-speed demonstrations.
- Foot orientation gate: 5 degrees, independently checked after resampling.

Example `forward_walk1.2s_step0.5s_motion.npz`: 489 -> 1,400 frames, 2.8668x
duration; mean foot error 0.843 mm, maximum 6.160 mm; maximum hand error 0.797 mm;
peak speed ratio 0.79253. Original source-grid diagnostics are not confused with
final dense-grid endpoint errors.

`validation.json` provides per-file results, hashes, threshold outcomes and
kinematic limitations. Rejection diagnostics identify the 8 missing candidates.
The full-bundle validator is intentionally **false** and the launcher blocks
training on both the original G1 data and this incomplete 29-clip subset.

## Delivery and next step

Deliverables include the repair patch, repaired source files, tests, this report,
all validation/rejection diagnostics, and only the 29 passing motion/terrain
pairs. The 19 failing candidate NPZs are not promoted as repaired training data.
Repairs are not pushed to GitHub while credential rotation is unconfirmed.

Next work is a temporally constrained arm retargeting pass for the remaining
27 clips, followed by collision/ground-contact visualization and dynamics checks
in IsaacSim. A bounded GPU smoke test requires separate approval after review.
Do not deploy a policy on hardware from these CPU checks. Estimated armatures,
torque feasibility, self-collision and support/COM remain unvalidated.
## Purpose

Defines how the HU_D04 humanoid robot model is loaded into the PHP MuJoCo sim2sim flow, with its head degrees of freedom locked so the policy-facing joint space matches the G1 29-DOF layout.

## ADDED Requirements

### Requirement: HU_D04 model loadable in MuJoCo sim2sim

The system SHALL load the HU_D04 robot description (MJCF) into the PHP MuJoCo sim2sim flow, using the variant whose ankle is modeled as a serial controlled hinge pair (`ankle_pitch` / `ankle_roll`) rather than the parallel A/B-only description.

#### Scenario: Model loads without error
- **WHEN** the HU_D04 sim2sim variant MJCF is provided to the sim2sim loader
- **THEN** MuJoCo parses it successfully and reports a root free-joint plus the actuated hinge joints

#### Scenario: Serial ankle joints present and controlled
- **WHEN** the loaded model is inspected
- **THEN** `left_ankle_pitch_joint`, `left_ankle_roll_joint`, `right_ankle_pitch_joint`, and `right_ankle_roll_joint` exist as hinge joints and are the physical ankle joints that the policy drives

#### Scenario: Parallel achilles bodies are non-colliding visuals
- **WHEN** the loaded model is inspected for the A/B achilles bodies
- **THEN** those bodies are present for visualization but do not carry collision geometry or participate in the controlled joint set

### Requirement: Head degrees of freedom locked

The system SHALL lock the HU_D04 head joints (`head_yaw_joint`, `head_pitch_joint`) at their default angles so the policy-facing joint space excludes them.

#### Scenario: Head joints not part of policy output
- **WHEN** the policy-facing joint list for HU_D04 is assembled
- **THEN** it contains 29 joints and does not contain `head_yaw_joint` or `head_pitch_joint`

#### Scenario: Head held at default angle during control
- **WHEN** the sim2sim control step publishes a command
- **THEN** `head_yaw_joint` and `head_pitch_joint` are commanded to their default angles with holding gains, independent of the policy output

#### Scenario: Policy joint count matches G1
- **WHEN** the HU_D04 policy-facing joint list is compared against the G1 29-DOF joint list
- **THEN** both contain 29 joints covering the same limb and waist structure

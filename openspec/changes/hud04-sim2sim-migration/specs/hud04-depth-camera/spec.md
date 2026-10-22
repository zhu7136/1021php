## Purpose

Defines how the HU_D04's D435i depth camera is configured in the PHP MuJoCo sim2sim so it produces a depth image with the same spec as G1's, feeding the unchanged depth backbone to yield the depth latent the migrated policy needs.

## ADDED Requirements

### Requirement: D435i camera mount from URDF geometry

The system SHALL mount the HU_D04 depth camera using the D435i mount geometry already authored in the HU_D04 URDF, rather than inventing a new pose.

#### Scenario: Camera parent and transform come from the URDF
- **WHEN** the HU_D04 depth camera config is built
- **THEN** it uses the `waist_camera_joint` transform (parent `waist_pitch_link`, xyz `0.092 0.0175 0.2751`, rpy `0 2.1818 0`) from `HU_D04_01.urdf`

#### Scenario: Parent link exists in the loaded model
- **WHEN** the camera config's parent frame is resolved against the loaded HU_D04 MJCF
- **THEN** the parent body (`waist_pitch_link`) exists and the camera pose is computed relative to it

### Requirement: Depth image spec matches the G1 pipeline

The HU_D04 depth camera SHALL emit depth images with the same resolution and crop as the G1 camera so the unchanged depth backbone accepts them.

#### Scenario: Image resolution and crop match G1
- **WHEN** the HU_D04 camera renders a depth image
- **THEN** the post-crop image is 58x87, matching the G1 `G1FlatRsD435iConfig` crop of `RsD435iConfig`

#### Scenario: Backbone accepts the image
- **WHEN** the HU_D04 depth image is passed to the depth backbone
- **THEN** the backbone (input `[1, 58, 87]`) returns a `[1, 32]` depth latent without shape errors

### Requirement: Ray-cast bodies mapped to HU_D04 meshes

The depth camera's ray-cast geometry SHALL reference HU_D04 link/mesh pairs so the rendered depth reflects HU_D04's actual geometry.

#### Scenario: Ray-cast body map covers HU_D04 limbs
- **WHEN** the camera's `ray_cast_bodies` map is resolved
- **THEN** every entry names a body that exists in the loaded HU_D04 MJCF and a mesh present in the HU_D04 meshes directory

#### Scenario: Self-occlusion is depth-correct
- **WHEN** the camera renders while the robot is in its default pose
- **THEN** near-field hits on the robot's own links are present in the depth image (no missing self-occlusion)

### Requirement: Depth latent wired into the policy observation

The depth latent SHALL be produced from the HU_D04 depth image and placed into the student observation's depth block.

#### Scenario: Latent fills the observation depth block
- **WHEN** the HU_D04 observation is assembled with a live depth image
- **THEN** the trailing 32-dim depth-latent block contains the backbone output, and the full observation remains 140 dims

#### Scenario: Zero-depth fallback still runs
- **WHEN** no depth image is available
- **THEN** the observation still assembles with a zero depth block, so the control loop runs in a degraded (non-perceptive) mode

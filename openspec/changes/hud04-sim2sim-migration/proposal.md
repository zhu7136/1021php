## Why

PHP 的 depth-student 管线目前只服务 Unitree G1（29-DOF）。要评估逐际动力 HU_D04 人形能否复用同一套 parkour 策略与部署栈，必须先有一个可运行的 sim2sim 验证路径：把 HU_D04 的机器人模型接入 PHP 的 MuJoCo sim2sim 骨架，并验证将 G1 训练的 29-DOF 策略迁移到 HU_D04 的可行性（关节空间对齐、顺序映射、动作缩放）。这是最终上真机之前的必要可行性门槛。

## What Changes

- 新增 HU_D04 机器人模型接入：从 `humanoid-description/HU_D04_description`（URDF/MJCF/USD）接入 PHP 的 MuJoCo sim2sim 流程，使用逐际的串联踝 MJCF 变体（`ankle_pitch/roll` 作为受控 hinge 关节）。
- 新增头部自由度固定机制：将 HU_D04 的 `head_yaw_joint` / `head_pitch_joint` 锁定在默认角，使策略关节空间从 31 降到 29，与 G1 对齐。
- 新增 G1→HU_D04 关节映射层：把 G1 的 29-DOF 关节顺序映射到 HU_D04 的 29 个策略关节（含腰部位序差异、踝关节命名对齐）。
- 新增动作缩放/默认角适配：将 G1 student 的输出语义（归一化 action 或关节目标）转换为 HU_D04 的 PD 目标，复用或重标定 `action_scale` 与 Kp/Kd。
- 新增 sim2sim 验证运行路径：在 MuJoCo 中加载 HU_D04 并驱动 G1 策略，观察 HU_D04 是否能执行 locomotion / 翻箱类动作。
- 新增 HU_D04 深度相机接入：使用 HU_D04 URDF 中已有的 D435i 挂载几何（`waist_camera_joint`，父 link `waist_pitch_link`），配置 HU_D04 的深度相机参数与射线求交体映射，产出与 G1 同规格的 58×87 深度图并送入深度 backbone 得到 depth latent。
- 记录迁移可行性结论与上真机所需的后续工作（SDK 集成、传感器适配、安全层）。

## Capabilities

### New Capabilities

- `hud04-model-integration`: HU_D04 机器人模型（URDF/MJCF）接入 PHP MuJoCo sim2sim 流程的能力，包括模型加载、关节/执行器识别、头部自由度固定。
- `g1-to-hud04-policy-transfer`: 将 G1 训练的 29-DOF 策略迁移到 HU_D04 的能力，包括关节顺序映射、命名对齐、动作缩放与默认角适配。
- `hud04-depth-camera`: HU_D04 的 D435i 深度相机在 sim2sim 中的接入能力，包括相机挂载（frame/偏移）+ 射线求交体配置，产出与 G1 同规格的深度图与 depth latent。
- `hud04-sim2sim-validation`: 在 MuJoCo 中加载 HU_D04 并运行迁移后策略、观察行为并记录可行性结论的验证能力。

### Modified Capabilities

<!-- 无：PHP 现有 G1 管线行为不变，本变更仅新增 HU_D04 侧能力 -->

## Impact

- 新增代码：HU_D04 模型描述接入、关节映射、头部锁定、深度相机接入、sim2sim 运行适配（新增文件，不改动 G1 现有路径）。
- 复用资产：`humanoid-description/HU_D04_description`（MJCF/USD/URDF、`d435_link.STL`、URDF 中的 `waist_camera_joint`/`head_camera_joint` 挂载几何）、`limx_rl_lab/limx_ws/humanoid-mujoco-sim`（逐际 sim2sim 参考实现、`simulator.py` 的串联踝 MJCF）、`limx_rl_lab/deploy/sim2sim/configs/limx_walk_mimic_sdk.yaml`（31 关节定义、Kp/Kd/default_angles 参考）、Holosoma 的 `G1FlatRsD435iConfig`（相机参数与射线求交体映射的参考实现）。
- 依赖：MuJoCo（sim2sim）、PHP 现有 `run_php_sim.sh` / `run_php_inference.sh` 骨架、G1 student ONNX 对（`~/.cache/php-parkour/student`）、Holosoma 的 Warp 深度相机模块（`holosoma.sensors.warp`）。
- 不改动：G1 训练管线（`wbt_training/`）、G1 sim2sim 启动器、G1 ONNX 导出逻辑。
- 后续（本次不实现）：HU_D04 真机 SDK 集成、深度相机 ROS2 适配、安全/急停层——本变更记录为上真机的后续工作。

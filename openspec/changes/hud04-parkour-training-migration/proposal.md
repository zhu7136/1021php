## Why

PHP 的 depth-student 管线只服务 Unitree G1（29-DOF），而 `hud04-sim2sim-migration` 已证明：G1 策略**零样本迁移到 HU_D04 必然失败**（站立即跌倒），根因是形态学差异——HU_D04 质量 52.9 kg vs G1 33.6 kg（+58%），且关节 PD 刚度 Kp 是 G1 的 2–5.6 倍（腿部均值 3.2×）。由于 PHP 的动作缩放公式是 `action_scale × effort / Kp`，同一归一化动作在 HU_D04 上只产生 G1 的 10%–50% 位移，策略无法补偿。因此 HU_D04 要获得翻越障碍能力，**必须在训练侧重训教师与学生**，而不是继续修补推理接口。

本次变更补齐训练链路：用 GMR（General Motion Retargeting）的 `limx_oli_edu` 支持生成 HU_D04 原生动作数据，新建 HU_D04 训练预设，完成教师训练 → 学生深度蒸馏 → ONNX 导出。

## What Changes

- **新增 HU_D04 机器人配置**：在 `holosoma/config_values/robot.py` 新增 `hu_d04` RobotConfig，`dof_names` 只列 29 个主动关节（排除 `head_yaw_joint`/`head_pitch_joint`），`body_names` 精确保 32 个与 G1 语义对齐；配套 `build_hu_d04_implicit_actuators`。
- **新增 GMR 动作数据管线**：把 GMR 输出的 31 关节 / 42 bodies BeyondMimic npz 转换为训练可用的 29 关节 / 32 bodies，含头部裁剪、body 名映射（`pelvis`→`base_link`、`torso_link`→`waist_pitch_link`、`*_foot_contact_point`→`contact_foot_center_*`）、关节重排成 G1 顺序，并配对可复用的 `terrain.npy`。
- **新增 HU_D04 训练预设**：新建 `wbt/hu_d04/` 六个模块（command/observation/reward/termination/curriculum/experiment）与 `exp:hu-d04-wbt-terrain-ref-noscale`、`exp:hu-d04-wbt-terrain-warp-distill-finetune` 两个 preset，在 `DEFAULTS` 注册。
- **新增 HU_D04 资产接入**：将 `humanoid-description/HU_D04_description`（URDF/MJCF/USD）复制到 `holosoma/data/robots/hu_d04/`，使 IsaacSim 可加载。
- **新增 HU_D04 深度相机配置**：D435i 挂 `waist_pitch_link`，offset `(0.092, 0.0175, 0.4336)`、下俯 35°，保持 58×87 与 `[0.3, 3.0]` 裁剪契约不变。
- **重新标定动作缩放**：HU_D04 不能沿用 G1 的 `action_scale=0.25`；采用 `action_scale=1.0` + `action_scales_by_effort_limit_over_p_gain=False`（`*_noscale` 变体）或以逐际 `limx_walk_mimic_sdk.yaml` 增益重新标定。
- **不改动 G1 路径**：G1 的训练预设、sim2sim 启动器、ONNX 导出逻辑全部保持现状。

## Capabilities

### New Capabilities

- `hud04-robot-config`: HU_D04 的 29-DOF 训练配置能力，包括关节/连杆清单、限位、执行器增益、头部被动排除，以及与 G1 语义对齐的 32 body 映射。
- `gmr-motion-pipeline`: 将 GMR 重定向输出（31 关节 / 42 bodies）转换为 PHP 训练输入（29 关节 / 32 bodies）的能力，含头部裁剪、body 名映射、关节重排与地形配对。
- `hud04-teacher-training`: 使用 HU_D04 配置训练地形教师的 ability，含动作缩放标定、观测/奖励/终止配置与翻越技能训练。
- `hud04-student-distillation`: 从 HU_D04 教师蒸馏深度学生的能力，含 140 维观测（depth latent 32）与相机外参适配。
- `hud04-policy-export`: 导出 HU_D04 学生 ONNX 对（`student.onnx` + `depth_backbone.onnx`）的能力，元数据随机器人配置自动跟随。

### Modified Capabilities

<!-- 无：G1 现有能力的行为不变，本变更只新增 HU_D04 侧训练能力 -->

## Impact

- **新增代码**：`holosoma/config_values/robot.py` 的 `hu_d04` 条目（约 600 行）、`wbt_training/config_values/robot_config.py` 的 `build_hu_d04_implicit_actuators`、`wbt/hu_d04/` 六模块、GMR 数据转换脚本（约 100 行）、两个实验 preset。
- **修改代码**：`wbt_training/config_values/experiment.py`（preset 与 `DEFAULTS` 注册）、`terrain.py`（`body_name_ref` 与连杆白名单正则参数化）、`depth_distillation.py`（相机挂载 link 参数化）。
- **复用资产**：`humanoid-description/HU_D04_description`（URDF/MJCF/USD）、`GMR`（`limx_oli_edu` IK 配置与 `smplx_to_robot.py`/`bvh_to_robot.py`）、`limx_rl_lab/deploy/sim2sim/configs/limx_walk_mimic_sdk.yaml`（Kp/Kd/default_angles 参考）、已下载的 `~/.cache/php-parkour/motions-*` 中的 `terrain.npy`（机器人无关，直接复用）。
- **依赖**：IsaacSim + NVIDIA GPU（教师训练与学生蒸馏）、GMR 的 conda 环境与 SMPL-X 人体模型（动作生成）。
- **不改动**：G1 训练管线、G1 sim2sim 启动器、G1 的 `run_php_sim.sh`/`run_php_inference.sh`、ONNX 导出的元数据生成逻辑（已从 `env.robot_config` 反射，自动跟随）。
- **已知遗留（本次不实现）**：真机部署需要 limx bridge（当前 `bridge/__init__.py` 只有 unitree/booster）；ROS2 深度相机驱动；安全急停层。

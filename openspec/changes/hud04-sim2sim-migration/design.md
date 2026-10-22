## Context

PHP 的 depth-student 管线已支持 Unitree G1（29-DOF，串联踝）的完整训练→sim2sim→部署流程。逐际动力 HU_D04 是 31-DOF 人形（含头部 2 DOF，并联踝机构），但提供了**训练简化版 MJCF**：串联踝 `ankle_pitch/roll` 作为受控关节，A/B 并联杆退化为纯视觉。

现有代码库已包含：
- `humanoid-description/HU_D04_description/`：官方 MJCF/URDF/USD
- `limx_rl_lab/deploy/sim2sim/`：逐际自有的 sim2sim 实现，含头部固定逻辑（`head_hold_targets`）
- `limx_rl_lab/limx_ws/humanoid-mujoco-sim/simulator.py`：MuJoCo 仿真器加载 + 控制回调

本设计复用 PHP 的 `run_php_sim.sh` / `run_php_inference.sh` 骨架，新增 HU_D04 模型接入与 G1 策略迁移层。

## Goals / Non-Goals

**Goals:**
- 在 PHP 的 MuJoCo sim2sim 中加载 HU_D04 并使用 G1 训练的策略（29-DOF ONNX）驱动
- 固定头部 2 DOF，使 HU_D04 策略空间 = 29，与 G1 维度对齐
- 实现关节顺序映射（G1 → HU_D04）与动作缩放适配
- 输出迁移可行性结论与上真机建议

**Non-Goals:**
- 不修改 G1 训练管线（`wbt_training/` 保持不变）
- 不改动 G1 sim2sim 启动器（`run_php_sim.sh` 仅可选扩展 HU_D04 模式）
- 不实现真机 SDK 集成、ROS2 深度相机驱动、安全急停层（记录为后续工作）
- 不重新训练 HU_D04 教师/学生（用 G1 已有 ONNX 验证迁移）

## Decisions

### D1: 头部固定策略 → 控制器层兜底，策略层排除

**选择**：策略输出 29-DOF（不含头），控制器用 `head_hold_targets` 机制补发头部固定命令。

**理由**：
- 逐际已有实现（`gamepad_policy_controller.py:713-720`），验证可用
- 无需改动 ONNX 导出逻辑，仅需部署配置排除头部关节
- 头部 default 角由 MJCF 提供，不用额外标定

**备选**：
- 策略层直接输出 31-DOF（含头）→ 需重训教师，成本高
- 策略输出 31-DOF 但头部固定为零 → 浪费 2 维输出，推理计算白费

### D2: 关节映射 → 静态 `joint_ids_map` 配置，启动时解析

**选择**：在 sim2sim 启动脚本中加载 `joint_ids_map`（G1 索引 → HU_D04 索引），控制器查表重排动作向量。

**理由**：
- G1 与 HU_D04 的 29-DOF 顺序不同（腰部位置、腿部交错方式）
- 静态映射表易维护，支持运行时校验（长度、范围）
- 复用 PHP 现有 `depth_distillation.py:_setup_joint_reordering` 逻辑

**映射表草案**（待最终确认）：
```python
G1_JOINT_ORDER = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll"
]

HUD04_JOINT_ORDER = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",  # 位置可能不同，需查逐际配置
    ...
]
```

### D3: 动作缩放 → 复用 G1 `action_scale`，HU_D04 PD 增益单独标定

**选择**：G1 student ONNX 输出归一化 action（`[-1, 1]`），乘以 `action_scale` 后加到 default 角。HU_D04 沿用 `action_scale=0.25`，但 Kp/Kd 用逐际推荐值（220/80/...）。

**理由**：
- G1 的 `action_scale` 是经验安全值，HU_D04 电机扭矩特性接近（同为 600W 级关节）
- PD 增益需匹配 HU_D04 质量/摩擦，直接复用逐际 sim2sim 配置

**风险**：若 G1 ONNX 输出已是关节角（非归一化），则 scale 需调整为 1.0 或重标定。

### D4: MJCF 选择 → 用 `hu_description/xml/HU_D04_01.xml`（串联踝版）

**选择**：加载 sim2sim 简化版 MJCF（串联踝 + 并联视觉），不用完整并联机构版。

**理由**：
- 串联踝版与 G1 运动学一致，策略可直接迁移
- 并联版需固件逆解算，闭源，无法在 MuJoCo 仿真

**备选**：
- 自己写五杆并联运动学 → 工作量大，且真机仍依赖逐际固件

## Risks / Trade-offs

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| G1 ONNX 输出语义不匹配（归一化 vs 关节角） | 中 | 高 | 先跑冒烟测试：单步推理 + 打印输出范围，确认 scale|
| 关节顺序映射错误 → 机器人乱抖 | 中 | 高 | 启动时打印映射表 + 单关节验证模式（逐个关节发 0.1rad 指令看动作） |
| HU_D04 质量/摩擦参数与 G1 差异大 → 策略振荡 | 中 | 中 | 先降增益（Kp × 0.5）+ 限制 action 幅度（`action_scale` × 0.5） |
| 头部固定不牢 → 头抖影响平衡 | 低 | 中 | 加高头部 Kp（≥400）+ 监控头部角度漂移日志 |
| 仿真通过但真机不工作（SDK 兼容问题） | 高 | 高 | 本阶段只验证 sim2sim，真机风险记入后续任务 |

## Migration Plan

1. **Week 1**: 接入 HU_D04 MJCF → MuJoCo 加载成功，头部固定验证
2. **Week 2**: 实现关节映射层 + action_scale 适配 → G1 策略驱动 HU_D04 sim2sim
3. **Week 3**: 跑通翻箱/locomotion 测试 → 记录成功率，调参
4. **Week 4**: 输出可行性报告 + 真机后续任务清单

**Rollback**: 删除 `hud04-` 相关配置，PHP 恢复 G1-only 模式（无侵入，回滚成本 ≈ 0）

## Open Questions

- （已解决）G1 student.onnx 输出是归一化 action 还是关节角？→ **归一化 action**；
  ONNX 元数据 `action_scale` 全为 1.0，`q_target = action * scale + default_joint_pos`。
- （已解决）迁移时的默认角偏移用谁？→ **必须用 G1 策略元数据的 `default_joint_pos`**
  （策略的训练参考），不是 HU_D04 自身默认姿态。用后者会给每个关节引入固定偏移。
- （已解决）HU_D04 深度相机如何挂载？→ 用 URDF 中已有的 `waist_camera_joint`
  （父 `waist_pitch_link`），对标 G1 的 `torso_link` 挂载。
- （新发现）G1 策略无法零样本站立/翻箱，根因是 **G1 与 HU_D04 形态学差异**
  （腿长约 1.5 倍、质量 35 vs 53 kg），已排除接口/映射/深度因素。

## Morphology Gap (后续工作)

接口层已全部打通，但策略零样本迁移失败，根因是形态学差异。后续可选路径：

1. **重新训练** HU_D04 的教师/学生（路线 A）：用 HU_D04 的 motion retarget + 训练，
   成本最高但最彻底。
2. **域随机化微调**：以 G1 策略为初始化，在 HU_D04 sim2sim 中少量微调。
3. **形态学对齐**：把 HU_D04 的腿长/质量参数向 G1 靠拢做消融，验证差异贡献
   （仅用于分析，不用于真机）。

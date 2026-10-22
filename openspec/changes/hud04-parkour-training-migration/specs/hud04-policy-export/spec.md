## Purpose

把 HU_D04 学生策略导出为可部署的 ONNX 组合，使其能在推理阶段仅依赖深度相机与本体感知输入实时生成关节目标，且导出的元数据与训练配置一致。

## ADDED Requirements

### Requirement: ONNX 组合包含策略主体与深度骨干

导出产物 MUST 包含两个 ONNX 文件（`student.onnx` 与 `depth_backbone.onnx`），并 MUST 与训练时的深度潜变量维度一致。导出 MUST 在保存检查点时自动触发。

#### Scenario: 一对 ONNX 被导出
- **WHEN** 学生训练保存检查点
- **THEN** 同一迭代目录下生成 `student.onnx` 与 `depth_backbone.onnx`

#### Scenario: 深度潜变量为 32 维
- **WHEN** 检查导出的骨干网络
- **THEN** 骨干输入为 58×87 深度图，输出为 32 维潜变量

### Requirement: 元数据从机器人配置自动跟随

导出的 ONNX 元数据（`joint_names`、`action_scale`、`default_joint_pos`、`kp`、`kd`）MUST 从训练时使用的机器人配置自动读取，MUST NOT 写死为 G1 的值。

#### Scenario: 元数据正确
- **WHEN** 从 HU_D04 学生导出 ONNX
- **THEN** 元数据中的 `joint_names` 为 HU_D04 的 29 个受控关节名

#### Scenario: 缩放与默认角正确
- **WHEN** 检查导出的 `action_scale` 与 `default_joint_pos`
- **THEN** 两者来自训练配置，且与 G1 的对应值不同

### Requirement: 部署配置与训练一致

部署用的推理配置 MUST 使用与训练阶段一致的关节顺序、动作缩放与默认角。配置不匹配时 MUST 报错而非静默运行。

#### Scenario: 部署配置匹配训练
- **WHEN** 使用训练生成的 ONNX 推理
- **THEN** 部署配置中的关节顺序、缩放与默认角与训练元数据一致

#### Scenario: 检测到失配
- **WHEN** 部署配置与 ONNX 元数据不一致
- **THEN** 推理启动失败并指出差异项

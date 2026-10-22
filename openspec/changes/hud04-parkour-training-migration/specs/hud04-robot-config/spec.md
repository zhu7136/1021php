## Purpose

为 HU_D04 人形机器人提供一套可训练的 29-DOF 机器人配置，使其策略空间与 Unitree G1 对齐，同时保留 HU_D04 的真实连杆语义与执行器特性，作为教师训练、学生蒸馏与策略导出的共同基础。

## ADDED Requirements

### Requirement: 策略关节集为 29 个主动关节

HU_D04 训练配置的受控关节集合 MUST 恰好包含 29 个关节，MUST NOT 包含 `head_yaw_joint` 与 `head_pitch_joint`。头部两个自由度 MUST 在 URDF 中保留并参与物理仿真，但不获得执行器、不产生 action、不进入观测。

#### Scenario: 策略关节数量与内容
- **WHEN** 读取 HU_D04 训练配置的受控关节列表
- **THEN** 列表长度为 29，且不包含任何 `head_*` 关节

#### Scenario: 头部关节被动存在
- **WHEN** 在仿真中加载 HU_D04 并施加策略动作
- **THEN** `head_yaw_joint` 与 `head_pitch_joint` 仍存在于机器人模型中，但不被任何执行器驱动

### Requirement: 关节名集合与 G1 一致

HU_D04 的 29 个受控关节名集合 MUST 与 G1 的 29 个受控关节名集合完全相同。顺序 MUST 显式声明，且 MUST NOT 依赖两个机器人 URDF 的默认顺序一致。

#### Scenario: 集合一致
- **WHEN** 比较 HU_D04 与 G1 的受控关节名集合
- **THEN** 两个集合相等，且元素个数均为 29

#### Scenario: 顺序差异被显式处理
- **WHEN** HU_D04 的默认关节顺序与 G1 不同（手腕 roll/yaw 次序不同）
- **THEN** 配置中 MUST 提供显式的重排索引，而不是依赖隐式同名匹配

### Requirement: 连杆表为 32 个且语义对齐 G1

HU_D04 训练的连杆（body）列表 MUST 恰好包含 32 项，且 MUST 与 G1 的 32 项在语义上一一对应。映射关系 MUST 显式声明，至少覆盖：`pelvis`→`base_link`、`torso_link`→`waist_pitch_link`、左右 `*_foot_contact_point`→左右 `contact_foot_center_*`。HU_D04 特有的 `head_*`、`*_hand_contact`、`*_hand_manip` 与 `contact_foot_{heel,tip}_*` MUST 被排除。

#### Scenario: 连杆数量与 G1 相同
- **WHEN** 读取 HU_D04 训练配置的连杆列表
- **THEN** 列表长度为 32，且与仿真器解析出的连杆名逐项相等

#### Scenario: 语义映射可验证
- **WHEN** 按映射表把 HU_D04 连杆名映射到 G1 语义
- **THEN** 每个 G1 连杆都有一个 HU_D04 对应项，无重复、无遗漏

### Requirement: 关节限位从 HU_D04 模型读取

关节位置限位 MUST 取自 HU_D04 官方模型描述，MUST NOT 沿用 G1 的限位值。力矩与速度限位 MUST 同样来自 HU_D04 规格。

#### Scenario: 限位与官方模型一致
- **WHEN** 比较配置中的关节限位与 HU_D04 模型描述的限位
- **THEN** 两者逐关节相等（例如膝关节上限为 2.583 rad、髋俯仰 ±2.618 rad）

#### Scenario: 超限动作被物理拒绝
- **WHEN** 策略输出导致某关节超过 HU_D04 限位
- **THEN** 仿真中该关节被限位约束，动作不越界

### Requirement: 执行器增益使用 HU_D04 标定值

关节 PD 刚度与阻尼 MUST 使用 HU_D04 的标定值，MUST NOT 沿用 G1 的电机标定。配置 MUST 显式记录各关节的刚度、阻尼、力矩上限与 armature。

#### Scenario: 增益与 G1 不同
- **WHEN** 比较 HU_D04 与 G1 同一关节的刚度
- **THEN** 两者不相等（HU_D04 腿部刚度显著高于 G1）

#### Scenario: 增益可追溯
- **WHEN** 审查配置
- **THEN** 每个受控关节的刚度、阻尼、力矩上限均有明确来源标注

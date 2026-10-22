## Purpose

把 GMR 重定向产出的 HU_D04 动作（31 关节、42 连杆）转换为 PHP 训练管线可直接消费的数据格式（29 关节、32 连杆），并配对可复用的地形数据，使 HU_D04 无需 G1 专用动作库即可获得训练素材。

## ADDED Requirements

### Requirement: 接受 GMR 的 BeyondMimic 输出

管线 MUST 能读取 GMR 对 `limx_oli_edu` 导出的 BeyondMimic 格式动作文件，至少包含 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`、`joint_names`、`body_names` 九个字段。四元数 MUST 按 wxyz 解读。

#### Scenario: 读取 GMR 原始输出
- **WHEN** 输入一个 GMR 导出的 HU_D04 动作文件
- **THEN** 管线读出 31 个关节、42 个连杆，且 `joint_pos` 的时间长度与 `body_pos_w` 一致

#### Scenario: 字段缺失时报错
- **WHEN** 输入文件缺少上述任一必需字段
- **THEN** 管线拒绝该文件并明确指出缺失字段

### Requirement: 裁剪头部关节到 29 维

转换后的动作数据 MUST 只包含 29 个关节，头部两个关节 MUST 被移除。移除 MUST 基于关节名而非列索引，以避免顺序假设。

#### Scenario: 头部被移除
- **WHEN** 转换含 31 个关节的动作
- **THEN** 输出含 29 个关节，且 `joint_names` 中不含 `head_yaw_joint` 与 `head_pitch_joint`

#### Scenario: 基于名字裁剪
- **WHEN** 输入关节顺序与预期不同
- **THEN** 管线仍按关节名裁剪，输出关节集合正确

### Requirement: 连杆表映射到 32 项

转换后的连杆数据 MUST 恰好包含 32 项，映射规则 MUST 与训练配置的连杆语义对齐一致。映射 MUST 覆盖 `pelvis`→`base_link`、`torso_link`→`waist_pitch_link`、左右脚接触点→`contact_foot_center_*`，并排除其余 HU_D04 特有连杆。

#### Scenario: 输出连杆数量
- **WHEN** 转换含 42 个连杆的动作
- **THEN** 输出含 32 个连杆，顺序与训练配置一致

#### Scenario: 映射失败时报错
- **WHEN** 某个必需连杆在输入中不存在
- **THEN** 管线报错并指出缺失的连杆名，不产生静默错位的结果

### Requirement: 关节顺序重排到训练约定

输出关节顺序 MUST 显式重排为训练配置声明的顺序。重排 MUST 由名字驱动的映射表完成，MUST NOT 假设输入顺序即输出顺序。

#### Scenario: 顺序被重排
- **WHEN** 输入关节顺序与训练约定不同
- **THEN** 输出关节顺序符合训练约定，且每个关节的数据仍对应正确的物理关节

#### Scenario: 重排表可校验
- **WHEN** 校验重排表
- **THEN** 长度与实际关节数一致，索引无重复且在范围内

### Requirement: 产出可通过训练数据校验

输出文件 MUST 通过 PHP 现有的动作数据校验（九个字段齐全、配对的地形存在、数值有限）。输出 MUST 可直接作为训练数据源被引用。

#### Scenario: 通过校验
- **WHEN** 用现有校验逻辑检查转换后的数据目录
- **THEN** 校验通过，无缺失字段、无配对错误

#### Scenario: 地形与动作配对
- **WHEN** 转换后的动作与一个地形文件配对
- **THEN** 两者在训练时被一并读取，配对关系在目录结构中显式可见

### Requirement: 复用机器人无关的地形数据

地形数据 MUST 被视为与机器人无关并可直接复用既有地形文件。管线 MUST NOT 修改地形内容，但 MUST 在转换报告中记录地形的实际几何参数。

#### Scenario: 地形原样复用
- **WHEN** 为 HU_D04 动作配对一个既有地形文件
- **THEN** 地形文件内容不被修改，且可被训练流程正确读取

#### Scenario: 地形参数被记录
- **WHEN** 完成一次转换
- **THEN** 转换报告记录该地形的障碍高度与尺寸，供难度评估使用

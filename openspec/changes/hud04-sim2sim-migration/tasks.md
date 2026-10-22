## 1. HU_D04 模型接入与验证

- [x] 1.1 定位 sim2sim 简化版 MJCF（串联踝版 `hu_description/xml/HU_D04_01.xml`），复制到 PHP 可访问的资产路径，用 MuJoCo 加载并打印 `nq/nv/nu/nbody`，验证解析成功且含 root free-joint
- [x] 1.2 验证串联踝关节存在：断言 `left_ankle_pitch_joint`/`left_ankle_roll_joint`/`right_ankle_pitch_joint`/`right_ankle_roll_joint` 为 hinge 类型且属受控关节集合
- [x] 1.3 验证 A/B 并联 achilles 体为纯视觉（无碰撞几何、不参与受控关节集）
- [x] 1.4 编写 HU_D04 模型加载适配器（对齐 PHP sim2sim 骨架接口），在 MuJoCo 中渲染并确认模型无穿透/无爆炸
- [x] 1.5 记录 HU_D04 全部受控执行器列表（名称 + 索引 + 默认角 + 关节限位），输出为配置文件

## 2. 头部自由度固定

- [x] 2.1 从 `limx_rl_lab/deploy/sim2sim/configs/limx_walk_mimic_sdk.yaml` 提取 `head_yaw_joint`/`head_pitch_joint` 的 default 角与 PD 增益
- [x] 2.2 实现 `head_hold_targets` 等价机制：策略关节列表排除头部 2 DOF，控制器在每步控制中对头部单独发 default 目标；验证策略关节列表为 29 且不含 `head_*`
- [x] 2.3 验证头部固定效果：仿真运行中记录头部关节角，确认其稳定在 default ±0.01 rad

## 3. G1→HU_D04 关节映射与动作适配

- [x] 3.1 确认 G1 student ONNX 的输出语义（归一化 action vs 关节角）：加载 ONNX 做单步推理，打印输出范围与形状，记录结论
- [x] 3.2 从 `wbt_training/config_values/robot_config.py` 读取 G1 29-DOF 关节顺序，从 HU_D04 配置读取其策略关节顺序，构建静态 `joint_ids_map` 映射表
- [x] 3.3 实现关节重排层：按 `joint_ids_map` 把 G1 输出重排到 HU_D04 策略关节顺序，启动时校验映射表长度与索引范围并在不合法时报错
- [x] 3.4 实现动作缩放适配：归一化输出 × `action_scale` + default 角 → HU_D04 PD 目标；若 ONNX 输出为关节角则将 scale 置 1.0，并在配置中记录选择依据
- [x] 3.5 单关节验证：逐个给 HU_D04 关节发小角度指令（0.1 rad），观察对应关节动作方向与幅度符合预期

## 4. sim2sim 迁移运行与验证

- [x] 4.1 按 `head_hold_targets` 的接口约定，将 HU_D04 模型 + 迁移层接入 sim2sim 运行
- [x] 4.2 运行 HU_D04 sim2sim 驱动 G1 策略冒烟测试，确认仿真启动、策略推理、控制下发全链路无异常（可先用降低增益/缩放保稳）
- [x] 4.3 分阶段验证策略行为：先站立静止 → 再低速前进 → 再转向，记录每阶段成功/失败的观测现象
- [x] 4.4 翻箱能力测试：在带箱体地形上运行，记录 HU_D04 是否能执行翻越动作及其成功率（依赖任务组 5 的深度相机接入）
- [x] 4.5 整理迁移可行性结论报告：记录关节映射表、动作缩放结论、各测试阶段结果、以及上真机所需后续工作（SDK 集成、ROS2 深度相机、安全层）清单

## 5. HU_D04 深度相机接入

- [x] 5.1 从 `HU_D04_01.urdf` 提取 D435i 挂载几何（`waist_camera_joint`：父 `waist_pitch_link`，xyz `0.092 0.0175 0.2751`，rpy `0 2.1818 0`），在加载的 MJCF 中确认父 link 存在，输出相机挂载配置
- [x] 5.2 编写 HU_D04 深度相机配置（对齐 Holosoma `RsD435iConfig` 参数：58×87 裁剪后分辨率、HFOV、range），并建立 `ray_cast_bodies`：HU_D04 各 link → `HU_D04_description/meshes/HU_D04_01/*.STL` 的映射
- [x] 5.3 验证射线求交体映射：所有条目在 MJCF 中存在对应 body 且 mesh 文件存在，解析失败时报错
- [x] 5.4 渲染一帧 HU_D04 深度图，验证其裁剪后为 58×87 且与 G1 同规格（对齐/尺寸校验），并确认默认姿态下含自身遮挡命中
- [x] 5.5 将深度图送入未改动的 depth backbone（输入 `[1,58,87]`），验证输出 `[1,32]` depth latent 且形状正确
- [x] 5.6 把 depth latent 接入 student 观测的深度块，验证 140 维观测完整且控制回路可用；无深度时的零填充回退仍可运行
- [x] 5.7 用真实 depth latent 重跑站立测试（4.3），记录是否站立成功，对比零深度结果

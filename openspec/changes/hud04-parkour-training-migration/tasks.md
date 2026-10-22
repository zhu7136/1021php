## 1. GMR 动作数据通路

- [x] 1.1 按 `GMR/README.md` 创建 `gmr` conda 环境并 `pip install -e .`，验证 `python -c "import general_motion_retargeting"` 成功
- [x] 1.2 初始化 `GMR/assets/humanoid-description` submodule，验证 `HU_D04_01_vis.xml` 存在且可被 MuJoCo 加载
- [x] 1.3 准备 SMPL-X 体模与一个测试动作（SMPL-X 或 LAFAN1 BVH），验证 GMR 能读入
- [x] 1.4 运行 `smplx_to_robot.py --robot limx_oli_edu` 生成一个动作，验证输出 npz 含 31 关节、42 连杆、九个必需字段
- [x] 1.5 记录 GMR 输出的真实 `joint_names` 与 `body_names` 完整清单，写入转换层的映射基准

## 2. 数据转换层

- [x] 2.1 实现关节裁剪：按名移除 `head_yaw_joint`/`head_pitch_joint`，验证输出 29 关节且不含 `head_*`
- [x] 2.2 实现关节重排到 G1 顺序，验证每个关节的数据仍对应正确物理关节（用关节角范围做冒烟对比）
- [x] 2.3 实现连杆映射 42→32（`pelvis`→`base_link`、`torso_link`→`waist_pitch_link`、`*_foot_contact_point`→`contact_foot_center_*`），验证映射表无重复无遗漏
- [x] 2.4 加断言：`set(hu_d04_joint_names) == set(G1_29DOF_JOINT_NAMES)` 且两侧长度均为 29，验证故意注入错误名字时断言触发
- [x] 2.5 实现地形配对：把转换后的动作与既有 `terrain.npy` 组成目录，验证可通过 `validate_bundle`
- [x] 2.6 输出转换报告，记录地形障碍高度与尺寸，验证报告含 climb-76 的 0.382 m 台阶高度

## 3. HU_D04 配置与资产

- [x] 3.1 从 `HU_D04_01.urdf` 提取 31 关节的限位、力矩、速度上限，输出为可审查的中间表，验证与官方手册数值一致
- [x] 3.2 从 `limx_walk_mimic_sdk.yaml` 提取 31 关节的 Kp/Kd/default_angles，验证腿部刚度与 G1 不同（髋俯仰 220 vs 40.2）
- [x] 3.3 新增 `hu_d04` RobotConfig（`dof_names` 29 项、`body_names` 32 项、限位、增益），验证 `len(dof_names)==29` 且 `len(body_names)==32`
- [ ] 3.4 实现 `build_hu_d04_implicit_actuators`，验证 29 个受控关节各自获得正确的 armature 与摩擦参数
- [x] 3.5 把 URDF/MJCF/USD 复制到 `data/robots/hu_d04/`，验证 IsaacSim 能通过 `urdf_file` 路径加载
- [ ] 3.6 启动仿真并断言 `dof_names`/`body_names` 与配置逐项相等，验证不触发 `isaacsim.py:861-862` 的一致性断言

## 4. HU_D04 训练预设

- [x] 4.1 基于 `wbt/g1/` 复制生成 `wbt/hu_d04/` 六模块，验证模块可导入且无 G1 残留名字
- [x] 4.2 参数化 `terrain.py` 的 `body_name_ref` 与连杆白名单正则，验证 HU_D04 preset 下解析为 `base_link`/`waist_pitch_link`/`contact_foot_center_*`
- [x] 4.3 新增 `exp:hu-d04-wbt-terrain-ref-noscale` preset 并在 `DEFAULTS` 注册，验证按名可调用
- [ ] 4.4 配置 noscale 动作缩放（`action_scale=1.0`、`action_scales_by_effort_limit_over_p_gain=False`），验证每个关节的实际缩放为 1.0
- [ ] 4.5 处理头部被动摆动：为头部关节加阻尼或评估 weld 方案，验证深度图中头部不产生明显遮挡
- [ ] 4.6 短程 smoke 训练（数百步），验证关节位移量级合理、奖励不早期崩塌
- [ ] 4.7 对比验证 G1 preset 行为不变：跑 G1 短程训练，确认输出与本变更前一致

## 5. 教师训练

- [ ] 5.1 用 locomotion 数据训练教师，验证检查点含策略权重与观测归一化统计
- [ ] 5.2 用 low-step / high-step 数据训练两个教师，验证各自收敛
- [ ] 5.3 用 low-climb / high-climb 数据训练两个翻越教师，验证能完成越障动作
- [ ] 5.4 统计翻越成功率并记录，验证 0.382 m 台阶在 HU_D04 上可达（必要时按设计加高并重测）

## 6. 学生蒸馏与导出

- [x] 6.1 配置 HU_D04 深度相机（挂 `waist_pitch_link`、offset `(0.092, 0.0175, 0.4336)`、下俯 35°、58×87），验证渲染出的深度图为 58×87
- [x] 6.2 新增 `exp:hu-d04-wbt-terrain-warp-distill-finetune` preset，验证按名可调用
- [ ] 6.3 单教师蒸馏，验证学生观测为 140 维且含 32 维深度潜变量
- [ ] 6.4 多教师蒸馏（5 个教师 + 5 个数据集），验证数量不匹配时报错、匹配时按下标配对
- [ ] 6.5 验证保存检查点时自动导出 `student.onnx` 与 `depth_backbone.onnx`
- [ ] 6.6 校验 ONNX 元数据：`joint_names` 为 HU_D04 的 29 关节、`action_scale`/`default_joint_pos` 与 G1 不同
- [ ] 6.7 验证导出失败不中断训练且产生可见警告（可人为制造导出错误测试）

## 7. sim2sim 集成验证

- [ ] 7.1 用导出的 HU_D04 ONNX 对复用 `hud04_sim2sim/` 工具跑站立，验证能稳定站立（对比既有报告中的跌倒结果）
- [ ] 7.2 跑前进与转向，验证方向键指令有响应
- [ ] 7.3 在带箱体地形上跑翻越，记录成功率并与教师训练阶段结果对比
- [ ] 7.4 输出最终可行性报告，记录各阶段结果、标定参数、以及上真机的遗留工作清单

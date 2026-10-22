## Context

动机见 `proposal.md - Why`。此处只记录影响技术路线的实测约束。

**已完成的既有工作**：`hud04-sim2sim-migration`（25/25 完成）已打通推理接口——HU_D04 的 MuJoCo 加载、29 关节映射、头部固定、深度相机接入全部 PASS，但 G1 策略零样本迁移在站立阶段即失败。

**关键实测数据**（本次调查重新核验，修正了既有报告的一处错误）：

| 项 | G1 | HU_D04 | 说明 |
|---|---:|---:|---|
| 总质量 | 33.6 kg | 52.9 kg | +58% |
| 大腿（髋→膝） | 0.177 m | 0.198 m | — |
| 小腿（膝→踝） | 0.300 m | 0.385 m | — |
| 髋→踝（竖直投影） | 0.477 m | 0.583 m | **1.22×**（既有报告写 1.5×，实测有误） |
| 膝关节 Kp | 99.1 | 240 | 2.4× |
| 髋俯仰 Kp | 40.2 | 220 | **5.5×** |
| 腿部 Kp 均值 | — | — | **3.2×** |

**动作缩放公式**（`managers/action/terms/joint_control.py:59-74`）：

```
action_scales[i] = control.action_scale * effort_limit[i] / stiffness[i]
```

代入实测增益后，HU_D04 的实际缩放只有 G1 的 10%–50%（髋偏航 0.10×、踝 0.18×、髋俯仰 0.29×）。这是零样本迁移失败的直接数值解释。

**关节与连杆的结构性差异**：

```
关节：29 个名字集合完全相同，4 个手腕关节顺序不同（roll/yaw 互换）
连杆：G1 = 32，HU_D04 = 42
       pelvis            -> base_link
       torso_link        -> waist_pitch_link
       *_foot_contact_point -> contact_foot_center_*  (HU_D04 另有 heel/tip)
       HU_D04 独有：head_*、*_hand_contact、*_hand_manip
```

**动作数据实测**（正确解读 `joint_pos` 为 36 维 qpos = 7 root + 29 dof）：

| 数据集 | 越界关节 | 最大超出 |
|---|---|---:|
| low-climb-76 | `wrist_yaw` | 0.491 rad (28°) |
| high-climb-76 | `shoulder_roll` | 0.025 rad |
| low-step | 无 | 0 |
| high-step | `ankle_roll` | 0.320 rad |

膝关节仅超 0.012 rad（0.7°），不是主要障碍。**G1 动作数据不可直接复用，但 GMR 可绕过整个问题。**

**GMR 能力实测**：`GMR/README.md` 确认支持 `limx_oli_edu`，6 种输入格式 IK 配置齐全，直接输出 BeyondMimic 格式（九字段与 PHP 校验完全对应）。

## Goals / Non-Goals

**Goals:**
- 让 HU_D04 具备可训练的 29-DOF 配置，且与 G1 在观测维度上同构（蒸馏复用现有 140 维契约）
- 建立 GMR → PHP 的动作数据通路，摆脱对 G1 专用动作库的依赖
- 用最小侵入方式接入：G1 路径行为完全不变

**Non-Goals:**
- 不实现真机部署（limx bridge、ROS2 相机驱动、安全层）
- 不追求 HU_D04 与 G1 的权重级兼容（形态学差异决定了必须重训）
- 不重建 G1 的动作库或修改 G1 的 `database_g1_*`

## Decisions

### D1: 头部排除放在配置层，而非数据层或控制器层

**选择**：`hu_d04.dof_names` 只列 29 个关节，头部两关节完全不出现。

**理由**：`isaacsim.py:397-442` 已原生支持——执行器只为 `dof_names` 中的关节创建，`find_joints` 只解析这些名字。头部仍存在于 URDF 与物理仿真中但无驱动，语义上等价于 BeyondMimic 的「被动存在 + 部署端 `head_hold_targets` 托住」。

**备选**：
- 策略输出 31 维再丢弃 → 浪费 2 维输出且破坏 140 维观测契约
- 在控制器层补发头部目标 → 训练侧不需要，属部署侧职责

**注意**：仿真中头部无驱动会像钟摆摆动，可能遮挡深度视野。缓解见 R2。

### D2: GMR 保留 31 维，转换层负责裁剪

**选择**：GMR 不排除头部（按用户决定），由新增转换层把 31/42 裁成 29/32。

**理由**：GMR 的 `frozen_joints` 框架当前无机器人使用（`_test_frozen_joints.py` 确认），强行改 GMR 会偏离上游。转换层是本仓库自有代码，可控且可测试。

**备选**：改 GMR 的 IK 配置排除头部 → 与上游分叉，升级成本高。

### D3: 关节顺序统一重排为 G1 顺序

**选择**：转换层输出时把关节重排成 `G1_29DOF_JOINT_NAMES` 的顺序。

**理由**：PHP 的 `ISAACLAB_TO_MUJOCO_DOF` 等重排索引是显式的，顺序统一后这些索引可直接复用，减少出错面。

**备选**：保留 HU_D04 顺序，运行时按名对齐 → 要多维护一套索引，且 4 个手腕关节的差异易被静默放过。

**防御**：转换层与 `hu_d04` 配置都必须加断言 `set(hu_d04_names) == set(g1_names)`，并显式声明顺序差异。

### D4: 动作缩放采用 noscale 变体

**选择**：HU_D04 训练使用 `action_scale=1.0` + `action_scales_by_effort_limit_over_p_gain=False`。

**理由**：`effort/Kp` 缩放会引入 3.2× 的增益耦合，而 HU_D04 的 Kp 来自逐际的部署标定（`limx_walk_mimic_sdk.yaml`），与训练仿真并非同一标定体系。用 noscale 让策略直接输出关节增量，增益差异由 PD 环独立承担，语义更干净。

**备选**：沿用 `action_scale=0.25` + `effort/Kp` → 实测会导致动作幅度压到 G1 的 10%–50%，策略需重新学习幅度，收敛慢且易饱和。

**风险**：noscale 下动作幅度更依赖 `default_joint_pos` 的准确性，必须用 HU_D04 自身的默认角而非 G1 的。

### D5: 复用 terrain.npy，不重做地形

**选择**：地形数组被视为机器无关资产直接复用。

**理由**：`terrain.npy` 的 shape 为 `(boxes, 10, 10)`，只含障碍位姿与尺寸，无机器人信息；`validate_bundle` 也只校验 shape 与有限性。

**但需记录难度差异**：

| 台阶 | 占 G1 腿长 | 占 HU_D04 腿长 |
|---|---:|---:|
| step 0.176 m | 36.9% | 30.2% |
| climb-76 0.382 m | 80.0% | 65.6% |

同样的台阶对 HU_D04 相对更容易，因此 G1 的地形可作为起点，后续可考虑加高。

### D6: 逐参数分离而非新建并行目录树

**选择**：在 `holosoma/config_values/robot.py` 新增 `hu_d04` 条目，`wbt/hu_d04/` 由 `wbt/g1/` 复制改名。

**理由**：现有代码通过 `replace` 链式覆盖构建 preset，机器人无关部分（exporter 反射、观测 term 动态计算）已可复用。新建平行树可让 G1 与 HU_D04 互不干扰，回滚成本接近 0。

**代价**：`wbt/hu_d04/` 六个模块存在与 `wbt/g1/` 的重复。接受该重复，因为两机器人的连杆语义与奖励权重需要独立演进。

## Risks / Trade-offs

**R1: 大量连杆/关节名硬编码在多个位置，改漏会静默错位。** `terrain.py:93` 的 `body_name_ref=["pelvis"]`、`terrain.py:311-315` 的正则白名单、`wbt/g1/command.py:19-35` 的 14 个连杆名，若不改会**解析成功但语义错误**。→ 缓解：`hu_d04` 配置启动时断言连杆/关节集合与仿真器解析结果逐项相等（`isaacsim.py:861-862` 已有断言可兜底）；新增单元测试覆盖每个硬编码点。

**R2: 头部无驱动会在仿真中摆动，干扰深度图。** → 缓解：在训练 URDF 中为头部加阻尼；若仍干扰，可选在 MJCF 中把头部固定为 weld（仅训练侧，不影响真机）。

**R3: `action_scale` 标定不当导致训练不收敛。** 实测增益差异达 3.2×，若仍有幅度问题会表现为早期奖励崩塌。→ 缓解：先跑短程 smoke（数百步），确认关节位移量级合理再投入长期训练；必要时回退到 `effort/Kp` 并重新标定 `action_scale`。

**R4: 缺少攀爬类人体动作素材。** GMR 需要 SMPL-X/BVH 输入，仓库不自带体模与动作。→ 缓解：用 GVHMR 从攀爬视频生成动作（GMR 支持该路径）；或先用 LAFAN1 等公开 BVH 验证通路，再补攀爬素材。

**R5: HU_D04 的 URDF 是否被 IsaacSim 正确解析未知。** 仓库 `data/robots/` 只有 g1/t1，`urdf_file` 路径必须正确。→ 缓解：把 URDF/USD 复制进 `data/robots/hu_d04/` 后，第一步任务就是模型加载验证，失败则先解决资产问题再谈训练。

**R6: 42→32 的连杆映射可能丢失接触信息。** G1 每脚 1 个接触点，HU_D04 有 3 个（center/heel/tip），选 `center` 可能不足以表达翻越时的足尖接触。→ 缓解：先用 `center`；若翻越成功率低，改用 `tip` 或扩展为多点接触。

## Migration Plan

分阶段推进，每阶段有独立验收点，失败可停在当前阶段：

1. **数据通路验证**（不改训练代码）：GMR 跑通一个动作 → 转换层输出 29/32 → 通过 `validate_bundle`
2. **模型加载验证**：`hu_d04` RobotConfig + 资产入库 → IsaacSim 加载 → 断言全过
3. **单技能短程训练**：locomotion 教师训练数百步 → 确认动作幅度合理
4. **完整教师训练**：5 个技能（locomotion / low-step / high-step / low-climb / high-climb）
5. **学生蒸馏**：多教师蒸馏 → ONNX 导出
6. **sim2sim 验证**：复用 `hud04_sim2sim/` 既有工具

**Rollback**：删除 `hu_d04` 相关配置与 preset 注册，G1 路径不受影响。

## Open Questions

- HU_D04 的 armature 参数公开资料未给出，暂用与 G1 相近的经验值，待真机辨识实验校准。此项不影响规格与任务拆解。
- 翻越技能的目标台阶高度尚未确定。当前 climb-76 的 0.382 m 对 HU_D04 仅占腿长 65.6%，是否加高到 0.46 m 左右需在阶段 3 之后按实际表现决定。

# HU_D04 sim2sim 迁移可行性报告

任务 4.5 产物。记录 G1 → 逐际 HU_D04 策略迁移的技术结论、已验证事实、
以及上真机所需的后续工作。

## 1. 结论摘要

| 项目 | 结论 |
|---|---|
| HU_D04 模型接入 MuJoCo | ✅ 成功（31 执行器，串联踝） |
| 头部 2 DOF 固定 | ✅ 成功（策略空间 29 = G1） |
| G1→HU_D04 关节映射 | ✅ 成功（29 关节同名，纯排列） |
| 动作缩放适配 | ✅ 成功（scale=1.0，输出为归一化 action） |
| ONNX 推理 + 控制链路 | ✅ 成功（无 NaN/爆炸） |
| HU_D04 D435i 深度相机接入 | ✅ 成功（58×87 深度图 → depth latent 32） |
| 零观测/有深度下站立 | ❌ 失败（形态学差异，非接口问题） |
| 翻箱测试 | ⛔ 受阻（同站立根因） |

**核心判断**：机械/接口层面的迁移**全部打通**（G1 策略零改动可驱动 HU_D04
的 29 个 body 关节、深度感知链路完整），但**策略本身不能零样本迁移**——根因是
**形态学差异**（HU_D04 腿长/质量与 G1 差异大），不是接口或映射问题。

## 2. 已验证技术事实

### 2.1 MJCF 变体选择

HU_D04 提供两个 MJCF：

| 文件 | nq/nv/nu/nbody | 踝 | 用途 |
|---|---|---|---|
| `HU_D04_01.xml` | 68/61/31/55 | 并联 A/B（achilles 电机） | 完整机构描述 |
| `HU_D04_01_gmr.xml` | 38/37/31/43 | **串联 pitch/roll** | 训练/sim2sim |

**选择 `HU_D04_01_gmr.xml`**（已复制为 `hud04_sim2sim/assets/HU_D04_01_serial.xml`）：
串联踝 `ankle_pitch/roll` 是受控 hinge，与 G1 运动学一致；并联 A/B 仅在完整
版里作为电机存在，sim2sim 不需要。

### 2.2 执行器列表（31）

```
腿部 12: hip_pitch/roll/yaw + knee + ankle_pitch/roll  (L/R)
腰   3: body_waist_yaw/roll/pitch
头   2: head_yaw, head_pitch          <- 固定
臂  14: shoulder_pitch/roll/yaw + elbow + wrist_yaw/pitch/roll  (L/R)
```

### 2.3 头部固定

从 `limx_walk_mimic_sdk.yaml` 提取：

| joint | default | kp | kd |
|---|---|---|---|
| `head_yaw_joint` | 0.0 | 20.0 | 0.8 |
| `head_pitch_joint` | 0.0 | 20.0 | 0.8 |

实现于 `head_lock.py`（`HEAD_HOLD_TARGETS`）。验证（`head_lock_check.py`）：
从 0.3 rad 偏移起步，3000 步后头部稳定在目标 ±0.004 rad。

### 2.4 关节映射

**G1 与 HU_D04 的 29 个 body 关节名称完全相同**，只有顺序不同：

```
G1 顺序:      左右腿各 6，腰 3，左右臂各 7
HU_D04 顺序:  腿 L/R 交错，腰位序 2/5/8，腕 yaw/pitch/roll
```

映射是**纯排列**，实现于 `joint_mapping.py`（`JOINT_IDS_MAP`，29 项）。
启动时校验：无重复、长度一致、索引在范围内。

### 2.5 动作缩放

从 G1 student ONNX 元数据读取：

- `action_scale` = 全 1.0（逐关节张量）
- 输出 `actions` shape `[1,29]` = **归一化 action**（非关节角）
- Holosoma 推理：`q_target = action * action_scale + default_joint_pos`

配置 `configs/hud04_action_adapter.json`：scale 用 G1 元数据（排列到 HU_D04 顺序），
default 用 LimX mimic 配置（去头后的 29 项）。

### 2.6 单关节验证

`single_joint_check.py`：对每个关节施加恒定正扭矩，验证 qpos 正向移动。

**结果：29/29 执行器正确驱动对应关节，符号一致。**

### 2.7 观测空间

G1 student obs(140) 分解：

```
robot_anchor_projected_gravity(3)
base_ang_vel(3)
dof_pos(29) + dof_vel(29) + actions(29)
velocity_command(15, one-hot)
depth_latent(32)  <- 来自 depth_backbone.onnx（输入 58x87 深度图）
```

**除 depth_latent 外全部可从 HU_D04 状态构建**（`observation.py` 已实现）。
无 body-frame 运动参考需求，因此不需要 14 个 G1 body 名的对应表。

## 3. 测试结果

### 3.1 机械集成（PASS）

`runner.py --mode held_pose`：模型加载，adapter+head-lock 输出合法 31 执行器
命令，物理步进 4s 无 NaN/爆炸，头部保持目标。

### 3.2 闭环冒烟（PASS）

`runner.py --mode standing --policy student.onnx`：ONNX 推理 + 重排 + 缩放
+ 控制下发全链路执行，无异常。

### 3.3 站立测试（FAIL，根因：形态学差异）

| 配置 | base_z (起始→结束) | 结果 |
|---|---|---|
| stand cmd, scale ×1.0, 零深度 | 1.0 → 0.32 | 跌倒 |
| stand cmd, scale ×0.3, 零深度 | 1.0 → 0.23 | 跌倒 |
| fwd cmd, scale ×0.3, 零深度 | 1.0 → 0.09 | 跌倒 |
| stand cmd, **真实深度 latent** | 1.0 → 0.24 | 跌倒 |
| stand cmd, 真实深度 + G1 默认角偏移修正 | 1.0 → 0.54 | 跌倒（改善但仍失败） |

**根因：形态学差异，非接口问题。** 已排除的假设：

1. ❌ 不是缺深度 —— 接入真实 depth latent 后仍跌倒（depth 链路已证明正确：58×87 → latent 32）
2. ❌ 不是关节映射错误 —— 单关节验证 29/29 正确、映射是纯排列
3. ❌ 不是默认角偏移写错 —— 已修正为策略参考默认角（见下），有改善但不足以站立
4. ✅ **是 G1 与 HU_D04 的形态学差异**：

| 指标 | G1 | HU_D04 |
|---|---|---|
| 总质量 | ~35 kg | 52.9 kg |
| 大腿长 (hip→knee) | ~0.19 m | 0.467 m |
| 小腿长 (knee→ankle) | ~0.30 m | 0.385 m |
| 站立基座高度 | ~0.79 m | 1.016 m |

腿长约 1.5 倍差异导致：同一组关节角下足端位置完全不同，且
`dof_pos`/`gravity` 观测分布外（OOD），策略输出极端动作（range ±22 rad）。

**重要修正**：`default_dof_angles` 偏移必须用**策略自身训练的参考默认角**
（G1 元数据 `default_joint_pos`），而非 HU_D04 机器人自身默认姿态。用后者会让
每个关节产生固定偏移（尤其肘部 -0.8 rad），加剧跌倒。已修正
`configs/hud04_action_adapter.json` 的 `default_dof_angles` 为 G1 元数据。

### 3.4 翻箱测试（已记录，失败）

场景：0.6×0.6×0.3 m 箱体置于 x=1.2 m，HU_D04 站在箱后，真实深度 + 前进命令。

| 指标 | 值 |
|---|---|
| 前进进度 | -0.035 m（几乎无前进） |
| x 最大 | 0.057 m（远未到箱体 1.2 m） |
| 基座最高 | 1.0 m（未抬升） |
| 成功翻越 | ❌ 否 |
| 失败步数 | 167（基座塌陷） |

**结论**：HU_D04 无法翻越箱体。根因同 3.3（形态学差异），非接口问题。
深度感知链路本身正常（见 3.5），但策略因形态学差异无法输出合理动作。

### 3.5 HU_D04 深度相机接入（PASS）

| 检查 | 结果 |
|---|---|
| 相机挂载 | `waist_pitch_link`（对标 G1 的 `torso_link`），偏移来自 URDF |
| 深度图分辨率 | 58×87（与 G1 同规格） |
| depth backbone | 输入 `[1,58,87]` → 输出 `[1,32]` ✅ |
| 自身遮挡 | 默认姿态下 5036 个近场像素（robot 自身可见）✅ |
| 观测接入 | depth latent 填入 140 维观测尾部 32 维 ✅ |

## 4. 上真机后续工作清单

| # | 工作项 | 说明 |
|---|---|---|
| 1 | HU_D04 D435i 相机挂载配置 | 相机 frame（torso 或头部）+ 偏移 + `ray_cast_bodies` 映射 |
| 2 | 深度图像重跑站立/翻箱测试 | 用真实 depth_latent(32) 替换零填充 |
| 3 | 逐际 SDK 集成 | `limxsdk` 的 `parallel_solve_required=True` 标志（真机并联解算在固件） |
| 4 | ROS2 深度相机驱动 | 真机 D435i 数据流接入 PHP 推理栈 |
| 5 | 安全/急停层 | 关节限位、扭矩限制、看门狗 |
| 6 | PD 增益标定 | G1 kp/kd vs HU_D04 推荐值（240/220/...）需在真机验证 |
| 7 | 质量/摩擦参数对齐 | HU_D04 实际质量分布可能与 MJCF 有偏差 |

## 5. 交付物清单

| 文件 | 用途 |
|---|---|
| `hud04_sim2sim/assets/HU_D04_01_serial.xml` | 串联踝 MJCF 变体 |
| `hud04_sim2sim/assets/meshes -> ...` | 网格软链接 |
| `hud04_sim2sim/configs/hud04_actuators.json` | 31 执行器表（索引/关节/限位） |
| `hud04_sim2sim/configs/hud04_action_adapter.json` | scale + default（G1 策略参考） |
| `hud04_sim2sim/model_loader.py` | 模型加载 + 不变量校验 |
| `hud04_sim2sim/head_lock.py` | 头部固定目标 |
| `hud04_sim2sim/head_lock_check.py` | 头部固定仿真验证 |
| `hud04_sim2sim/joint_order.py` | 31/29 关节顺序定义 |
| `hud04_sim2sim/joint_mapping.py` | G1→HU_D04 重排层 |
| `hud04_sim2sim/action_adapter.py` | 动作缩放 + default 偏移 |
| `hud04_sim2sim/single_joint_check.py` | 单关节方向验证 |
| `hud04_sim2sim/observation.py` | 140 维观测构建 |
| `hud04_sim2sim/runner.py` | sim2sim 运行入口 |
| `hud04_sim2sim/depth_camera.py` | HU_D04 D435i 深度相机配置（来自 URDF）+ 射线求交体映射 |
| `hud04_sim2sim/depth_render.py` | 基于 MuJoCo 的深度渲染（58×87 → backbone → latent 32） |

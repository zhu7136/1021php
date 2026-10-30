# HuD04 单 episode 终止诊断（CPU 验证版，尚未经过原生 GPU 验证）

## 目标与边界

诊断 3,686-update 学生在 high-step 固定留出案例中的首次终止原因。
不训练、不续训、不修改物理/动作/终止阈值；不认证五技能成功或收敛。
此版本只使用已有 high-step 单案例输入隔离器；不宣称覆盖所有技能。

源学生：`zhul136-university-of-science-and-technology-of-china/HuD04_WBT/b6826eul`。
源检查点 SHA256：`e4e92fd80abacc3023339dd379d48e3f7eee64eb63ff84928cf0c6e5a795d3af`。
代码基于已使用的 HuD04 私有输入交付/仿真代码提交
`6890742222958de814cad20589e5920a86f61cb5`，不是上游 G1 直接部署入口。

## 已发现的证据缺口

- high-step 旧回放：1,000 步 / 20 秒，8 次 done/reset。
- low-climb 旧回放：1,000 步，6 次 reset，包含一次零动作初始化预热。
- 旧诊断主要在 `env.step()` 返回后取样；终止时机器人可能已经自动重置。
- 旧数据没有逐次保存所有原生 termination term，不能从它证明 8/6 次 reset 都是摔倒。
- 退出异常和物理任务结果必须分开报告；既有 native shutdown 问题未修复。

## 新行为

`hud04_episode_probe.py` 包装原生终止检查和各 term 的原有调用。
每个 term 仍调用一次，返回原来的 tensor；不重复执行条件。
在 native auto-reset 前保存 term 名、failure/timeout 掩码、root 状态、关节状态和参考帧。
每次写入 `termination-checks.jsonl` 并 flush，以保留退出异常前的部分证据。

`hud04_single_episode_algo.py`：

- 显式区分 `student` / `teacher` 角色；学生角色禁止教师 action 调用；
- 禁止 optimizer.step、训练 step 和训练 rollout；
- 只用一个物理环境，第一次原生终止或 timeout 后结束，不继续执行重置后的 episode；
- 原生 reset_all 的零动作预热单独记录，预热终止则标记初始化无效；
- 最多 1,000 次策略动作，参考帧回绕也结束；
- 比较前后模型摘要，输出 `single-episode-result.json`；
- `task_success=null`，不把 timeout、参考结束或活到时限当成任务成功。

环境可能已在最后一次 step 内完成自动重置；入口不禁用它，但绝不再执行下一次策略动作。
`learn()` 之前 setup 内部的 reset 不在本探针覆盖范围内，结果中显式标记。

## CPU 验证

```bash
python -m unittest -v test_hud04_episode_probe test_hud04_single_episode_algo
```

24 项通过：包括重置前状态不被覆盖、原生 term 单次调用、stateful reset 转发、
首次 done 后停止、学生/教师动作隔离、初始化失败分离和禁止训练。
这些是 fake-environment CPU 测试，不是 IsaacSim 物理验证。

## 接入既有有界 GPU worker

`obstacle_video_worker.py` 新增允许的 child `hud04_eval_child.py`；既有视频默认路径不变。
该 child 复用原有 hash-verified 输入、固定案例配置和原生生命周期，不输出 RGB 视频。
深度相机仍由原生学生配置提供，不能用 RGB recorder 替代。

必须由既有调度器重新生成当前提交绑定的配置、inline execution 和最小权限输出 capability。
不要重用过期 capability，不把 capability/密钥写入仓库，不跳过代码提交/输入哈希检查。
在现有 request schema 上显式设置：

```json
{
  "native_child": "hud04_eval_child.py",
  "evaluation_role": "student",
  "evaluation_seed": 0,
  "source_completed_updates": 3686
}
```

其余原有必填字段（scope/source_plan/checkpoint/output_cap/code_commit 等）仍需由既有交付链生成。
这不是可单独提交的完整 task request。

首轮建议范围（等待用户批准，尚未提交）：

- 沿用既有 Flux IsaacSim 单 GPU 执行路径；1 个任务、1 个环境、1 个 high-step 案例；
- 先只运行 student，最多 1,000 策略步、worker 最多 900 秒；
- 训练更新 0，自动重试 0，不运行后续 teacher/其他技能任务；
- worker 上限包含准备/回传，但不保证平台计费/资源释放也受相同时限约束；
- 按结果再决定是否运行同案例 teacher 对照，后者需要单独批准。

## 判读

- `native_termination`：看 triggered_terms，不能直接叫作 fall。
- `native_timeout` / `reference_wrapped_not_success` / `horizon_reached_not_success`：均不是障碍成功。
- `engineering_invalid_*`：先修初始化、接口或记录链，不扩展到大批量评估。
- worker native exit 非零：保留原始退出失败，不能被有效诊断 JSON 覆盖成 pipeline 成功。

下一关才是增加几何任务目标与成功规则、配对 teacher 对照和多案例统计。

## 分项观测补丁：2026-10-30

首轮原生诊断 TASK_20261030_152 在初始化零动作预热时触发
`bad_tracking`，学生策略执行 0 步，随后原生退出仍发生段错误。
这是初始化无效结果，不是学生策略摔倒、障碍通关或收敛的证据。

新模块 `hud04_tracking_details.py` 在每次原生 BadTracking 调用期间
暂时包装其已有分项方法，记录原有方法返回值，每个原生分项仍只执行一次。
调用结束或异常后恢复方法；不刷新目标缓存、不推进参考帧、不改动作、阈值或返回值。
另外记录原生位置/朝向输入、双踝双腕目标与实际位置、位置误差及当前参考帧。
若额外输入读取失败，将保存 observation_error_type，不能视为诊断信息齐全。

CPU 检查发现 reset() 不更新 body_pos_relative_w，IsaacSim 的终止检查早于
command.step() 的目标缓存更新。合成数据能复现旧缓存导致位置分项终止的风险。
这不是首轮故障根因证明：首轮没有记录分项结果或缓存，仍需原生采样核对。

CPU 回归：

```bash
python -m unittest -v test_hud04_episode_probe test_hud04_single_episode_algo test_hud04_tracking_details
```

32 项通过，其中新增 8 项覆盖分项单次调用、原返回 tensor 对象不变、
方法恢复、异常传播、输入不变和状态 reset 转发。本补丁尚未经过原生 GPU 验证。
发布代码不等于批准新的收费任务；后续任务仍需明确的单次 GPU 范围授权。

## 有界修复试验：目标缓存刷新调度

TASK_20261030_174 原生观测确认：初始化的四个末端目标缓存全零，
锚点位置误差约 0.0339 m 且朝向分项未触发，末端位置分项触发。
学生执行 0 步，不能评价策略性能；退出时仍有独立的原生段错误。

`hud04_cache_fixed_command.py` 为单环境评估安装调度补丁：
在原生 `_check_termination()` 前更新当前帧位置和姿态目标缓存。
`hud04_tracking_cache_sync.py` 在远端从固定 SHA256 的原生 MotionCommand
源码提取其已有目标计算块，不复制私有源码到本仓库。
源码不匹配则拒绝运行，不回退到不同公式。
不调用 command.step，不推进参考帧、不执行采样器、不改物理状态、
不改动作、原生终止检查或阈值。拒绝多环境和训练模式。
补丁覆盖命令 setup 后的检查，包括算法构造期的 reset；
诊断 JSON 的 tracking_cache_sync 记录实际调用次数与源码哈希。

CPU 回归：原有 32 项 + 新增 10 项，共 42 项；另在本地私有原生代码上
验证非零世界坐标和 90 度 yaw 下的几何结果及输入不变。
这仍不是 GPU 验证成功的声明。单案例通过也不是障碍成功率或实机安全认证。
用户随后授予本问题所需有界 GPU 测试的持续授权；每次仍单任务、零训练、
900 秒 worker 上限、1,000 策略步上限、首次原生终止即停、零自动重试。

```bash
python -m unittest -v test_hud04_episode_probe test_hud04_single_episode_algo test_hud04_tracking_details test_hud04_tracking_cache_sync
```
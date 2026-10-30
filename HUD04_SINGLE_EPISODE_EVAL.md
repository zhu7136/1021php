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
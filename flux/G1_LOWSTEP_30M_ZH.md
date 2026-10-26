# G1 Low-speed step 教师：唯一一次 4096 环境、30 分钟段

- 授权：用户 2026-10-26T08:21:03Z 明确批准；1×4090D24GB、4096 环境；零自动重试、零自动续训；按平台实际费用计费，启动/克隆/释放可能另计。
- 输入：官方 `motions-low-step`，56 个 motion/terrain 对、113 个文件。逐文件 SHA256 校验；从头训练，不接入 HU_D04 或 locomotion checkpoint。
- 基础代码：已验证 Flux 集成 commit `0d7f5ad332b09363db249925391d6276284f7538`；Holosoma `4a0bf04cde5930af5ed59da797568d32be972525`。主分支不改动。
- 实际4096环境先完成20次PPO更新和原生checkpoint保存/重载，模型、两套优化器、normalizer、迭代及学习率逐字段一致才允许正式段。CPU fixture测试不代表GPU成功。
- 原生配方：24 steps/env、5 epochs、4 minibatches、actor/critic初始LR0.001；G1 29动作、action scale1.0，normalization关闭。只使用 low-step registry。
- 门禁通过后正式worker段上限1800秒；1680秒请求在完整更新边界停止，余下120秒用于保存与原生重载。正式更新最多20000；每500次保存，交付最终、验证及最近两个定期checkpoint。
- worker总计硬上限3300秒，包括准备、最多1080秒的启动/验证窗口、正式段和打包上传。启动前的资源/代码克隆不在worker计时内，不能声称整个平台费用仅30分钟。
- 数据通过W&B单文件签名GET交付；结果通过单对象签名PUT交付。账号密钥不传入worker、不写入公开Git。依赖重建沿用公开历史包；包内旧locomotion数据不会作为本任务registry或恢复源。
- Flux控制台实时日志 + W&B原生offline记录；提交控制run不是训练曲线。只有结果取回、哈希及原生状态验收后才能声称完成；不声称策略已收敛。

入口 `gm-run 1021php/flux/run_g1_lowstep4096.py` 还需要最终固定commit、实际task ID和限权交付参数；本文件不是通用启动payload，也不包含这些签名URL。
# task048 后新增1000次更新：在线监控准备稿

状态：仅本地修改与CPU测试完成。未发布Git、未创建Flux任务、未启动GPU；不代表worker在线链路已验证。

## 固定实验

- 输入：task048最终checkpoint，iteration3847。
- 输入SHA256：`265bb2eff169a08c890f63f70fd3b8782db23fed9ebcb44ebbfcb0fe42367718`。
- 来源ZIP SHA256：`bea9ea4f56c4cb1efbb999656c9a28fbdc9f2ce233a5b2dd3c5fb4fe21aacbc0`。
- 原生actor/critic/优化器/学习率恢复；4096环境、24steps/env、5epochs、4minibatches不变。
- 目标为新增1000次PPO更新，不是1000环境步。原生循环最终索引预计4846。
- 每新增500次更新保存：500→iteration4346，1000→iteration4846；正常收尾额外保存model_smoke_final.pt。
- 绝对索引4000/4500触发的原生周期保存被过滤，不把全局索引500间隔误称为新增500间隔。
- 提议单任务45分钟worker上限，自动重试0。实际启动仍需资源/预算授权与新调度器，不复用旧dispatcher。
- 不足1000次，即使成功保留部分模型，也不能验收为目标完成。

## 新入口

- `flux/run_continue1000.py`
- `flux/continue1000_bootstrap.py`
- `flux/continue1000_teacher.py`
- 配套`continue1000_contract.py`、`continue1000_input.py`、`continue1000_algo.py`、`live_monitor.py`。

## 监控与凭据边界

日志双写文件与控制台；每20秒输出心跳。心跳仅代表进程仍在，不证明PPO取得进展。
更新事件含x/1000、原生索引、平均rollout样本吞吐和已用训练时间。

只接受worker通过平台安全机制预先注入的WANDB_API_KEY；不把密钥放入Git、启动命令、参数、配置或普通上传文件。
如何在当前Flux账号安全注入尚未确认；没有注入则明确失败，不降级offline。
原生wandb.init必须返回online模式的新run，禁用续用旧run；保留ARIA来源，记录task048来源。

checkpoint保存后请求W&B即时上传，但上传请求返回不是服务端备份成功证明。
运行期间仍需从服务器核对文件存在、大小与下载哈希；通过前不能承诺平台强制终止后一定可恢复。
worker出网、认证权限、曲线接收、checkpoint上传和平台控制台展示均待真实环境验证。

## 测试

`python -m unittest discover -s flux -p 'test_*.py'`

本次160项CPU测试通过；含输入包哈希/提取、配置拒绝漂移、源码生成语法、相对保存周期、日志双写、心跳、跨块脱敏、超长行抑制、超时和失败清理。
这些测试不是GPU PPO、Flux安全注入或实时W&B链路通过的证据。
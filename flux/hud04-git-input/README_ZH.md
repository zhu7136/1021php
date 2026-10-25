 # HuD04 Git 输入交付：CPU 验证候选，不是教师训练结果

本目录采用普通 Git 文件交付输入。`input.zip` 为 17,305,073 字节，
SHA-256：`79df4443fe607a46d993f20896f189290e81271116310d30267e4b8b71fd3136`。
当前体积无需拆片；`manifest.json` 使用可兼容多片的格式记录此单片。
不是 Git LFS、Release 附件、签名下载链接或个人存储挂载。

## 内容与边界

- 39 个当前 HU_D04_01 使用的机器人资产文件、1 份动作 NPZ、12 个 HuD04 Python overlay 文件。
- `content-manifest.json` 记录包内全部 52 个文件的大小及 SHA-256。
- 资产来源：`https://github.com/zhu7136/beifen1001`，评估使用的固定提交
  `a01ad1f4a73b81acb94f606caf4cf250028efa86`。
- 原始 `HU_D04_description/package.xml` 保存在 `provenance/`；其 license 字段为 TODO。
  本次发布按仓库用户明确许可公开的范围执行，不声称已完成法律认证，
  不将第三方机器人资产自动重许可为本仓库代码许可证。
- 动作是 `CPU_SCHEMA_ONLY_31joint_NOT_PHYSICS_VALIDATED` 候选，
  通过接口检查不代表运动可行、稳定行走或教师训练质量通过。
- CPU OpenUSD 检查：原包与精简包所组合的场景一致；31 活动关节、42 刚体、
  27 碰撞体，无组合错误。`OmniPBR.mdl` 在 CPU 解析器下仍未解析，
  需由目标 IsaacSim 镜像确认，不能声称运行环境完全自包含。

## 校验与入口选择

`flux/hud04_git_input.py` 在无网络、无仿真条件下重组并验证输入。
先核验清单 SHA-256、每片大小与 SHA-256，再核验整包；拒绝符号链接、
LFS 指针、缺片、错序、重复片、不安全路径和已有输出覆盖。

控制参数新增：

```text
input_git_manifest = flux/hud04-git-input/manifest.json
input_git_manifest_sha256 = SHA-256(manifest.json 的原始字节)
input_bytes = 17305073
input_sha256 = 79df4443fe607a46d993f20896f189290e81271116310d30267e4b8b71fd3136
```

采用 Git 输入时必须完全省略 `input_url`；两种来源同时存在会报错。
Git 输入失败不会静默切换到 HTTPS。单片整包与原始候选压缩包逐字节一致。

此变更仅替换输入交付路径，不覆盖依赖安装、结果回传或所有网络请求。
后续仍有 8 个 wheel 的固定来源下载，结果仍使用受限对象存储上传。
没有把账户凭据、任务控制 payload 或签名 URL 放入仓库。

## 授权与运行限制

本分支是公开交付准备，**没有启动或授权任何新 GPU 任务**。
现有入口的第 1–4 次尝试限制保持不变，不能借用旧任务或旧控制参数重跑。
后续若执行，需要单独确认资源、次数、时限、预算和新任务身份。
Git clone 时间也可能计费；CPU 下载速度不代表 Flux worker 的速度。

## 2026-10-25 14:39 UTC：关闭修复候选

仅替换包内 overlay/flux/hud04_validate_child.py，其他 51 个成员字节不变。
使用固定版本 Holosoma 的 close_simulation_app 处理无窗口关闭挂起。
该上游函数跳过 USD close_stage、禁用 IsaacLab STOP 渲染回调，且不等待 Replicator；
这不等于验证了原生 stage 清理。只有关闭函数真实返回后才写完成标记，
仍要求子进程退出码为 0；提前 exit(0)、异常或超时均不能判成功。
本候选尚未经过 GPU 验证；该资料包本身不是 GPU 启动授权。
此前 task093 已验证原输入交付及短仿真，但正常退出失败。未启动 PPO，
未验证动作物理可行性、头部保持效果或教师 checkpoint。资产来源及许可限制不变。

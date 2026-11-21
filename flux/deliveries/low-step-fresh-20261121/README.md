# Hu-D04 低台阶：经用户授权的公开 Git 分片交付

本仓库只交付本次从零训练所需的离线输入。**这不是训练成功记录，也不是已通过 GPU 验证的训练入口。**

- 目标：low-step teacher，从零初始化，不加载旧 teacher/student checkpoint。
- 数据：56 个 clip、25,676 帧；来源记录 `HuD04_WBT/j5fpnbpy`。
- 完整运行时包：113,889,241 bytes；低台阶输入包：91,493,471 bytes。
- 总计 205,382,712 bytes，拆为 5 个普通 Git 二进制文件，每片最多 48 MiB；不使用 Git LFS。
- `inputs/shard-manifest.json` 固定每片顺序、长度、SHA-256 和两个整包的长度、SHA-256。
- 清单 SHA-256：`24e92ac169f998acd280b670d48b7bae7f1d0dcc914ac5a276574d72c4a91e54`。
- 本地 CPU 流式重组已回读并通过两个整包 SHA-256；未运行 CUDA、仿真或 PPO。
- 用户于 2026-11-21T12:21:12Z 明确允许将本次数据分片和机器人资产公开上传到 `zhu7136/1021php` 新分支；不改 main。
- 此授权仅覆盖本次输入；账号密钥、签名能力、私有业务日志不在发布范围。

## Flux 交付方式

1. 通过本次指定公开分支获取普通 Git 分片；不要把 token 写入 URL、代码、JSON、日志或普通命令行参数。
2. 固定本次发布 commit，而不只是可变分支名。由交付回执提供 commit。
3. Flux 在代码拉取阶段获取分片。不能把浏览器网页 ZIP 上传当作此路径的前置条件。
4. 使用匹配镜像的 Python，在 CPU 预处理阶段执行：

   ```bash
   python reassemble_inputs.py --destination /workspace/low-step-reconstructed
   ```

   目标目录必须不存在；程序拒绝覆盖已有数据、符号链接、未列明分片、不安全名称和哈希错误。
5. 解压前仍需检查 ZIP 路径/符号链接、内部 manifest、机器人资产和动作数据结构。

## 尚未完成的上线条件

- Flux 对本次分支的 clone **尚未验证**；GitHub 上传成功不等于 GPU 已收到数据。
- `runtime.zip` 包含历史 fixture 的源码/配置，不能直接当作本次训练入口。
- `low-step-input.zip` 内是低台阶 teacher 的原始训练代码；新入口仍需核实 fresh-init 断言、镜像 ABI、更新上限、退出与结果回传。
- 不包含旧权重或账号密钥；不修改其他 teacher/student。
- 未创建 Flux 草稿，未启动 GPU，没有训练更新。

手册修订见 `docs/FLUX_TRAINING_PLAYBOOK_ZH.md`。
# HU_D04 Flux 离线依赖包（准备版，未启动 GPU）

## 适用范围

- Linux x86_64、CPython 3.11（目标镜像 Python 3.11.13、glibc 2.35）。
- 镜像须已安装 Torch 2.7.0 及其 CUDA/系统依赖；本包不安装或替换 Torch/CUDA。
- 基于仓库 `zhu7136/1021php` 的分支 `aria/hud04-git-data-v8-20261023`，
  基础提交 `202852c72168e45d03d427b408f088379169b18f`。
- 这是增量覆盖包，不是完整训练仓库；不重复包含原有数据分片、机器人资产或 Holosoma 子模块。
- 不要只替换某一个启动文件；应同时应用本包全部文件，包含 repair-manifest.json。

## 包含内容

保留原 requirements.txt 的全部 51 个固定版本。显式增加两个依赖闭包补充：
`pillow==13.0.0`（TensorBoard 依赖）和 `markupsafe==3.0.4`（Werkzeug 依赖）。
共 53 个 wheel，逐文件 SHA-256 和大小位于 offline-deps/manifest.json。
antlr4-python3-runtime 4.9.3 的源码包来自 PyPI，经源包哈希校验后在 CPU 上
构建成通用 Python wheel；其源包哈希和构建产物哈希均已记录。

offline_dependencies.py 在实际安装前检查 Python/平台、现有 Torch 版本、
原始版本清单哈希、锁定清单哈希以及所有 wheel 文件的哈希和大小。
安装使用 `--no-index --no-deps --only-binary=:all: --require-hashes`，
只读取本地 wheel；文件缺失或损坏会停止，绝不回退至 PyPI。
git_smoke_v8.py 和 train_hud04.py 的依赖安装路径均已适配此方式。

## 已验证与未验证

- 53 个 wheel ZIP 完整性和版本元数据检查通过。
- 按目标 Python 3.11/Linux 标记检查，53 个软件包的直接依赖在锁定包集及
  镜像提供的 Torch 2.7.0 范围内均可满足（不声称重新验证镜像内 Torch 自身依赖）。
- pip 实际完成 CPython 3.11/Linux 交叉目标安装，53/53 安装源均为本地文件，
  且禁用索引、缓存，强制哈希校验。
- 全部 117 个 CPU 回归测试通过，包括新增的 10 个离线安装器测试。
- 当前准备沙箱只有 Python 3.13，因此没有运行 Python 3.11 扩展模块导入测试。
- 未在实际 Flux 镜像中验证导入、IsaacSim、CUDA、物理接触/碰撞或 PPO。
- CPU 制作阶段没有创建/启动新 Flux 任务。此次独立 Git 分支发布同样不包含 GPU 费用授权。
- 离线包 manifest.json 中的状态字段记录 CPU 制作时点，不是后续 Git 发布状态。

## 安全检查（不启动训练）

把覆盖包中的 `1021php/` 内容合并到上述基础仓库，保留原有数据和子模块；
在仓库根目录执行以下命令只做完整性检查，不安装或启动 GPU：

```bash
python flux/offline_dependencies.py --verify-only
```

本修订使用独立分支 `aria/hud04-offline-deps-v1-20261024`，
保留旧分支和旧任务不变；远端发布与克隆核验结果单独记录。
旧 Flux 任务仍使用旧分支，不会因本次 Git 发布而自动切换或重新启动。
实际远程运行需要另行确认一次性资源、时限和费用；现有 32 环境/2 PPO 迭代/
900 秒工作进程限制没有放宽，它不是包括平台启动和资源释放在内的账单硬上限。
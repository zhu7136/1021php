# Flux 教师训练输入：公开 Git 分片

这是经用户授权的训练准备材料，不是教师 checkpoint，也不是 GPU 启动授权。

## 包含内容与边界

- 官方 `motions-locomotion` 全量201个文件，100个动作/地形对，逐文件SHA-256不变。
- Python3.11/Linux x86_64 的70个公开增量wheel，包含其原始许可证/第三方告知文件。
- easydict/certifi原始源码包、ANTLR源码包、四个补充许可证及来源固定commit。
- 六个普通Git分片，单片不超过48MiB。没有Git LFS指针。
- `manifest.json`明确列出8个暂缓公开的native wheels及其完整文件名、版本、大小和哈希。

**这不是完整训练环境。** 暂缓公开的包为：grpcio、lxml、ml-dtypes、onnxruntime、opencv-python、orjson、pillow、shapely。其二进制第三方源码交付义务尚未全部核验，不能只依据顶层MIT/BSD/Apache标签就公开分发。原W&B准备包内仍有完整78个wheel；本次不改动它。

许可证清单是工程核查记录，不是法律认证；第三方条款不受仓库根Apache许可证替代。数据来自仓库官方 `training-motion-examples-v1` release，根LICENSE、NOTICE及数据README已随依赖归档保留。

## CPU核验与重组

在已递归克隆并固定版本的PHP仓库根目录执行：

```bash
# 默认只检查分片，无网络、无仿真、无训练。
python flux/prepare_inputs.py

# 在新目录拼接、安全解压，并核验每个展开文件。
python flux/prepare_inputs.py --output ./prepared-inputs
```

脚本拒绝覆盖已有输出目录，拒绝缺片、乱序、大小/哈希不匹配、LFS指针、符号链接和不安全归档路径。失败不删除输入，不自动联网补包，也不切换数据版本。

展开后数据位于 `prepared-inputs/motions-locomotion`；公开wheel位于 `prepared-inputs/php-offline/wheelhouse`。

## 完整离线安装：需要另外交付暂缓公开的8个wheel

先通过已批准的非公开方式，把manifest中8个匹配wheel放到一个目录。本目录不能提交到公开Git。

以下是准备命令模板，需替换解释器和私有wheel目录。解释器必须是镜像真正用于训练的Python3.11；输出目录必须尚不存在：

```bash
python flux/prepare_inputs.py --output ./prepared-installed \
  --install --python /absolute/path/to/training-python \
  --withheld-wheel-dir /absolute/path/to/privately-delivered-wheels
```

脚本会先核对全部暂缓公开wheel的大小/哈希，再以 `--no-index --no-deps --require-hashes` 安装完整78包到新目录 `.flux-deps`。未交付完整时明确失败，绝不从PyPI静默补包。

Torch/CUDA/IsaacSim/IsaacLab/Warp仍由镜像提供，不在分片内，不会被本脚本升级。训练时还必须设置 `.flux-deps`、PHP和固定版本Holosoma的导入路径；此脚本不自动调用训练入口。

## Flux接入与启动限制

代码以固定commit递归拉取；Holosoma保持 `4a0bf04cde5930af5ed59da797568d32be972525`。分支可变，真正执行前必须记录最终commit并在worker核验。

若Flux实际克隆目录为 `1021php`，gm-run入口路径应使用 `1021php/flux/bootstrap.py`，不是旧示例中的 `php_parkour`。仅入口路径正确不等于可启动：审批文件、数据位置、导入路径、运行时兼容、结果取回和预算仍须确认。

Git分片只改变传输路径，clone时间仍可能计费；本地重组通过不代表Flux worker克隆速度、CUDA、仿真或PPO已经通过验证。

没有创建Flux任务，没有启动GPU，`gpu_start_authorized`保持false。checkpoint下载/哈希/框架原生加载仍是后续验收项目。
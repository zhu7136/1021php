# 三教师student：输入交付，尚不可启动GPU

本次授权：新公开分支、不改main；单GPU、worker最多1200秒、96环境、最多10次student更新、自动重试0。允许本次三教师权重、三技能动作数据和所需机器人资产公开交付。不公开账号凭证、签名URL、原始业务日志或本次新student输出。

三教师：行走iteration1135、低台阶iteration2049（累计2050次更新）、高台阶iteration2046。所有权重SHA256精确匹配已评测文件。输入archive与Git blob逐字节哈希校验后复用；不含高台阶旧续训checkpoint。

216个训练动作片段、123231帧：行走100/71042、低台阶56/25676、高台阶60/26513，50Hz。它们是训练参考，不是held-out评测集。

必须修复的CPU阻塞：现有depth相机引用G1的torso_link/pelvis，HuD04 URDF不存在；三技能NPZ都没有现有student要求的vel_cmd。不能将全零指令、去掉depth、旧教师替代作为隐式修复。关节速度限位差异也必须在统一原生环境冻结回放。

该交付包含preflight.py输入校验器，不含可提交GPU训练入口。预检会验证三份teacher、原始reference字段及camera阻塞，并以非零状态拒绝启动。不创建Flux任务、不运行CUDA或优化器；不把输入交付成功当作训练开始。

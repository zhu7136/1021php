G1严格关闭单次对照；不是教师训练。继承已公开固定G1源码与已有公开输入，新增仅文本。
2环境/reset+8零动作/0PPO；单worker900秒（不含排队、镜像拉取、clone与释放计费）；close30秒；0自动重试。
真实USD close_stage、STOP-disable、fastShutdown=false，无额外Python profiler。退出0但缺少原生关闭返回标记也不通过。
使用r49已核验官方33wheel overlay；与历史G1正式训练的依赖集合及动作集并非完全相同，不据此单独判定根因。
使用locomotion公开动作包及PHP G1 terrain预处理；不是r276低台阶56动作的完全复现，不以短零动作结果判断任务学会。
输出只含诊断、阶段收据和受限日志；无checkpoint、账号凭据、模型或新动作发布。

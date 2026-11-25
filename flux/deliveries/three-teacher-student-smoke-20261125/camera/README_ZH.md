# Hu-D04 躯干深度相机适配（CPU已检查，尚未接入训练入口）

- 用户于2026-11-25T06:32:36Z选择躯干相机。父link固定为`waist_pitch_link`，不使用头部相机。
- 安装平移（m）：`(0.092, 0.0175, 0.2751)`；URDF安装rpy（rad）：`(0, 2.1818, 0)`。配置已转换为渲染器所需的度。
- 官方来源固定于`limxdynamics/humanoid-description@4c0a3c71200b810a52831c7c2b2de0ed6ce819a4`的`HU_D04_description/urdf/HU_D04_01.urdf`。相机joint在URDF中被注释。当前训练URDF的这段安装定义相同。
- 直接在现有躯干link上加传感器偏移，不启用注释joint，不改变URDF、USD、教师权重、关节限位或机器人动力学。
- 自遮挡映射使用当前训练URDF的32个有效visual mesh；包含base_link以及头部link对应的neck mesh，不沿用G1的pelvis/torso和网格。
- 保留原RsD435i成像参数；深度观测仍为58×87、延迟3–4帧。仅相机安装随机化关闭以便首轮确定性验收。

## 光学坐标边界

URDF只给CAD相机link安装位姿，没有光学frame/实机标定。此模块显式采用临时仿真约定：optical +Z = CAD +Z、optical +X = CAD -Y、optical +Y = CAD +X；右乘`Rz(-90deg)`，不使用G1的`[-90,0,-90]`。

该约定下，中心射线在躯干坐标系中为`[0.819073,0,-0.573690]`，即名义向前下方35.008度。这是几何计算，不是已验证的物理镜头外参。

`optical_frame_calibrated=false`，`native_render_verified=false`。必须在原生环境检查深度图方向、地面/台阶、人体遮挡和body-name查找；不能仅凭CPU针孔测试宣称相机已经可用于训练或部署。

## 安装与选择

在本目录运行：

```bash
python install_camera_overlay.py --bundle /path/to/extracted/bundle --check-only
python install_camera_overlay.py --bundle /path/to/extracted/bundle
```

`--bundle`为原输入包解压后的bundle目录。安装器严格校验既有URDF与32个mesh的SHA-256，仅添加相机模块和JSON契约，不修改既有preset或资产。

新相机类路径为：

`holosoma.sensors.warp.camera_config.hu_d04_torso_depth_config.HuD04TorsoRsD435iConfig`

构建Hu-D04 student观测时，对其ObservationManagerCfg调用模块中的`apply_torso_depth_config`，会返回仅替换depth_camera/depth_cam中camera_sensor_cfg的副本，保留尺寸、延迟和其他观测。此函数尚未接入可执行student入口，不能把安装成功当成已开始使用该相机训练。

## 验证和未完成项

`cpu-camera-report.json`记录21项CPU检查；包括真实基础配置类、项目实际四元数函数、32个mesh的文件与三角形、名义针孔射线及观测替换fixture。没有运行Warp renderer、PhysX或GPU。

三技能参考文件缺少vel_cmd的问题仍未解决；在补齐合法指令语义、相机原生验证和冻结教师原生回放通过前，仍禁止student更新。本变更新增GPU任务0、训练更新0；此前预算不增加。
# UAVNav-Airsim

**Deep reinforcement learning for UAV visual obstacle avoidance in confined tunnels — Cosys-AirSim + Unreal Engine.**

面向隧道受限空间的无人机视觉避障，以单目深度图序列和机体状态为输入，输出三维速度指令。仓库包含 MA-SAC、对照算法以及 AirSim 训练和评估入口。

## 亮点

- **MA-SAC**：CoordAtt 对深度特征做方向感知重标定 + Mamba 选择性状态空间模型对历史观测做选择性记忆，作为 Actor/Critic 的时序主干
- 支持多 AirSim 实例并行训练、检查点保存和冻结模型评估

## 方法概述

- **问题建模**：隧道内重复截面与有限视场导致部分可观测，建模为 POMDP；观测为 60×90 深度图序列 + 无人机状态，动作为连续三维速度指令（前向速度 / 垂直速度 / 偏航角速度）
- **策略网络**：CNN + CoordAtt 编码深度图 → Mamba 时序主干（Actor 与 Critic 共用结构）→ 连续动作分布
- **训练机制**：面向不等长轨迹的序列化 SAC——特征缓存、掩码损失、分阶段梯度更新
- **对比基线**：SAC(FrameStack4) / CoordAtt-SAC / LSTM-SAC / Transformer-SAC / Mamba-SAC

## 仓库结构

```
configs/      各算法与场景的训练配置
gym_env/      Gymnasium 环境封装、网络结构、回放缓冲、wrappers
scripts/      训练、评估和分析入口
tools/        共用配置、仿真进程管理和验证工具
tests/        无仿真自检
Logs/         训练与评估产物目录（见 Logs/README.md）
```

## 快速开始

```powershell
# Python 环境（pixi）
pixi install

# 查看训练和评估参数
pixi run python -m scripts.training.train_trainenv --help
pixi run python -m scripts.evaluation.evaluate_single --help

# 训练曲线
tensorboard --logdir Logs/TrainData/Runs

# 无仿真自检（不需要 AirSim / GPU）
pixi run python -m unittest discover -s tests
```

## 文档

- [TrainEnv 与其他脚本入口](scripts/README.md)
- [历史隧道实验复现协议](RETRAINING.md)
- [产物目录约定](Logs/README.md)

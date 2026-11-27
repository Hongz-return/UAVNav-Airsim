# UAVNav-Airsim

**Deep reinforcement learning for UAV visual obstacle avoidance in confined tunnels — Cosys-AirSim + Unreal Engine.**

面向隧道受限空间的无人机视觉避障，以单目深度图序列和机体状态为输入，输出三维速度指令。仓库包含 MA-SAC 方法、UE5.4 TrainEnv 训练入口及 UE4 隧道实验复现工具。

## 亮点

- **MA-SAC**：CoordAtt 对深度特征做方向感知重标定 + Mamba 选择性状态空间模型对历史观测做选择性记忆，作为 Actor/Critic 的时序主干
- 自建 **4 类 UE4 隧道场景**：常规 / 静态障碍 / 动态障碍 / 动静混合
- **6 算法 × 3 随机种子 × 100 回合**标准化评估；多 AirSim 实例 RPC 并行训练、checkpoint 自动筛选、评估 manifest 归档

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
tests/        无仿真自检
archive/      历史筛选、便利入口、打包和地图工具
Logs/         训练与评估产物目录（见 Logs/README.md）
```

## 快速开始

```powershell
# Python 环境（pixi）
pixi install

# 第四章 TrainEnv 基础模型（四实例、2× 仿真时钟、300000 条转移）
pixi run python -u -m scripts.training.train_trainenv

# UE4 隧道批量训练计划
pixi run python -m scripts.training.train_parallel --scene tunnel --dry-run

# 评估
pixi run python -m scripts.evaluation.evaluate_completed_runs --dry-run

# 训练曲线
tensorboard --logdir Logs/TrainData/Runs

# 无仿真自检（不需要 AirSim / GPU）
pixi run python -m unittest discover -s tests -t .
```

## 文档

- [TrainEnv 与其他脚本入口](scripts/README.md)
- [UE4 隧道实验复现协议](RETRAINING.md)
- [历史工具与复现命令](archive/README.md)
- [产物目录约定](Logs/README.md)

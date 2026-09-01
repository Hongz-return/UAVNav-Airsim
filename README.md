# UAV Navigation DRL for AirSim

本仓库用于在 Cosys-AirSim 隧道环境中训练和评估无人机深度强化学习策略。
当前维护的算法均基于 SAC，包括：

- `sac`：SAC + FrameStack4
- `coordatt`：CoordAtt-SAC
- `lstm`：LSTM-SAC
- `mamba`：Mamba-SAC
- `ma`：Mamba + CoordAtt SAC
- `transformer`：Transformer-SAC

## 文档入口

为避免把历史描述误当成当前状态，文档按以下职责划分：

1. [`RETRAINING.md`](RETRAINING.md)：固定实验口径、训练预算和一键命令。
2. [`Logs/README.md`](Logs/README.md)：训练、评估、筛选与归档数据的统一目录结构。
3. [`scripts/README.md`](scripts/README.md)：当前维护的脚本入口。
4. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：模块职责、观测/动作数据流与
   训练→评估→筛选的调用链。

若文档与磁盘产物冲突，以 `run.json`、模型文件、评估 manifest 和批处理
`results.json` 为准，不以旧聊天记录或 `plan.json` 作为完成证据。

## 当前实验环境

- Python 环境：仓库根目录的 Pixi 环境
- 仿真器：Cosys-AirSim，`Multirotor`
- 训练与评估入口：`scripts/training/`、`scripts/evaluation/`、`scripts/selection/`
- 配置：`configs/`
- 训练数据：`Logs/TrainData/`
- 评估数据：`Logs/EvaData/`
- 可视化：TensorBoard

当前代码不使用 SwanLab，也不应根据历史 SwanLab 说明判断实验状态。

## 常用命令

在 PowerShell 中从仓库根目录执行：

```powershell
pixi run python -m scripts.training.train_parallel --scene tunnel --dry-run
pixi run python -m scripts.evaluation.evaluate_completed_runs --dry-run
tensorboard --logdir Logs/TrainData/Runs
```

正式运行前请阅读 [`RETRAINING.md`](RETRAINING.md)。

## 无仿真自检

`tests/` 下的用例只依赖标准库，不需要 AirSim、GPU 或实验数据，覆盖实验路径
约定、并行训练计划与场景步数预算，以及各评估调度器与
`scripts.evaluation.evaluate_single` 的命令行契约：

```powershell
pixi run python -m unittest discover -s tests -t .
```

## 安装提示

项目依赖 Cosys-AirSim、Gymnasium、Stable-Baselines3、PyTorch、TensorBoard
等组件。推荐使用仓库中的 `pixi.toml` 和 `pixi.lock` 复现 Python 环境，
不要再参考旧版 GUI、fixed-wing、SwanLab 或仓库内嵌 Stable-Baselines3
的历史安装流程。

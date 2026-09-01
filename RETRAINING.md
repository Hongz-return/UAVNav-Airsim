# 当前实验复现说明

固定口径版本：2026-07-30

实验数据结构请参阅 [`Logs/README.md`](Logs/README.md)。本文件只记录实验协议和命令，不记录“正在运行”之类容易过时的信息。

## 实验矩阵

每个场景包含 `sac`、`coordatt`、`lstm`、`mamba`、`ma` 和
`transformer` 六种方法，每种方法使用 seed `0 1 2`。

| 场景参数 | 含义 | 每个 seed 的实际预算 |
|---|---|---:|
| `tunnel` | 常规隧道 | 80,000 steps |
| `tunnel_obs` | 静态障碍隧道 | 150,000 steps |
| `tunnel_activate` | 动态障碍隧道 | 200,000 steps |
| `tunnel_mixed` | 动静混合隧道 | 200,000 steps |

注意：早期文档曾把 `tunnel_obs` 写成 100,000 steps，但现有完整实验实际使用
150,000 steps。后续统计、论文和复现实验统一按 150,000 steps 描述。

## 固定训练口径

- 一个训练进程只连接一个 AirSim 实例。
- 不在单个模型内部启用向量化 AirSim 环境。
- 多任务通过不同 RPC 端口并行，默认最多三路。
- 六种方法统一使用起飞高度 `z=-1.5`（AirSim NED）。
- 保持 `cnn_update_mode=both`、`seq_len=16` 和
  `inference_seq_len=8`。
- 使用 TensorBoard 记录原始 episode return、20 回合移动平均等指标。
- 运行时配置只覆盖场景和 AirSim 连接参数，算法配置来自 `configs/`。

## 一键训练

先生成计划并检查：

```powershell
pixi run python -m scripts.training.train_parallel `
  --scene tunnel `
  --methods sac coordatt lstm mamba ma transformer `
  --seeds 0 1 2 `
  --max-parallel 3 `
  --dry-run
```

确认无误后移除 `--dry-run`。静态和动态场景分别把 `--scene` 改为
`tunnel_obs` 和 `tunnel_activate`。

默认端口为 `41451`、`41461`、`41471`。需要指定时添加：

```powershell
--ports 41451 41461 41471
```

新动静混合场景在本机只训练 `sac`、`coordatt` 和 `lstm`，三种方法均使用
seed `0 1 2`、每个任务 200,000 steps，并统一从新打包的
`D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe` 启动 `Tunnel_AO`。先检查计划：

```powershell
.\scripts\training\train_env4_local.ps1
```

确认计划后，一键启动本机的 9 个正式训练任务：

```powershell
.\scripts\training\train_env4_local.ps1 -Execute
```

该入口不会训练 `mamba`、`ma` 和 `transformer`；三者保留给另一台电脑，
合并结果时必须保持相同的场景、步数、seed 和算法配置。

不要为了补做单个实验重新启动完整矩阵，应同时限定 `--methods` 和
`--seeds`，并使用新的 run id。

## 一键评估

对筛选到的所有完整训练 run 执行 100 轮最佳模型评估：

```powershell
pixi run python -m scripts.evaluation.evaluate_completed_runs -n 100 --max-parallel 3
```

运行前检查计划：

```powershell
pixi run python -m scripts.evaluation.evaluate_completed_runs -n 100 --max-parallel 3 --dry-run
```

评估固定使用：

- `best_model.zip`
- 100 episodes
- `scenario_seed=2026`
- 初始航向范围 ±10°
- 同场景各方法相同的场景序列

每个模型的结果写入 `Logs/EvaData/Results/<scene>/<method>/<seed>/<run_id>/eval_*`；
批处理计划、日志和最终汇总写入 `Logs/EvaData/FormalBatches/<run_id>/`。

当前启动器不会根据已有 `eval_100_best_*` 自动跳过任务。全部评估已经完成
后不要无筛选地重跑；如果只是缺少批次汇总，使用：

```powershell
pixi run python -m scripts.evaluation.rebuild_summary `
  Logs/EvaData/FormalBatches/<run_id>
```

## 完成判定

训练完成不能只看 `plan.json`。至少需要同时存在：

1. `models/checkpoint_<target>_steps.zip`
2. `models/model_sb3.zip`
3. `models/best_model.zip`
4. 对应训练日志的正常完成标记

100 轮评估完成至少需要：

1. `eval_100_best_*` 目录
2. `summary_metrics.json`
3. `evaluation_manifest.json`
4. 所属批处理完成时，`results.json` 中该任务没有失败

`plan.json` 只表示任务曾被计划，不能证明训练或评估成功。

## 重复实验口径

同一场景、方法和 seed 可能存在不同 run id 的独立训练。它们不是新的随机
seed，不能在未说明的情况下直接合并为更大的 seed 样本。

当前动态场景 `ma` 有两套 seed `0/1/2`：

- `2026_07_28_07_47_57_517511`：论文主实验批次
- `2026_07_28_15_55_32_685674`：独立重复批次

默认论文主表使用前者；后者用于复现稳定性或补充分析，并应明确标为
independent repeat。

## TensorBoard

```powershell
tensorboard --logdir Logs/TrainData/Runs
```

建议查看 `episode/return_raw`、`episode/return_ma20`、
`episode/reward_per_step` 和 `episode/success_rate_ma50`。TensorBoard
界面的 `Smoothing` 建议设为 0，避免二次平滑。

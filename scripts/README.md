# 规范化脚本入口

脚本统一采用英文 `snake_case`、动作优先命名，并按职责分目录。Python
入口从仓库根目录使用 `python -m` 调用，避免脚本移动后依赖相对导入路径。

| 分类 | 入口 | 用途 |
|---|---|---|
| 训练 | `scripts.training.train_single` | 单模型、单 AirSim 实例训练 |
| 训练 | `scripts.training.train_parallel` | 多端口训练调度与冻结计划 |
| 训练 | `scripts/training/train_env4_local.ps1` | ENV4 本机方法子集的一键训练 |
| 训练 | `scripts/training/train_env4_remote.ps1` | ENV4 远端方法子集的一键训练 |
| 评估 | `scripts.evaluation.evaluate_single` | 单个冻结模型评估 |
| 评估 | `scripts.evaluation.evaluate_completed_runs` | 批量评估完整训练运行 |
| 评估 | `scripts.evaluation.rebuild_summary` | 从现有逐模型结果重建汇总 |
| 评估 | `scripts.evaluation.evaluate_history_ablation` | 历史截断消融评估 |
| 评估 | `scripts.evaluation.evaluate_history_lengths` | 历史长度扫描 |
| 评估 | `scripts.evaluation.record_parallel` | 分波次录制评估 |
| 筛选 | `scripts.selection.select_checkpoints` | 通用检查点筛选与冻结盲测计划 |
| 筛选 | `scripts.selection.select_and_evaluate_env4` | ENV4 筛选与评估入口 |
| 筛选 | `scripts.selection.select_env4_coordatt` | ENV4 CoordAtt 定向重筛 |
| 筛选 | `scripts.selection.recheck_targeted_checkpoints` | 指定场景/算法的检查点复核 |
| 分析 | `scripts.analysis.analyze_training_metrics` | TensorBoard 训练指标分析 |
| 打包 | `scripts/packaging/` | 检查点、候选与论文数据打包 |

## 常用命令

只检查训练和评估计划：

```powershell
pixi run python -m scripts.training.train_parallel --scene tunnel --dry-run
pixi run python -m scripts.evaluation.evaluate_completed_runs -n 100 --max-parallel 3 --dry-run
pixi run python -m scripts.selection.select_and_evaluate_env4
```

ENV4 本机训练需显式授权：

```powershell
.\scripts\training\train_env4_local.ps1          # 只检查计划
.\scripts\training\train_env4_local.ps1 -Execute # 正式启动训练
```

恢复冻结的 ENV4 筛选计划需显式授权：

```powershell
pixi run python -m scripts.selection.select_and_evaluate_env4 --execute --resume <run_id>
```

汇总某个场景的训练指标（只读 TensorBoard 事件文件）：

```powershell
pixi run python -m scripts.analysis.analyze_training_metrics --scene tunnel
```

只重建已有评估批次汇总：

```powershell
pixi run python -m scripts.evaluation.rebuild_summary `
  Logs/EvaData/FormalBatches/<run_id>
```

所有新训练输出写入 `Logs/TrainData`，所有新评估、筛选和诊断输出写入
`Logs/EvaData`。目录规则见 [`../Logs/README.md`](../Logs/README.md)，实验协议
见 [`../RETRAINING.md`](../RETRAINING.md)，各入口之间的调用关系见
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)。

`scripts/**/*.ps1`、`scripts.evaluation.record_parallel` 的窗口摆放，以及默认的
`D:\UEBuilds\...` 与 `C:\Users\...` 路径都是 Windows 专用；`--dry-run` 计划生成
不受此限制。

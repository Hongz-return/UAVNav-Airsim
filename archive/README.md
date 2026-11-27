# 历史实验工具

在仓库根目录执行以下历史入口。UE4 仿真包和论文数据使用当时的实验路径，也可以通过已有命令行参数指定。

| 路径 | 用途 |
|---|---|
| `scripts/training/train_env4_local.ps1`、`train_env4_remote.ps1` | 本机与远端 ENV4 方法子集 |
| `scripts/evaluation/evaluate_env4_local.ps1` | ENV4 本机评估 |
| `scripts/selection/` | 第三章检查点筛选、冻结测试与定向重筛 |
| `scripts/evaluation/evaluate_history_ablation.py`、`evaluate_history_lengths.py` | ENV2、ENV3 历史长度与截断实验 |
| `scripts/packaging/package_checkpoint_audit.ps1` | 复核包与论文数据快照 |
| `configs/checkpoint_selection_*.json` | 原筛选任务配置 |
| `tools/map_generation/` | 旧体素地图生成与查看工具，样例输入位于仓库的 `tools/map_generation/map.binvox` |

```powershell
.\archive\scripts\training\train_env4_local.ps1
pixi run python -m archive.scripts.selection.select_and_evaluate_env4
pixi run python -m archive.scripts.selection.select_and_evaluate_env4 --execute --resume <run_id>
pixi run python -m archive.tools.map_generation.map_generation --help
```

历史入口调用当前的共享训练、评估和运行时组件。批量评估、汇总和第四章 TrainEnv 入口见 [脚本说明](../scripts/README.md)。

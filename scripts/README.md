# 规范化脚本入口

脚本统一采用英文 `snake_case`、动作优先命名，并按职责分目录。Python
入口从仓库根目录使用 `python -m` 调用，避免脚本移动后依赖相对导入路径。

| 分类 | 入口 | 用途 |
|---|---|---|
| 训练 | `scripts.training.train_trainenv` | 启动本地 UE5.4 TrainEnv 并训练一个 MA-SAC |
| 训练 | `scripts.training.train_single` | 连接已启动的 AirSim 实例训练单模型，默认单端口 |
| 训练 | `scripts.training.train_parallel` | 多端口训练调度与冻结计划 |
| 评估 | `scripts.evaluation.evaluate_single` | 单个冻结模型评估 |
| 评估 | `scripts.evaluation.evaluate_completed_runs` | 批量评估完整训练运行 |
| 评估 | `scripts.evaluation.rebuild_summary` | 从现有逐模型结果重建汇总 |
| 评估 | `scripts.evaluation.record_parallel` | 分波次录制评估 |
| 分析 | `scripts.analysis.analyze_training_metrics` | TensorBoard 训练指标分析 |

## UE5.4 TrainEnv

```powershell
pixi run python -u -m scripts.training.train_trainenv --instances 4 --clock-speed 2.0 --seed 0
```

默认读取 `configs/ch4_train_trainenv.ini`，共采集 300000 条环境转移，四实例每批采集 4 条。预算必须能整除实例数。
入口从 `D:\UEBuilds\TrainEnv\Windows\TrainEnvAirSim.exe` 定位打包运行程序，以
`--first-port`（默认 41771）起分配独立端口。RPC 确认四个实例都加载本次配置后才训练，退出时回收这批 UE 进程。

每个实例使用独立的临时 settings，开启 `NoDisplay` 和 60 FPS 上限，只请求 320×180 的
`DepthPlanar`。相机的深度捕获参数和 90×60 策略输入由原配置提供。可用
`--clock-speed 1.0` 按原速运行。 `--instances 1` 使用同步采样。

异步更新只采样已入库的回放数据。第一次达到回放就绪条件的批次会在入库后更新。
`--no-overlap-updates` 在每批入库后再执行更新，
`--no-fast-history` 使用按有效历史长度分组的原时序计算。

运行目录下的 `launch.json` 保存端口、进程号和实际 UE 参数，`config/config.ini` 留存
训练配置。进度行给出累计环境转移和梯度更新数，TensorBoard 事件和模型分别在
`tb_logs/` 与 `models/`。可用 `--run-dir` 指定新目录。

## 常用命令

只检查训练和评估计划：

```powershell
pixi run python -m scripts.training.train_parallel --scene tunnel --dry-run
pixi run python -m scripts.evaluation.evaluate_completed_runs -n 100 --max-parallel 3 --dry-run
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
`Logs/EvaData`。目录规则见 [产物说明](../Logs/README.md)，原隧道批次沿用
[复现协议](../RETRAINING.md)。历史 ENV4 筛选、评测与打包入口见 [归档](../archive/README.md)。

`scripts/**/*.ps1`、`scripts.evaluation.record_parallel` 的窗口摆放，以及默认的
`D:\UEBuilds\...` 与 `C:\Users\...` 路径都是 Windows 专用；`--dry-run` 计划生成
不受此限制。

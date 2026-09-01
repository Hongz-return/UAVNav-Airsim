# 实验数据目录

`Logs` 保存训练数据、评估数据和实验索引。

## 目录结构

```text
Logs/
├── TrainData/
│   ├── Runs/                 # scene/method/seed/run_id
│   └── RunMetadata/          # 并行训练计划与配置快照
├── EvaData/
│   ├── Results/              # scene/method/seed/run_id/evaluation_id
│   ├── FormalBatches/
│   │   ├── ENV1_Tunnel_and_ENV3_TunnelActivate/
│   │   └── ENV2_TunnelObs_and_ENV3_TunnelActivate/
│   ├── CheckpointSelection/
│   │   ├── ENV1_Tunnel/
│   │   ├── ENV2_TunnelObs/
│   │   ├── ENV3_TunnelActivate/
│   │   ├── ENV4_TunnelMixed/
│   │   └── CrossScene/       # 同一次筛选覆盖多个场景
│   ├── DemoRecordings/       # 演示录像索引和记录
│   └── Diagnostics/          # 消融与诊断实验
├── experiment_inventory.json
└── experiment_inventory.csv
```

训练记录的固定粒度是 `scene/method/seed/run_id`，评估记录在其后增加 `evaluation_id`。补跑实验位于 `Runs/_repeat` 和 `Results/_repeat`，不会被误算为新的正式种子。

脚本生成的压缩包写到 `outputs/packages`。

## 场景名称

| 实验编号 | 目录名称 | 代码场景名 |
| --- | --- | --- |
| ENV1 | `ENV1_Tunnel` | `tunnel` |
| ENV2 | `ENV2_TunnelObs` | `tunnel_obs` |
| ENV3 | `ENV3_TunnelActivate` | `tunnel_activate` |
| ENV4 | `ENV4_TunnelMixed` | `tunnel_mixed` |

## 当前快照（2026-08-14）

- 训练清单：80 个带 `run.json` 的训练运行，另有 1 个无清单的历史消融模型目录。
- 模型文件：1,378 个 ZIP，其中周期检查点 1,224 个、`best_model.zip` 78 个、最终 `model_sb3.zip` 76 个。
- 评估清单：902 个 `evaluation_manifest.json`。
- 核心覆盖：4 个场景 × 6 个算法 × 3 个种子，共 72 个组合；训练与“至少一份评估”均为 72/72。
- 统一论文检查点目录 `reports/paper_checkpoint_catalog` 验证了 72 个 `4 场景 × 6 算法 × 3 种子` 记录和 24 个场景/算法汇总，并逐个核对了模型 SHA256。
- ENV1–ENV3 均为每 seed 100 回合；ENV4 (`tunnel_mixed`) 覆盖全部 18 个算法/种子组合，但跨算法协议不统一：SAC 与 CoordAtt 为 100 回合，LSTM、Mamba、MA 与 Transformer 为 60 回合。目录通过 `episodes`、`protocol_status` 和成功条件指标 seed 数如实保留该差异。

准确的逐场景/算法计数与例外路径以每次重建的 `experiment_inventory.json` 为准。

## 已知例外

核心覆盖没有缺失，但保留了 5 个中断或已被后续运行替代的 ENV4 运行：这 5 个均没有最终模型，其中 2 个也没有最佳模型。它们只用于训练过程追溯，不承担当前核心覆盖或论文检查点来源。

`tunnel_mixed/coordatt_ablation_noattn/seed_1` 是消融产物，含配置与最终模型，没有 `run.json`。

## 重建与验证

```powershell
pixi run python tools/build_experiment_inventory.py
pixi run python tools/validate_experiment_metadata.py
pixi run python tools/build_paper_checkpoint_catalog.py --verify-only
```

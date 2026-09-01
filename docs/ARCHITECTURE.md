# 代码结构与数据流

本文件描述**当前仓库代码**的模块职责和运行流程。实验协议见
[`../RETRAINING.md`](../RETRAINING.md)，磁盘目录约定见
[`../Logs/README.md`](../Logs/README.md)，脚本清单见
[`../scripts/README.md`](../scripts/README.md)。

## 1. 分层总览

```text
scripts/            调度层：训练 / 评估 / 筛选 / 分析 / 打包的命令行入口
  └─ 通过 subprocess 调用其他 scripts.* 模块，自身不加载模型
tools/              共享基础设施：运行时装配、AirSim 进程管理、路径约定、日志、可视化
gym_env/            算法与环境库：Gymnasium 环境、感知、任务、序列网络、回放缓冲
configs/            六种方法的 .ini 算法配置 + 检查点筛选 JSON
Logs/               训练与评估产物（内容不入 Git，只保留目录说明）
```

依赖方向固定为 `scripts → tools → gym_env`。`gym_env` 只有一处反向依赖：
`gym_env/algorithms/mamba_sac.py` 引用 `tools.training_logging` 的 TensorBoard
记录函数。

## 2. 环境组装（gym_env）

`gym_env/__init__.py` 向 Gymnasium 注册 `airsim-env-v0`，实现在
`gym_env/envs/airsim_env.py` 的 `AirsimGymEnv`。它本身不读配置文件，而是由
`set_config(cfg)` 按 `.ini` 装配三个可替换部件：

| 部件 | 配置键 | 实现 |
|---|---|---|
| 动力学 | `[options] dynamic_name`（只支持 `Multirotor`） | `envs/dynamics/multirotor_airsim.py` |
| 感知 | `[options] perception` | `envs/perception/`（`depth_state`、`vector`） |
| 任务 | `[options] env_name` | `envs/tasks/`（`Tunnel`/`Forest`/`Trees` → `TunnelTask`；`Obs`/`Tunnel_Obs` → `ObsTask`） |

一个 step 的数据流：

```text
policy action (v_forward, v_z, yaw_rate)
  → MultirotorDynamicsAirsim.set_action  → AirSim moveByVelocityAsync(dt)
  → perception.get_observation           → DepthPlanar 深度图 (60×90) + 6 维归一化状态
  → task.is_done / compute_reward        → terminated / reward
  → AirsimGymEnv.step                    → (obs, reward, terminated, truncated, info)
```

要点：

- 坐标系：AirSim 用 NED，环境内部几何一律换算成 ENU（z 向上），
  `get_position()` 返回的是 ENU。
- 观测：`DepthStatePerception` 返回 `Dict{image: (H,W,1) uint8, state: (6,) float32}`。
  `min_distance_to_obstacles` 取有效深度的 1% 分位数，而不是最小值，用于避开噪声尖峰。
- 终止 vs 截断：碰撞、到达、越界是 `terminated`；`max_episode_steps` 是
  `truncated`，超时惩罚在截断分支单独叠加，让 critic 仍能自举。
- 评估复现：`set_evaluation_scenarios()` 装入一条有序场景列表，`reset()`
  逐条消费，因此同场景各方法看到完全相同的起点与初始航向序列。

## 3. 策略与回放（gym_env）

`tools/runtime_support.py` 的 `create_model()` 是唯一的模型装配点，按配置分成两条路径：

**A. 标准 SB3 路径**（`use_mamba = False`，即 `sac` 与 `coordatt`）

`DictFrameStack(n_stack=4)` → `Monitor` → `DummyVecEnv` → `VecTransposeImage`，
策略为 SB3 `MultiInputPolicy`，特征提取器是
`networks/stacked_dict_extractor.py` 的 `FrameStackedDictExtractor`：把
n_stack 帧沿通道维拼接后过普通 CNN，无循环结构。`coordatt` 与 `sac` 的唯一区别是
`[vision] cnn_attention = coordatt`，启用 `CoordAttention2d`。

**B. 序列 SAC 路径**（`use_mamba = True`，即 `lstm`/`mamba`/`ma`/`transformer`）

`gym_env/algorithms/mamba_sac.py` 的 `MambaSAC` 直接继承 SB3 `BaseAlgorithm`，
配合 `replay/sequence_buffer.py` 的 `SequenceReplayBuffer`：按整回合存储，采样
长度 `seq_len` 的连续片段并带 mask 计算 TD 目标。策略在
`networks/mamba_policy.py`，结构为 **共享 CNN → Actor/Critic 各自独立的序列块 → MLP 头**。
序列块由 `[mamba] mamba_backend` 选择：

| 方法 | `mamba_backend` | `cnn_attention` |
|---|---|---|
| `lstm` | `lstm` | `none` |
| `mamba` | `auto` | `none` |
| `ma` | `auto` | `coordatt` |
| `transformer` | `transformer` | `none` |

训练用 `seq_len = 16`，推理只保留 `inference_seq_len = 8` 步历史；
`--inference-seq-len` / `--no-history` 可在评估时截断历史做消融。

注意两条路径读取 CNN 参数的位置不同：标准路径读 `[vision]`，序列路径读
`[mamba]`（见 `gym_env/utils.py` 的 `build_mamba_policy_kwargs`）。

`replay/` 另外提供可选的 PER（`[DRL] use_per`）：`prioritized_replay_buffer.py`
与 `sum_tree.py` 实现缓冲，`per_algorithms.py` 提供 `SAC_PER` / `TD3_PER`。
当前六种方法的配置都是 `use_per = False`。

## 4. 训练流程

```text
scripts.training.train_parallel            # 计划 + 多实例调度
  ├─ SCENES / METHOD_CONFIGS               # 场景 → 地图/步数；方法 → configs/*.ini
  ├─ 写运行时 .ini（覆盖 env_name 与 AirSim ip/port，不改算法超参）
  ├─ 写 plan.json（含 git 状态与全部源码 SHA256）
  ├─ tools.airsim_instances.launch_instances   # 每个端口一个打包 UE 实例
  └─ 每个作业 subprocess → scripts.training.train_single
                             ├─ tools.runtime_support.build_runtime_env / create_model
                             ├─ tools.training_logging 回调（TensorBoard、best_model）
                             └─ models/{checkpoint_*_steps,best_model,model_sb3}.zip
```

调度是**方法优先**（method-major）：作业顺序为「方法 A 的三个 seed，方法 B 的
三个 seed……」，配合 `--max-parallel 3` 使每一波正好是同一算法的三个 seed；
只跑单 seed 时同一波会并行不同算法。一个训练进程只连一个 AirSim 实例，
默认端口 `41451/41461/41471`。

产物落在 `--logdir` 下，`train_parallel` 传入的是
`Logs/TrainData/Runs/<batch_run_id>/<scene>/<method>/seed_<n>/`，
`train_single` 再追加一层 `<timestamp>_<dynamic>_<policy>_<algo>/`。
`tools/experiment_paths.normalize_training_relative()` 在下游读取时去掉最外层
批次目录，把两种历史布局都归一到 `scene/method/seed/run_id`。

## 5. 评估流程

```text
scripts.evaluation.evaluate_completed_runs      # 发现 + 批量调度
  ├─ discover_completed_runs()                  # 要求 checkpoint_<target>、model_sb3、best_model 三件齐全
  ├─ 按场景分组复用 AirSim 实例，逐波并行
  ├─ subprocess → scripts.evaluation.evaluate_single --best --save ...
  └─ rebuild_summary.reconstruct()              # results.json + summary_all_models.{json,csv}
```

`evaluate_single` 是唯一真正加载模型跑回合的脚本，其余评估入口都是它的调度器：

| 入口 | 用途 |
|---|---|
| `scripts.evaluation.evaluate_completed_runs` | 全量 best-model 批量评估 |
| `scripts.evaluation.record_parallel` | 分波次启动带窗口的 UE 供录屏（仅 Windows） |
| `scripts.evaluation.evaluate_history_ablation` | 历史截断（Trunc）消融 |
| `scripts.evaluation.evaluate_history_lengths` | 历史长度 L 扫描 |
| `scripts.selection.select_checkpoints` | 两阶段检查点筛选 |
| `scripts.evaluation.rebuild_summary` | 只从已有产物重建批次汇总，不跑仿真 |

因为这些调度器通过 `subprocess` 拼命令行，它们能用的选项必须与
`evaluate_single` 声明的 argparse 选项一致；`tests/test_evaluation_cli_contract.py`
对此做静态校验。

单次评估写出：`summary_metrics.json/.csv`、`evaluation_manifest.json`、
`scenario_manifest.json`、`results.npy`、`trajectories.npy`、`step_data.npy`、
`trajectories_csv/` 与 `terminal_events.csv` 及四类终止点 CSV。

## 6. 检查点筛选流程

`scripts.selection.select_checkpoints` 负责论文用的检查点选择，协议是
**筛选（screen）→ 冻结 → 盲测（test）** 两阶段，正式测试结果永远不回流影响选择：

1. 从训练期 TensorBoard 标量（`episode/return_ma20`、`episode/success_rate_ma50`）
   构造候选集，而不是从评估结果构造；
2. 每个候选跑短程 screen 批次；
3. 只用 screen 结果为每个 method/seed 冻结一个赢家；
4. 冻结后的赢家跑正式盲测批次。

按场景分两种协议：`tunnel`/`tunnel_obs` 为 deterministic（screen 10 回合 ×1
seed，test 100 回合 ×1 seed），`tunnel_activate`/`tunnel_mixed` 为 dynamic
（screen 10 回合 ×2 seed，test 50 回合 ×2 seed，合计 100 回合）。
`select_and_evaluate_env4` / `select_env4_coordatt` / `recheck_targeted_checkpoints`
是它的预置封装，默认只出计划，需要 `--execute` 才会启动 UE 与评估进程。

## 7. 共享基础设施（tools）

| 模块 | 职责 |
|---|---|
| `experiment_paths.py` | 所有 `Logs/` 路径常量与训练→评估的目录映射 |
| `runtime_support.py` | 环境包装、模型创建/加载、检查点回调、回放缓冲存取 |
| `training_logging.py` | TensorBoard 指标写入、`UAVLoggingCallback`、`BestModelCallback` |
| `airsim_instances.py` | 打包 UE 实例的 settings 生成、启动、端口检查、P-core 绑定 |
| `build_experiment_inventory.py` | 重建 `Logs/experiment_inventory.{json,csv}` |
| `validate_experiment_metadata.py` | 校验实验 JSON/CSV 结构与引用路径是否存在 |
| `build_paper_checkpoint_catalog.py` | 论文检查点目录构建与 SHA256 校验 |
| `visualize_*.py` / `plot_*.py` | 注意力、中间特征、轨迹与失败分析可视化 |

## 8. 平台约束

- `pixi.toml` 的 `platforms` 目前只有 `win-64`，依赖含 CUDA 12。
- `scripts/**/*.ps1`、`record_parallel.py` 的窗口摆放、`airsim_instances.auto_pin_p_core`
  以及默认的 `D:\UEBuilds\...`、`C:\Users\...` 路径都是 Windows 专用。
- 不依赖仿真的部分（计划生成、路径约定、元数据校验、`tests/`）在任何平台都能跑。

## 9. 无仿真校验

```bash
python -m unittest discover -s tests -t .
python -m scripts.training.train_parallel --scene tunnel --dry-run
python -m scripts.evaluation.evaluate_completed_runs --dry-run
python tools/validate_experiment_metadata.py
```

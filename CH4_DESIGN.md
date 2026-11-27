# 第四章设计：运动感知与动作条件深度预测的 MA-SAC 扩展

> 目标：在第三章 MA-SAC（CoordAtt + Mamba 序列 SAC）基础上，提升动态障碍（ENV3，基线成功率 59.3%）
> 与混合障碍（ENV4，基线 50.0%）场景的避障成功率。当前实验统一修正基础实现，
> `ma` 对照组和第四章方法使用相同环境及基础代码；已发表的历史结果不与本轮直接比较。

## 0. 总体思路

| 模块 | 名称 | 作用 | 开关 |
|---|---|---|---|
| A | 帧差分运动通道（Motion Channel） | 像素级显式运动线索，避免被逐帧 CNN 压缩丢失 | `perception = depth_motion` |
| B | 动作条件下一帧深度预测（Aux Prediction） | 自监督辅助任务：由 Actor 上下文 + 当前动作预测下一帧深度，学习有助于动态避障的时序表征 | `[aux] aux_weight > 0` |

帧差同时包含无人机自身运动和障碍物运动；动作条件预测也不保证自动分离这两种来源。
上述模块是待验证的方法假设，是否提高避障成功率须由闭环实验判断。

新增 3 个方法，与原 `ma` 构成 2×2 消融：

| 方法键 | 配置文件 | A | B |
|---|---|---|---|
| `ma`（已有，不改） | `configs/config_mamba_coordatt.ini` | ✗ | ✗ |
| `ma_motion` | `configs/config_ma_motion.ini` | ✓ | ✗ |
| `ma_aux` | `configs/config_ma_aux.ini` | ✗ | ✓ |
| `ma_motion_aux` | `configs/config_ma_motion_aux.ini` | ✓ | ✓ |

三个新配置均从 `config_mamba_coordatt.ini` 复制，只改动下文列出的键（及 `project_name`）。

## 1. 模块 A：`DepthMotionPerception`

文件：`gym_env/envs/perception/depth_motion_perception.py`（新建），注册键 `depth_motion`。

- 继承 `DepthStatePerception`。
- 观测：`Dict{image: (H, W, 2) uint8, state: (6,) float32}`。
  - 通道 0：与 `DepthStatePerception` **逐像素完全一致**的深度 uint8。
  - 通道 1：运动编码。设 `d_t` 为清洗后（NaN/Inf→max_depth，clip 到 `[0, max_depth]`）并 resize 后的
    浮点深度（米），则
    `diff = d_t - d_{t-1}`，`enc = round(clip(diff / motion_diff_max, -1, 1) * 127 + 128)`，
    取值 `[1, 255]`，零差值 = 128。
  - 回合第一帧（reset 后）：`diff = 0`，通道 1 全为 128。
- 配置：`[perception] motion_diff_max`，fallback `1.0`（米）。注意 `[perception]` 段可能不存在，须用 fallback 读取。
- `min_distance_to_obstacles` 的计算方式不变。
- 重置钩子：
  - `BasePerception` 增加空实现 `reset(self) -> None`。
  - `DepthMotionPerception.reset()` 清空上一帧缓存。
  - `AirsimGymEnv.reset()` 在采集 reset 观测**之前**调用 `self.perception.reset()`（含评估场景路径）。
- 不改变 `depth_state` 感知的任何输出（可抽取共享的清洗辅助函数，但输出必须逐位一致）。
  当前实现的深度通道先 resize 再清洗；新方法通道 0 沿用此顺序，运动通道则按上式先清洗再 resize。

下游适配：`MambaSACPolicy._build()` 从观测空间读取 `n_channels`，`SequenceCNN` 第一层为
`n_channels * n_stack`，理论上自动适配。实现时须核查序列路径（`use_mamba=True`）的包装链
（是否经 `VecTransposeImage`、回放池存储布局、Actor 历史缓存 `_init_history_buffer` 等）对 2 通道是否正确，
以及评估入口 `scripts/evaluation/evaluate_single.py` 是否有单通道假设。

## 2. 模块 B：动作条件下一帧深度预测

### 2.1 配置（新段 `[aux]`，全部带 fallback，旧配置零影响）

| 键 | 默认 | 说明 |
|---|---|---|
| `aux_weight` | `0.0` | `0` 时**完全不构建**预测头、不计算损失，策略 `state_dict` 与第三章完全一致 |
| `aux_target` | `next_depth` | `next_depth`：预测下一帧深度；`depth_delta`：预测下一帧与当前帧深度之差 |
| `aux_out_h` / `aux_out_w` | `15` / `22` | 预测分辨率（深度图自适应平均池化后） |
| `aux_hidden` | `256` | MLP 隐层宽度 |

`ma_aux`、`ma_motion_aux` 使用 `aux_weight = 0.1`，`aux_target = next_depth`。
通过 `gym_env/utils.py::build_mamba_policy_kwargs` 传入策略；须核查训练、续训、评估、加载路径都能拿到这些参数。

### 2.2 网络

- `NextDepthPredictor(context_dim=mamba_hidden_dim, action_dim, out_h, out_w, hidden)`：
  `Linear(ctx+act, hidden) → ReLU → Linear(hidden, hidden) → ReLU → Linear(hidden, out_h*out_w)`，
  `next_depth` 用 Sigmoid 输出（目标 ∈ [0,1]），`depth_delta` 用 Tanh 输出（目标 ∈ [-1,1]）。
- 作为 `MambaSACPolicy.aux_predictor` 持有，参数加入 **Actor 优化器**。
  预测头在原 SAC 网络全部初始化之后创建，避免其随机数消耗改变同种子的 Actor/Critic 初始权重。
- 输入上下文取 **Actor** 的 Mamba 输出（经 `context_norm` 之后、MLP 头之前）。理由：Actor 是部署时使用的分支。
  `cnn_update_mode = both`（MA-SAC 默认）且辅助任务开启时，辅助梯度训练 Actor 时序主干与共享 CNN（见 §4）；
  SAC 的 Actor 损失仍不更新 CNN，与第三章保持一致。
- 为避免 Mamba 前向两次：重构 `MambaActor`，新增返回上下文的接口（如
  `action_log_prob_sequence_from_features(..., return_context=True)`），原接口输出须逐位不变。

### 2.3 损失（在 `MambaSAC.train()` 的 Actor 阶段计算）

- 动作输入：回放池中的 `actions`（物理量）经 `actor.unscale_action` 映射到 `[-1,1]`，`detach`。
- 目标：取 `next_obs['image']` 的**深度通道（通道 0）**，`/255` 后 `adaptive_avg_pool2d` 到 `(out_h, out_w)`；
  `depth_delta` 时目标为 `pool(next) - pool(curr)`。须按实际张量布局（`(B,T,C,H,W)` 或其他）取通道。目标 `detach`。
- 掩码：`aux_mask = mask * (1 - dones)`。终止步的 `next_obs` 是回退的当前帧（见
  `sequence_buffer.py::get_sequence`），不是真实下一帧；填充步的 `dones = 1`，因此一并排除。
- `aux_loss = Σ(aux_mask · MSE_per_step) / clamp(Σ aux_mask, min=1)`，`MSE_per_step` 为像素均值。
- `actor_loss_total = actor_loss + aux_weight · aux_loss`，用于更新 Actor 时序主干、动作头和预测头；
  CNN 在此阶段只接收 `aux_weight · aux_loss` 的梯度。辅助模式的裁剪参数列表加入 aux 参数，并去除重复 CNN 引用。
- `cnn_update_mode = critic_only` 时，Actor 阶段 CNN 被冻结，辅助任务只塑造 Actor 主干：允许，但打印一次警告。
- TensorBoard 记录 `train/aux_loss`。
- 熵系数更新、目标网络 Polyak 更新等其余流程不变。

## 3. 接线与流水线

- `gym_env/envs/perception/__init__.py`：注册 `depth_motion`。
- `scripts/training/train_parallel.py::METHOD_CONFIGS`：加入 `ma_motion`、`ma_aux`、`ma_motion_aux`。
- 全仓检索其余硬编码方法列表（评估、汇总、筛选、实验目录、测试等），在**训练 → 评估 → 汇总**链路上必要处加入新方法；
  论文检查点目录类工具（`build_paper_checkpoint_catalog.py` 等）不必改。
- 可视化脚本（`tools/visualize_*.py`）不必改，但若明显假设单通道，在报告中列出。

## 4. 现有 CNN 更新行为（保留第三章路径）

实现核对（2026-11-15，PyTorch 2.12.1 / SB3 2.8.0）发现原设计对此处的描述与源码不符：
`MambaActor.set_cnn()` 和 `MambaCritic.set_cnn()` 将同一个 CNN 注册为 `_cnn` 子模块，
所以在 `cnn_update_mode = both` 下，**Actor 和 Critic 优化器都含 CNN 参数**。
Critic 阶段会更新 CNN；Actor 阶段冻结 `critic.parameters()` 时连带冻结共享 CNN，
因此旧路径中 **CNN 实际由 Critic 阶段更新**，而非原设计所述的仅由 Actor 更新。

为保证已有六种方法逐位兼容，`aux_weight = 0` 时完全保留原参数列表、冻结顺序和更新路径，不修正该行为。
只有 `aux_weight > 0` 且 `cnn_update_mode = both` 时，在冻结 Critic 后恢复共享 CNN 的梯度。
CNN 特征在送入 Actor/Critic 前 detach 成独立求导输入；先计算辅助损失对该输入的梯度，
再把这部分梯度传回 CNN。Actor 主干与预测头仍按总损失更新，Critic 的其余参数仍冻结。
因此无需第二次 Mamba 前向，也不会因开启辅助任务而额外开放 SAC Actor 损失到 CNN 的通路。
`critic_only` 仍在 Actor 阶段冻结 CNN，并在构建辅助头时打印一次警告。

辅助模式下，`actor.parameters()` 已含共享 CNN，梯度裁剪须按参数身份去重，避免同一梯度被重复裁剪。
关闭辅助任务时保持第三章原有参数列表与计算顺序。

## 5. 验证（无仿真，CPU 可跑）

1. `tests/test_depth_motion_perception.py`：用假 cfg / 假 client（或 monkeypatch `_get_depth`）验证
   形状、首帧通道 1 = 128、已知差值的编码、`reset()` 行为、通道 0 与 `DepthStatePerception` 逐位一致。
2. `tests/test_ch4_policy_smoke.py`：对 `ma`、`ma_motion`、`ma_aux`、`ma_motion_aux` 四个配置，用无 AirSim 的
   假 Gym 环境（相同观测/动作空间）构建 `MambaSAC`，向 `SequenceReplayBuffer` 写入含终止与截断的随机回合，
   执行 `train(gradient_steps=2, batch_size=4)`：
   - 所有损失有限；
   - `aux_weight > 0` 时 aux 参数获得非零梯度，且共享 CNN 参数发生变化；
   - `aux_weight = 0` 时策略 `state_dict` 键集合与 `ma` 完全相同（第三章检查点兼容）；
   - 2 通道配置的 CNN 第一层输入通道为 2。
   - 开启预测头不改变同种子 SAC 网络的初始权重；将辅助损失置零后，SAC 更新与基线一致。
   - Actor 阶段 CNN 的梯度与独立计算的辅助损失梯度一致，且梯度裁剪参数无重复。
3. 原有测试全部通过：`pixi run python -m unittest discover -s tests -t .`
4. `pixi run python -m scripts.training.train_parallel --scene tunnel_activate --methods ma ma_motion ma_aux ma_motion_aux --dry-run`
   （参数名以脚本实际 argparse 为准）能正确生成计划。

2026-11-15 审查验证：40 项无仿真测试全部通过，含四配置训练更新、保存/加载/续训和梯度隔离。
与 `main` 源码逐位对照，`ma`、`mamba`、`lstm`、`transformer` 在固定种子两次 CPU 更新后的权重与训练指标完全一致。
新增两项回归测试均能检出修正前的训练实现问题。上述 dry-run 正常生成 12 个任务、4 个波次；未启动仿真或正式训练。

## 6. 实验计划（实现完成后由用户在 AirSim 上执行）

- 场景：ENV3（`tunnel_activate`）、ENV4（`tunnel_mixed`），训练预算同第三章（各 2×10⁵ 步），3 个种子。
- 方法：`ma`、`ma_motion`、`ma_aux`、`ma_motion_aux`。
- 评估协议同第三章（筛选 → 冻结 → 盲测，每种子 100 回合）。
- 补充：`aux_weight ∈ {0.05, 0.1, 0.3}` 敏感性；`aux_target = depth_delta` 对比。

用户已选择不采用高度保持补偿、评估场景循环和碰撞对象记录这三处额外改动。
它们已从 `main` 另存到 `wip/env-tweaks` 分支（`d869921`），均未用于本轮实验。
`docs/` 的用户删除保持未提交状态。

2026-11-15 真实环境短测发现：原 reset 在传送至空中后等待、再调用 takeoff，
无人机已产生下降速度，服务器拒绝起飞；Future.join() 又未抛出 RPC 错误，导致回合从不稳定状态开始。
用户已批准四组统一修复初始化并重训 `ma`：reset 后先启用控制并解锁，再设置起始位姿，
执行 1 秒零速度指令稳定机体，并用 Future.get() 检查执行结果。回合内动作控制、奖励和终止条件不变，
不启用持续高度保持补偿。此初始化与历史第三章不同，所有四组都使用新流程，论文历史成功率仅作背景参照。

当前本机打包 ENV3 使用 `/Game/Tunnel_Col_Vol1/Maps/Tunnel_Activate`，
此前训练脚本默认的编辑器地图名 `ObsTunnel` 不在该仿真包中；当轮通过 `--map-path` 显式指定打包名称。
2026-11-16 已将训练脚本 ENV3 的默认地图更正为该打包名称，避免后续误用旧名称。

历史论文的 59.3% / 50.0% 可作为背景参照；正式消融表应采用同一评估协议下的结果。
复用第三章检查点前，核对训练配置、场景版本、种子和筛选/盲测流程；源码一致本身不足以证明评估条件一致。

## 7. 2026-11-15 修正及新一轮训练

按用户要求停止旧版 ENV3 两组探索训练并从头使用修正版训练；旧版模型、日志和源文件快照保留在
`reports/ch4_env3_launch_20261115/` 及原训练目录。两次训练的数据和进度不拼接。

- 先设置真实多旋翼动作空间再创建任务，使平滑惩罚按各维度真实范围归一化。
- 目标 Critic 已注册目标 CNN；每次仅通过 Critic 更新一次目标 CNN。
- 原奖励权重保持不变，分别记录进度、障碍、时间、平滑、裁剪修正、终止和超时分项。
- 评估区分墙钟时间、名义动作时长及 AirSim 原始状态时间戳；速度用三维世界速度及墙钟口径，
  不将尚未核实语义的时间戳称为真实仿真时间。与历史时间/速度字段口径不同。
- 每两万步后在回合结束时保存最新完整检查点，包含经验回放与随机状态；由于动态障碍和仿真
  物理状态没有恢复，断点续训仍不能保证轨迹完全一致。

隔离副本通过完整离线测试与真实 AirSim 运行链路短测后，修正内容已应用到主项目。

## 8. 2026-11-16 铁路动态场景适配

用户允许选择更合适的场景。基于本机既有铁路工程，候选场景为单线 S 弯动态检修地图
`/Game/RailwayTunnelPrototype/Maps/02_Conditions/SingleTrack/L_SingleTrack_SCurve_DynamicMaintenance`。
当前独立 `-game` 实测中检修车移动、工人位置在 20 秒观测内未变化；不把该工人描述为已验证移动。
该场景曾沿已知路线物理飞通；本轮在初始 NED `[0, 0, -1.5]`、统一终点
`[108.029625, 24.986841, -1.5]` 下用已知路径点引导再次到达，未启用持续高度补偿。
路线引导结果用于场景通行检查，不能作为自主避障性能结果。

铁路地图冷启动时，首张 DepthPlanar 可有正常尺寸却包含异常数值。现统一在第一次取深度前
激活并丢弃 Scene 与 DepthPlanar 的压缩帧，等待 0.25 秒；持续取图失败时抛出错误。
独立冷启动复测、坐标标定和运动记录见 `reports/ch4_railway_scene_20261116/`。
当前 Python 代码中的铁路场景需要先启动相应 UE 游戏实例并使用 `--no-launch-airsim`，
且车辆生成偏移需要保持与标定时相同。旧公路场景及不同相机/时钟下的结果不能直接拼接。

## 9. 2026-11-17 公共时序训练一致性修复

用户确认先修复训练历史窗口与 TD 上下文，gamma 保持 0.99，不改变网络结构、奖励、学习率或在线历史长度。

- `seq_len` 仍表示每条样本参与损失的连续位置数；例如批大小64、长度16仍最多有1,024个有效位置。
- 回放额外返回 `context_observations`，长度为 `history_len + seq_len`，包含真实历史前缀、当前位置和末尾下一观测。
  `history_lengths` 记录每个位置在本回合中实际已有的历史长度；回合开头不会把左侧占位帧送入时序网络。
- Actor、Critic、目标Critic在每个位置都使用与在线相同的滑动窗口；下一状态窗口向前移动一帧，保留当前历史中仍应存在的观测。
  真实终止仍不自举；超时截断继续使用保存的真实最终观测，回放没有跨回合拼接。
- CNN按连续上下文编码，时序窗口复用特征；分组前向后按原位置重组输出，保留重叠历史的梯度。
  辅助任务使用同一对齐后的上下文，原有CNN梯度隔离和优化器配置保留。
- 推理、权重结构、原始回放存储和检查点格式不变。旧检查点仍可读取；旧模型的在线动作输出不因本次训练代码修复而改变。

本节更新早先“保持旧训练更新逐位兼容”的范围：新代码中的零权重辅助分支与新MA基线仍一致，
但修复后的训练不再与旧的16步前缀训练逐位相同。旧论文结果、模型及冻结源码保留；后续方法比较应使用共同修复的新协议，
重新保存源码快照和训练结果，不能将旧结果直接解释为修复后性能。

验收包括逐步在线Actor与训练批量窗口的当前/下一状态对照、Critic与TD目标对照、终止/截断和填充掩码、
历史前缀梯度及辅助梯度隔离，共20项测试通过。两份旧真实检查点在CUDA、批大小64、长度16下各完成一次可丢弃副本的离线更新，
原文件摘要不变；没有新增环境交互训练或保存更新后的策略。完整记录见
[历史上下文修复记录](reports/ch4_control_pilot_20261117/history_context_fix_20261117/README.md)。

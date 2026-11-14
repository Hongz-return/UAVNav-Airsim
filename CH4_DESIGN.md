# 第四章设计：运动感知与动作条件深度预测的 MA-SAC 扩展

> 目标：在第三章 MA-SAC（CoordAtt + Mamba 序列 SAC）基础上，提升动态障碍（ENV3，基线成功率 59.3%）
> 与混合障碍（ENV4，基线 50.0%）场景的避障成功率。第三章方法的行为必须保持不变，以保证对比公平。

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
它们已从 `main` 另存到 `wip/env-tweaks` 分支（`d869921`），第四章分支仅保留感知接口接线，动力学与任务代码和 `main` 一致。
`docs/` 的用户删除保持未提交状态。

历史论文的 59.3% / 50.0% 可作为背景参照；正式消融表应采用同一评估协议下的结果。
复用第三章检查点前，核对训练配置、场景版本、种子和筛选/盲测流程；源码一致本身不足以证明评估条件一致。

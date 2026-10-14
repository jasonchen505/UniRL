# UniRL 深入学习文档

> 学习对象：上游 `Tencent-Hunyuan/UniRL`（`main`，HEAD `0ac8bc3e`，2026-10-14）
> Fork：`jasonchen505/UniRL`，`main` 停在 `80b24995`（2026-07-14），落后上游约 3 个月。
> 本文档写在 fork 的 `notes/learning` 分支，`main` 未动。
> 口径：只读调研，未跑 GPU、未跑测试、未提 PR、未联系 maintainer。所有结论标注来源，区分已验证（文档/代码阅读）与未验证（未执行）。
> 快照时间：2026-10-14 约 12:38 UTC。

## 1 一句话定位

UniRL 是腾讯混元的统一多模态模型强化学习框架：一套 RL 后训练循环（采样 → 评分 → 算优势 → 更新策略 → 权重同步回 rollout worker），横跨扩散（图像/视频）、自回归（LLM/VLM/Omni）、Prompt-Enhancer（AR 改写 + 扩散奖励）、统一 AR+扩散模型四类领域，再叠加 SFT、Agentic、Async 三条训练路径。

与 verl / verl-recipe 的关系：verl 是通用 RL 训练器，verl-recipe 是社区 recipe 集合；UniRL 是自带完整分布式 runtime + rollout 引擎 + 模型包 + 奖励服务的端到端框架，对多模态（尤其扩散/统一模型）的覆盖远深于 verl 生态。Uni-agent 是把任意 agent harness 接入 RL；UniRL 的 agentic 是框架内建的多轮工具使用训练路径。ProRL-Agent-Server 是 Rollout-as-a-Service（harness 零侵入，网关拦截）；UniRL 是训练框架本身，不做网关代理。

## 2 仓库快照

| 项 | 值 |
|---|---|
| 仓库 | `Tencent-Hunyuan/UniRL`，Framework for Unified Multimodal Model Reinforcement Learning |
| Stars / Forks | 1014 / 99（查询时实时值） |
| `open_issues_count` | 122 = open issues 53 + open PRs 69（不是纯 issue 数） |
| 默认分支 / 语言 | `main` / Python |
| 创建时间 | 2026-06-08T12:29:36Z |
| 许可证 | Apache-2.0 |
| 上游 HEAD | `0ac8bc3e`，2026-10-14，`feat(reward): add guarded CUDA MPS runtime (#492)` |
| 版本 | v0.2.0（2026-10 发布） |

根目录：`AGENTS.md`、`INSTALL.md`、`README.md`、`pyproject.toml`、`requirements.txt`；目录 `unirl/`（框架）、`examples/`（Hydra recipes）、`datasets/`、`benchmarks/`、`experimental/`、`lint/`、`CPPO/`、`DRPO/`、`FlowDPPO/`（三篇团队论文教程目录）、`unirl-reward-service/`（独立奖励服务）。

## 3 架构：分层与依赖方向

`unirl/` 顶层职责（来源：`unirl/README.md` Module Map + 目录实测）：

| 路径 | 职责 |
|---|---|
| `unirl/train_*.py`（8 个） | Hydra entrypoint，组合并校验 recipe，交给对应 Trainer |
| `unirl/trainer/` | 编排中枢：`base.py` + 领域 Trainer（diffusion / ar / pe / unified_model / sft / agentic / async_ar / async_diffusion / async_rollout）+ `residency.py`、`rollout_sleep.py`、`eval_suites.py`、`hydra.py`（`parse_hydra_cfg`） |
| `unirl/config/` | 扁平 recipe 校验：`require.py`（`require()` 前置条件）、`validation.py`（跨组件契约）；本身不持有 config dataclass |
| `unirl/distributed/` | Ray 运行时：`group/`（`Remote` / `Worker` / `Handle` / `DevicePool` / `placement` / `dispatch`）、`tensor/`（`TensorMeta` 数据平面 + `backend/{colocate_store,gpu_store,transfer_queue}`）、`weight_sync/`、`local.py`、`utils.py` |
| `unirl/rollout/` | Rollout：`engine/`（引擎）、`env/`（agentic 环境/工具协议）、`manager/`（driver 侧调度 `dispatch.py` / `buffers.py` / `filters.py` / `rollout.py`）、`harness/`、`engine/base.py`（`BaseRolloutEngine`）、`engine/ports.py`、`engine/sigma_verify.py` |
| `unirl/reward/` | `service.py`（`RewardService`，恰包一个 backend）、`base.py`、`remote.py`（HTTP 远程）、`managed_process.py`（独立进程 scorer）、`stack.py`、`local/`（约 24 个本地 scorer + `registry.py`） |
| `unirl/algorithms/` | 训练侧 loss：`base.py`（`StageAlgorithm`）+ flowgrpo / grpo / gspo / ppo / cppo / dppo / drpo / flowdppo / diffusionnft / diffusionopd / bagel_flow_unigrpo / cosmos3_sft / sft |
| `unirl/models/` | 24 个模型包 + `types/` 共享契约层 |
| `unirl/train/` | 训练栈：`backend/{fsdp,veomni}`、`stack/{single,unified,base,anchor,loss_scale,planner}`、`sft/track_builder.py`、`lora.py`、`ema.py`、`optim.py`、`configs.py` |
| `unirl/sde/` | SDE：`kernels.py`、`index_schedule.py`、`noise.py`、`runtime.py`、`unipc.py`（σ schedule / SDE-index schedule / 初始噪声） |
| `unirl/data/` | `data_source.py`、`datasets.py`、`sft.py`（SFT manifest 读取） |
| `unirl/types/` | 共享类型契约：`sample.py`（`Sample`/`Part`）、`primitives.py`、`sampling.py`、`noise_recipe.py`、`reward.py`、`advantages.py`、`loss_agg.py`、`media.py`、`conditions/`、`segments/` |
| `unirl/tools/` | 导出/检查：`export_adapter.py`、`export_full.py`、`_checkpoint.py`、`precompute_minimax_h3.py` |
| `unirl/utils/` | 领域无关小机制：日志、dtype、media/video、指标聚合、profiling、内存监控、wandb、graceful shutdown |

依赖方向（`unirl/README.md` Ownership）：entrypoints / trainer → rollout · reward · algorithms · train → models · data · sde → types · config · distributed，由 `lint/check_core_dependencies.py` 锁定部分方向。已知债务：`models/{hunyuan_image3,qwen3_5,qwen3_moe}` 触 `train/backend/veomni`、`models/cosmos3` 子类化 SFT track builder、`algorithms/` 翻转 `train/lora` adapter。

设计判断：分层是干净的，但三处债务都是“上层反向依赖下层实现细节”的典型破口。读代码时优先看这三处是否还在扩大。

## 4 Entrypoint → Config → Trainer

每个 entrypoint 都是 `@hydra.main(config_path="../examples", config_name=<默认 recipe>)`，构造对应 Trainer 后调 `trainer.train(...)`。

| Entrypoint | 默认 recipe | Trainer |
|---|---|---|
| `unirl/train_diffusion.py` | `diffusion/sd3/sd3_trainside` | `DiffusionTrainer`（`trainer/diffusion.py`） |
| `unirl/train_ar.py` | `ar/qwen_vl_grpo_geo3k_mc_4x8` | `ARTrainer`（`trainer/ar.py`） |
| `unirl/train_pe.py` | `pe/pe_trainside_pickscore` | `PETrainer`（`trainer/pe.py`，`diffusion_cfg` + `ar_cfg` 双路） |
| `unirl/train_unified_model.py` | `unified_model/hi3_vllmomni` | `UnifiedModelTrainer`（`trainer/unified_model.py`，`ar_rollout_cfg` / `dit_rollout_cfg` + `algorithm.ar` / `algorithm.image` 双算法） |
| `unirl/train_sft.py` | `sft/qwen3_sft` | `SFTTrainer`（`trainer/sft.py`，`train(num_steps=...)` 而非 `num_rollouts`） |
| `unirl/train_agentic.py` | `deep_research/deep_research_search_judge` | `AgenticTrainer`（`trainer/agentic.py`，包在 `GracefulShutdown` 中） |
| `unirl/train_async_ar.py` | `ar/qwen3_grpo_4b_base_dapo_sglang_async` | `AsyncARTrainer`（`trainer/async_ar.py`，含 `train_fraction` / `max_inflight` / `buffer_max_staleness`） |
| `unirl/train_async_diffusion.py` | `diffusion/bagel/bagel_vllmomni_async` | `AsyncDiffusionTrainer`（`trainer/async_diffusion.py`，`layout="separate"` 写死） |

Config 机制（`unirl/config/README.md` + `examples/README.md`）：recipe 是单文件扁平 YAML，首行必须是 `# @package _global_`；所有组件用 `_target_` 点路径实例化，无 Hydra config group、无 `defaults:` 列表；共享值用 `${...}` 插值，部署路径用 `${oc.env:...}`。实例化分工：`trainer/hydra.py` 的 `parse_hydra_cfg` 只解析顶层 `_target_`，`distributed/group/worker.py` 的 `Worker._resolve_init_kwargs` 在 worker 侧递归构造嵌套 `_target_`（不用 `hydra.utils.instantiate`）。校验两层：dataclass `__post_init__` 经 `require()`（当前唯一实际运行层）+ `config/validation.py` 跨组件契约（部分仅为意图、非自动门禁）。Recipe 命名：`<model>[_<task>][_<size>][_<algorithm>][_<engine>][_<backend>][_<adapter>][_<topology>]`。

注意：`validation.py` 部分契约“仅为意图、非自动门禁”——这是框架里最容易藏静默错误的地方，改 recipe 时不要假设校验会拦住你。

## 5 运行时数据流

依据：`unirl/README.md` Runtime Data Flow + `trainer/README.md` + `distributed/README.md` + `rollout/README.md` + `reward/README.md` + `train/readme.md` + `algorithms/README.md`。未做运行时/GPU 验证。

### 5.1 同步单步（`train_step` 导体序列）

1. Entrypoint 组合 recipe → Trainer 在 `placement(...)` 作用域内建远程图，一个共享 bundle 串起 `bundle → pipeline(bundle) → backend(bundle) → reward → algorithm → stack`，再建 rollout engine。布局决定拓扑：`colocate` = train + rollout 同 slab 兄弟；`separate` = 两个不交 `placement` slab + 一次跨 slab 握手（weight sync 用）。
2. `wake_up` →（若到点）权重同步 → `rollout.generate(sample)`：trainer 构建带 prompt 血缘的请求 `Sample`（`Sample.request` + `fork` 生成空 Part）；引擎返回同一 lineage、生成 Part 已填充（conditions / segment / primitive map / media preview）。
3. `RewardService.score_and_attach(sample)`：恰包一个 backend——本地单 scorer（PickScore / CLIP / HPS / OCR / GenEval2 / MathVerify / `llm_judge` / `mc_exact_match` / 视频/音频 scorer 等，注册表 `reward/local/registry.py`）或 `RemoteRewardBackend`（HTTP 客户端，调 `unirl-reward-service/` 多奖励面板 + 加权聚合 `weighted_sum/mean/min/max`）。NaN/null/失败停住该步而非污染 GRPO 组。
4. `Part.compute_advantages(...)`：组内 z-score 得 advantages（`normalize=True` 时 `(reward-group_mean)/(group_std+eps)`）。
5. `TrainStack.train_track(...)`（`train/stack/single.py`）：segment 上设备 → `prepare_segment_anchors` 冻结 π_old → 按 `num_updates_per_batch` 做多 optimizer 步，每步 micro-batch 循环 `compute_loss_and_backward`。算法只做 loss 数学：`stage.replay(...)` 得 `new_logp`，与冻结 π_old anchor + advantages 组合；`TrainStack` 在与训练完全相同的 mini/micro 切片几何上重算 anchor，以抵 bf16 批量形状敏感导致的 ratio 漂移。`FSDPBackend`（`train/backend/fsdp/backend.py`）持有 FSDP2 包裹模块 + optimizer/scheduler/EMA；`optimizer_step` 是唯一关口（clip → step → schedule → EMA，非有限 grad norm 时整步跳过）；结构注入（`inject_lora` / `inject_nft` / `inject_mirror`）在 `fsdp_wrap` 之前一次完成。`UnifiedModelTrainStack`（`stack/unified.py`）驱动 ar + image 双算法反向累积进同一共享 optimizer 步。
6. 专用 rollout 模式下，trainer 权重经 `distributed/weight_sync/` 同步回 rollout workers。Weight-sync 两条后端：`weight_sync/lora/`（`local.py` / `remote.py`）与 `weight_sync/full/`（`nccl.py` / `ipc.py` / `tensor.py` / `checkpoint.py` / `ckpt_engine_ipc.py`），另有 `weight_sync/transfer/`（`bucketed_transfer.py`、`ipc_dispatch.py`、`vllm_native_engine.py` 等传输层）。后端由 recipe 的 `sync:` 段选择。

### 5.2 分布式 runtime

逻辑 worker 代码是 `Remote`，跑在 `Worker`（每 GPU 一个 Ray actor）内；driver 侧 `Handle` 把 `@distributed` 方法调用按 dispatch 模式扇出：`BROADCAST` / `SCATTER` / `DP_SCATTER`（默认，generate 与 train_track 的分片方式）/ `DP_SCATTER_HEAD`。`DevicePool` / `placement` 把 GPU 切成 slab。数据平面是 handle 代理：产出张量被脱水为 `TensorMeta`（字节留在 worker store），driver 只传 handle、只 hydrate 小的 reward/advantage；`localize` 在消费者处解析：同 worker 本地读，不同 slab 则一次 worker→worker NCCL 跳，永不经 driver。Transport 三选一（`distributed/tensor/backend/`）：`colocate_store`（默认，跨 GPU 经 NCCL 解析）、`gpu_store`（每 GPU CUDA-IPC，多槽 colocate）、`transfer_queue`（全局总线：`simple` 内存版零基建 / `mooncake` RDMA 生产版）。`transport_kind` 是集群级选择；`BaseTrainer` 从顶层 cfg（`num_devices` / `transport_kind` + 可选 TransferQueue bootstrap）建 `DevicePool`。

部署模式：trainside 采样（训练 worker 直接采样，无 `sync:`）、separate rollout（独立 GPU 池 + `sync:` 必需）、colocated rollout（共享 GPU bundle + offload/onload + `sync:` 必需）。

### 5.3 Rollout 引擎

`BaseRolloutEngine`（`engine/base.py`）是 `Remote`，实现同步 `generate(sample)`；批量引擎用 `DP_SCATTER` 分发。六个单轮引擎：`trainside`（进程内，训练 actor 的 pipeline 即采样器）、`sglang_diffusion`、`sglang`、`vllm_omni`（BAGEL / HI3 / SD3 / HunyuanVideo）、`fastvideo`（加速视频采样）、`composed`（AR 子 + diffusion 子串联，供 PE）。`agentic` 在其一之上包环境做多轮轨迹。另有 `engine/vllm/` 目录存在，但 `rollout/README.md` 六引擎枚举未单列它，地位未验证。

数值一致性边界：扩散引擎逐字读取生成 Part 上钉住的 σ（`engine/sigma_verify.py` 校验后端实际所用），且 `trainside` / `sglang_diffusion` / `vllm_omni` 从同一 driver 编写的 `NoiseRecipe`（`types/noise_recipe.py`）解析出 bit-identical 的 `x_T`；`fastvideo` 只按 noise-group id 派生 seed，仅分组一致。

### 5.4 模型 bundle

每包实现 bundle / pipeline / stage / conditions 契约：`bundle.py`（纯权重容器，`from_config`；大模型另有 `from_meta_config` + `materialize()` 分片加载）、`pipeline.py`（`generate(sample) -> sample` 自同态）、`diffusion.py`/`ar.py`（`DiffusionStage` 暴露 `diffuse`/`replay`/`predict_noise_at_step`；`ARStage` 暴露 `autoregress`/`replay`；`trainable_module()` 在 stage 上）、`conditions.py`。核心正确性：rollout 前向与 train replay 跑在同一个 `bundle.transformer`（FSDP 就地变异的对象）上，否则 GRPO ratio 会静默漂移；σ 是 engine-pinned（pipeline 读 Part 的 `sigmas`，为 `None` 则 raise）。

### 5.5 Agentic / Async / SFT

- Agentic：`AgenticTrainer` 每次 rollout 前同步当前训练权重；driver 侧 manager 按远程 engine slot 并发分发 sibling 轨迹；等完整 GRPO 组齐后经 `RewardService` 对终态答案评分；每条成功轨迹得一个组归一化 advantage，应用于其全部生成 assistant turn；失败轨迹排除。环境/工具/轨迹契约见 `rollout/env/README.md`；harness 在 `rollout/harness/`。
- Async：训练与 rollout 在独立 GPU slab 上重叠；`RolloutManager` 的 progress 线程分发有界工作并观察就绪，trainer 线程收集、组装 agentic 兄弟、应用 filter；Async AR 在收集后立即向同一 manager 补充（下一轮生成与评分/训练重叠）；发布与持久化边界会 quiesce 该 manager。`train_async_ar` 有 `buffer_max_staleness`；`train_async_diffusion` 保持 rollout 常驻（sync 入口会 park 它）。
- SFT：消费监督 manifest（`SFT_DATA` / `SFT_EVAL_DATA`），无 rollout 路径；`train/sft/track_builder.py` 建 track；Cosmos3 仅 SFT（`Cosmos3Pipeline.generate` 未实现）。

## 6 算法

验证口径：文档/代码阅读已验证；未验证：无 GPU 复现、无测试运行；三篇自研论文实验数字未对照 arXiv 全文核对，仅转述仓库教程表述。

### 6.1 Flow-DPPO（扩散 / flow 侧）

- 论文：*FlowDPPO: Divergence Proximal Policy Optimization for Flow Matching Models*，arXiv:2606.11025。
- 公式直觉：沿用 GRPO 式 SDE rollout + 组内相对优势，但不用 PPO 统一 clip。两 SDE 转移分布是等方差高斯，KL 有闭式解（论文 Eq.14）：`KL = ‖μ_old − μ_θ‖² / (2σ_t²)`。仓库用 latent 维度 mean-reduced 版，`kl_score = mean[(μ_θ−μ_old)²/(2σ_t²)]`，故 `kl_mask_threshold` 按 mean 校准，不是论文全范数。
- Mask（论文 Eq.18）：仅当 KL 高且推动过激才丢弃：`drop = (kl_score ≥ τ) ∧ [(ρ>1 ∧ A>0) ∨ (ρ<1 ∧ A<0)]`，损失 `E[−A·ρ·1[¬drop]]`。低 KL 全放行；高 KL 但“纠正方向”（如 `A>0, ρ<1`）也放行。另有可选 `beta·KL(π_θ‖π_ref)`（Eq.17），`π_ref` 为 LoRA-disabled 基座，需 LoRA recipe，默认 `beta=0`。
- 实现：`unirl/algorithms/flowdppo.py`（`FlowDPPO`、`_flowdppo_kl_adv_loss`、`_gaussian_kl_div`）；SDE/replay：`unirl/models/sd3/diffusion.py`、`unirl/sde/kernels.py`；教程 `FlowDPPO/README.md` + `FlowDPPO/config.yaml`；recipe `examples/diffusion/sd3/sd3_flowdppo.yaml`（另有 `sd3_flowdppo_vllmomni.yaml`、Qwen-Image-Edit 的 FlowDPPO SGLang recipe）。关键 knob：`kl_mask_threshold=1e-5`、`add_kl_coefficient=true`。多更新支持（`old_logp` + `old_means` 在 `prepare_segment` 一次冻结）。
- 与标准算法区别：vs FlowGRPO/GRPO（扩散）：FlowGRPO 是 PPO-clip 比值 `ρ`；Flow-DPPO 保留无 clip 的 `−A·ρ`，只用精确 KL+优势方向做 0/1 mask，不统一收紧所有 step。vs DiffusionNFT：NFT 无比值、无 SDE rollout 轨迹，用前向加噪 + 双 adapter 重建 MSE；Flow-DPPO 是 on-policy SDE 比值 + KL 门。vs DanceGRPO/MixGRPO：后两者仍是 FlowGRPO 损失，只换 SDE 策略/调度；Flow-DPPO 换损失本身。vs LLM DPPO：LLM 需 Binary/Top-k 近似估计散度；flow 模型直接得等方差高斯 KL，无需近似。

### 6.2 DRPO（AR / LLM 侧，平滑正则）

- 论文：*Rethinking the Divergence Regularization in LLM RL*，arXiv:2606.09821。
- 公式直觉：谱系 PPO → GRPO → SPO → DPPO → DRPO。DPPO 用 Binary-TV 硬 mask `|π−μ| ≤ δ`，过界梯度直接归零；DRPO 把它光滑化为优势加权二次正则（论文 Eq.8）：`L = Σ [r·Â − |Â|·μ·(r−1)²/(2δ)]`，其中 `r=π/μ`，`μ` 为 behavior token 概率。等价于 token 自适应比值界 `|r−1| ≤ δ/μ`。梯度权重 `w = 1 − sign(Â(r−1))·|π−μ|/δ`：发散方向衰减到 0 后变负（纠正信号），收敛方向放大 >1。关键是多乘的 `μ`：SPO 罚 χ² 散度 `Σ(π−μ)²/μ`（对低概率 token 超敏感），DRPO 罚 ℓ₂² `Σ(π−μ)²`，权重有界。
- 实现：`unirl/algorithms/drpo.py`（`DRPO`、`_drpo_loss`）；教程 `DRPO/README.md` + `DRPO/config.yaml`；recipe `examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml`（Qwen3-4B-Base，DAPO-Math，SGLang，`MathVerifyRewardScorer` 规则奖励）。关键 knob：`drpo_epsilon=12.5`（代码名 ε = 论文 δ）、`loss_agg_mode=seq-mean-token-sum-norm`、`sampling_temperature` 必须等于 rollout temperature。`supports_multi_update=False`，`num_updates_per_batch` 必须为 1（复用 rollout logp 做 `old_logp`，不冻结 train 侧 anchor）。
- 与标准算法区别：vs GRPO（AR）：GRPO 用 PPO-clip 比值；DRPO 不用 clip，用光滑二次正则做信任域。vs DPPO（AR，`unirl/algorithms/dppo.py`）：同 Binary-TV 几何，但 DPPO 是硬 0/1 mask，DRPO 是光滑权重 + 越界纠正。vs SPO：SPO 同二次型但无 `μ` 因子；DRPO 加 `μ` 后把 χ² 罚变成 ℓ₂² 罚，低概率 token 权重不再无界。vs CPPO：同 DPPO 信任域出身，但 CPPO 保留硬 mask 并加位置/累积预算；DRPO 选择光滑化。

### 6.3 CPPO（AR / LLM 侧，硬 mask + 位置/累积预算）

- 论文：*Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning*，arXiv:2606.10968。
- 公式直觉：谱系 PPO → GRPO → DPPO → CPPO。DPPO 对所有 token 用同一 `δ`；CPPO 认为误差沿序列累积，需位置感知 + 累积预算。`D_t=|π−μ|`，位置权重 `w_t = w_min + (1−w_min)(T−t)/(T−1)` 从 1 递减到 `w_min`，加权偏离 `Z_t=w_t·D_t`，前缀和 `S_{t−1}=Σ_{j<t}Z_j`、`W_{t−1}=Σ_{j<t}w_j`。有效阈值 `c_t = min(δ, δ + δ_b^seq·W_{t−1} − S_{t−1})`；`keep ⇔ A_t(r_t−1) ≤ 0 ∨ Z_t ≤ c_t`（向 behavior 靠拢的更新恒保留）。损失仍是 DPPO 的 `−A·r`（仅 mask 变化）。动态预算 `δ_b^seq = clamp(P90(D_{1:T}), δ_b^min, 2·δ_b^min)`（论文 Eq.22）。
- 实现：`unirl/algorithms/cppo.py`（`CPPO`、`_cppo_loss`、`_cppo_mask`，mask 在 `no_grad` 下按 `torch.split(segment.lengths)` 逐序列算，前缀和不跨序列）；教程 `CPPO/README.md` + `CPPO/config.yaml`；recipe `examples/ar/qwen3_cppo_30b_a3b_base_dapo_sglang.yaml`（Qwen3-30B-A3B-Base，DAPO-Math，SGLang，全权重同步）。关键 knob：`cppo_delta=0.20`（30B-A3B；dense 论文 Table 3 为 0.15）、`cppo_w_min=0.8`、`cppo_delta_b=0.02`。`old_logp_source=rollout` 为 canonical 模式；rollout anchor 跨 mini-batch 冻结，多更新 OK（与 DRPO 单更新限制不同）。
- 与标准算法区别：vs GRPO（AR）：GRPO 统一 clip；CPPO 用 Binary-TV 绝对概率偏移 + 位置/前缀预算做硬 mask。vs DPPO（AR）：CPPO 是“DPPO + 位置权重 + 累积前缀预算”的硬 mask 兄弟；损失项不变。vs DRPO：同 Binary-TV 信任域，但 DRPO 光滑化（二次正则 + 纠正权重），CPPO 保留硬 keep/reject 并让阈值随位置/前缀收紧。vs Flow-DPPO：Flow-DPPO 的 mask 也是 KL+优势方向，但用于扩散高斯转移且 KL 精确；CPPO 用于离散 token，散度是采样 token 的 `|π−μ|`。

### 6.4 标准算法对照

| 标准算法 | 实现 / 变体方式 | 与三自研的核心差 |
|---|---|---|
| GRPO（AR） | `unirl/algorithms/grpo.py`，PPO-clip 比值 + 组内相对优势，无 critic | 三自研都抛弃“统一 clip”：Flow-DPPO→KL mask；DRPO→光滑二次正则；CPPO→Binary-TV + 位置/累积 mask |
| FlowGRPO（扩散版 GRPO） | `unirl/algorithms/flowgrpo.py`，SDE rollout + 每 SDE step clip 比值 | Flow-DPPO 直接复用其 SDE rollout，只换损失 |
| DiffusionNFT | `unirl/algorithms/diffusionnft.py`，ratio-free，双 adapter 重建 MSE，`r∈[0,1]` 前向加噪，`requires_ema_rollout=True`，`supports_multi_update=False`，另有 `ref_deviation_coef` 参考惩罚与 `adv_std_saturate` | 与 Flow-DPPO 同为扩散 RL，但无 rollout 轨迹/比值/信任域 mask；v0.2.0 新增 EMA shadow + 参考 KL |
| DanceGRPO | 不是独立类：`FlowGRPO` + `DanceSDEStrategy`（`pipeline.strategy`），Coupled 范式（早期 SDE 注噪、后期 ODE） | 只换 SDE 策略，不换损失；Flow-DPPO 换损失 |
| MixGRPO | 不是独立类：`FlowGRPO` + `WindowScheduler`（`sampling.scheduler`，`unirl/sde/index_schedule.py`），滑动窗口、只访完整窗口；v0.2.0 新增 `decay`/`exp_decay`（#489） | 只换 timestep 调度，不换损失；注意 `max_timesteps = sampling_steps−2` 坐标系与该 repo sample dict 不同（`unirl/sde/README.md` Gotchas） |
| 其它（v0.2.0 新增） | DPPO（AR 硬 mask）、GSPO、PPO+GAE+value head、DiffusionOPD（教师锚定蒸馏，`requires_advantages=False`，忽略优势）、SFT | DiffusionOPD 是蒸馏非 RL；GSPO 为序列级；PPO 带 critic，与 GRPO 系 critic-free 不同 |

算法家族补充（`algorithms/README.md`）：`StageAlgorithm` = loss combine + 声明式旋钮（`requires_ema_rollout`、`supports_multi_update`、`anchor_fields`/`recomputes_anchor`，后两者重配采样器与训练栈）。

## 7 模型支持矩阵

来源：`unirl/models/README.md` #support-matrix 全量；根 `README.md` #model-support 为缩略版。图例：✅ 可跑（包 + 至少一个 recipe）· 🧩 bundle-only（只有 bundle，无 pipeline，挂在别的包 pipeline 下跑）· 🧪 SFT-only（无 `generate`，无 RL rollout）· 🔗 composed（无自有权重，组合其它行）。“引擎”列只列有 recipe 的引擎，未列 = 无 adapter 或无 recipe、未验证。SFT 列只标矩阵明确列出 `train_sft` 的行。

| # | 包 / 模型 | 模态 | 入口 | 有 recipe 的引擎 | 限定 / 状态 |
|---|---|---|---|---|---|
| 1 | `sd3/` / SD 3.5-medium | 文→图 | diffusion, SFT, PE(图侧) | trainside, sglang_diffusion, vllm_omni | ✅；部分 recipe scorer-dependent |
| 2 | `qwen_image/` / Qwen-Image | 文→图 | diffusion | trainside, sglang_diffusion, vllm_omni | ✅ |
| 3 | `qwen_image_edit_plus/` / Qwen-Image-Edit-2511 | 文+图→图 | diffusion | trainside, sglang_diffusion | ✅；无 vllm_omni adapter |
| 4 | `flux2_klein/` / FLUX.2-klein-base-4B/-9B | 文→图；文+图→图 | diffusion | trainside, sglang_diffusion | ✅ |
| 5 | `z_image/` / Z-Image | 文→图 | diffusion | trainside, sglang_diffusion | ✅ |
| 6 | `boogu_image/` / Boogu-Image-0.1-Base | 文→图 | diffusion | trainside | ✅ trainside-only |
| 7 | `wan21/` / Wan2.1-T2V-1.3B / I2V-14B | 文→视频；图→视频 | diffusion, SFT | trainside, sglang_diffusion, fastvideo | ✅；fastvideo 需 `$FASTVIDEO_PATH` |
| 8 | `wan22/` / Wan2.2-T2V/I2V-A14B | 文→视频；图→视频 | diffusion | trainside, sglang_diffusion, fastvideo | ✅；扩展 `wan21/` 双专家 |
| 9 | `wan22_v2v/` / Wan2.2-T2V-A14B（V2V pipeline 复用 `wan22/`） | 视频→视频 | diffusion | trainside | ✅ trainside-only；checkpoint-path-required |
| 10 | `hunyuan_video10/` / HunyuanVideo | 文→视频 | diffusion | trainside, sglang_diffusion | ✅ |
| 11 | `hunyuan_video15/` / HunyuanVideo-1.5-480p_t2v | 文→视频 | diffusion | trainside, vllm_omni | ✅；无 sglang_diffusion adapter |
| 12 | `ltx2/` / LTX-2 | 文→视频 | diffusion | trainside, sglang_diffusion | ✅；I2V 在 config 中但无 recipe，不声称支持 |
| 13 | `ltx2/` / LTX-2.3 | 文→音+视频 | diffusion | trainside | ✅ trainside-only |
| 14 | `minimax_h3/` / MiniMax-H3(33B dense) | 文→视频+音频 | diffusion | trainside | ✅ trainside-only；checkpoint-path-required；canvas/帧数/1节点×8卡下限；vllm_omni 后端待 #378/#420 |
| 15 | `hunyuan_image3/` / HunyuanImage-3.0-Instruct | 文/文+图→图 | unified_model | trainside, vllm_omni | ✅ |
| 16 | `bagel/` / BAGEL-7B-MoT | 文→图；文+图→图；文+图→文 | diffusion, AR, unified, async_diffusion, SFT | trainside, vllm_omni | ✅；vllm_omni/unified recipe checkpoint-path-required；AR VQA recipe 多节点 2×8 |
| 17 | `sensenova_u1/` / SenseNova-U1.5-8B-MoT | 文→图（pixel flow） | diffusion | trainside | ✅ trainside-only；多节点 + scorer-dependent(OCR) |
| 18 | `janus_pro/` / Janus-Pro-7B | 文→图(AR)；文+图→文 | AR | trainside | ✅ trainside-only；LoRA-only |
| 19 | `qwen3/` / Qwen3-4B-Base（deep research 用 Instruct，PE 用 0.6B；vllm recipe 用 30B-A3B） | 文→文 | AR, async_AR, agentic, PE(改写侧), SFT | sglang, vllm, agentic(sglang), trainside(仅 PE recipe) | ✅；vllm recipe 仅 30B-A3B + checkpoint-path-required；agentic recipe scorer-dependent(`llm_judge`→`$JUDGE_URL`) |
| 20 | `qwen3_moe/` / Qwen3-30B-A3B（VeOmni EP 堆叠） | 文→文 | AR（走 `qwen3/` pipeline） | sglang | 🧩 bundle-only；仅 `bundle.py`，无 pipeline/stage/conditions；checkpoint-path-required |
| 21 | `qwen3_5/` / 9B / 35B-A3B | 文→文；文+图→文 | AR | sglang | ✅ sglang-only；checkpoint-path-required；无 SFT recipe；hybrid-attention fast path 需 `[sglang]` extra 的 `flash-linear-attention` |
| 22 | `qwen_vl/` / Qwen2.5-VL-7B-Instruct | 文+图→文 | AR, SFT | trainside, sglang | ✅；目标 Qwen2.5-VL（非 Qwen3-VL）；RL recipe 多节点 4×8 |
| 23 | `qwen3_omni/` / Qwen3-Omni-30B-A3B-Instruct(Thinker) | 文/图/音/视频→文 | AR, SFT | vllm_omni | ✅ vllm_omni-only；LoRA-only；checkpoint-path-required；TP 须整除音频塔 20 heads |
| 24 | `cosmos3/` / Cosmos3-Nano | 视频(+action)预测 | SFT | —（`Cosmos3Pipeline.generate` 未实现） | 🧪 SFT-only；dataset-dependent（`datasets/droid100/`） |
| 25 | `pe/` / Qwen3-0.6B 改写 + SD3 图 | 文→文→图 | PE | trainside, composed(sglang + sglang_diffusion) | 🔗 composed |
| 26 | `examples/deep_research/` / `qwen3/` + agentic engine(sglang) | 多轮工具使用 | agentic | agentic(sglang) | 🔗 工作流非模型；scorer-dependent(`llm_judge`) |

未支持（明确写出）：`sglang_diffusion` adapter 注册表另有 `FluxAdapter`（plain FLUX）、`MochiAdapter`，但无 `unirl/models/` 包、无 recipe，不算模型行。

## 8 v0.2.0

来源：核对 `releases/tag/v0.2.0` vs `README.md` / `pyproject.toml` / `INSTALL.md`。结论：README vs release 一致，未发现冲突的版本号或模型数；唯一口径差：根 README 模型表是缩略版，引用以 `unirl/models/README.md` 全矩阵为准。

- Release 基本信息：发布页显示 `08 Oct 08:43`，根 `README.md` News 标 `[2026-10]`；自述“首个自 `v0.1.0` 后的 release，约 300 个 merged PR”。`main` 于 2026-08-02 re-rooted，`v0.1.0`↔`v0.2.0` 的 GitHub compare 不可用。当前 HEAD（2026-10-14）已在 v0.2.0 之后，不可把当前 main 等同于 v0.2.0。
- 12 个新模型包——核对一致：Z-Image、Qwen-Image-Edit-2511、Boogu-Image、LTX-2/LTX-2.3、WAN 2.2 V2V、MiniMax-H3、SenseNova-U1.5、Janus-Pro、Qwen3.5、Qwen3-MoE（VeOmni EP）、Qwen3-Omni Thinker、Cosmos3。计数口径：LTX-2/2.3 计 1 个包（`ltx2/` 两行）、WAN 2.2 V2V 计 1 个包（`wan22_v2v/`），合计恰 12 个包。易混点：BAGEL、HunyuanImage 3.0 在 release 中是“plus new modes”，不计入 12；v0.1.0 清单已含 BAGEL、Hi3、SD3、Qwen-Image、FLUX.2-Klein、WAN 2.1/2.2、HunyuanVideo 1.0/1.5、Qwen3、Qwen2.5-VL、PE。
- SFT / agentic / async——核对一致：SFT：`train_sft`，CE + flow-matching，走共享 TrainStack；含 agent 对话、交错图文轨迹、Qwen3-Omni 音频、WAN 2.1 全 transformer SFT（release #209/#234/#352/#345/#320）。Agentic：`train_agentic`，多轮 `Sample` lineage，`AgenticTrainer` 每轮 rollout 前同步权重、同级轨迹并发、等完整 GRPO 组再评分训练（#214/#315/#363）。Async：`train_async_ar` + `train_async_diffusion`，staleness budgeting、动态 prompt 分发、评分前 rollout refill（#68/#192/#310/#289/#213）。
- SGLang 0.5.19 + vLLM-Omni 0.28——核对一致：Release Upgrade Notes：SGLang `0.5.12.post1`→`0.5.19`；vLLM/vLLM-Omni `0.20.0`→`0.28.0`；torch `2.11.0`→`2.13.0`（`+cu130`）；transformers `>=4.41`→`>=5.12,<5.13`；diffusers `>=0.30`→`>=0.38`；SGLang wheel 需 glibc ≥ 2.34。`pyproject.toml` 实测：`sglang[diffusion]==0.5.19`、`vllm==0.28.0`、`vllm-omni==0.28.0`、`torch==2.13.0+cu130`、`transformers>=5.12,<5.13`、`diffusers>=0.38.0`，全部吻合。`INSTALL.md` 亦称 torch `2.13.0+cu130` / CUDA 13 / glibc ≥ 2.34。
- 算法新增（v0.2.0 相对 v0.1.0）：CPPO（#58）、AR DPPO（#85）、GSPO（#178）、AR PPO+GAE+value head（#254/#256/#259）、DiffusionOPD（#232/#432）、DiffusionNFT 参考 KL（#411）、Flow-DPPO beta KL（#112）、WindowScheduler `decay`/`exp_decay`（#489）、ReFL/DRaFT-K 移入 `experimental/`。注意：CPPO 论文/教程虽标 2026-06，但 v0.1.0 算法清单无 CPPO，CPPO 是 v0.2.0 才入；Flow-DPPO/DRPO 在 v0.1.0 已有。
- 引擎/同步新增：SGLang Diffusion v2 + vLLM-Omni v2 替换 v1；canonical SGLang LLM/VLM 引擎替换 `sglang_llm`；新增 direct vLLM TP 引擎（vLLM-native IPC 权重同步，#480）与 FastVideo 引擎（WAN 2.1/2.2，#149/#415）；SGLang TP/EP 全权重 + LoRA 同步、checkpoint-engine IPC 同步。`fastvideo` extra 已移除（#573，PyPI 拒 Git 依赖 + 与 transformers≥5.12 冲突），改指 `$FASTVIDEO_PATH` 到 pinned checkout。

## 9 Issues / PRs 与标签

来源：MCP `list_issues` / `list_pull_requests` / `search_issues` / `search_code` / `get_file_contents` / `list_commits`，快照 2026-10-14 约 12:38 UTC。注意：MCP 的 `list_pull_requests` 对所有 closed PR 都返回 `merged=false`，以 `merged_at != null` 作为“已合并”代理指标，并用 `list_commits` corroborated。

### 9.1 数量与标签体系

- Open issues 53 个；Closed issues `totalCount` 64。
- Open PRs 69 个；69 + 53 = 122，与 `open_issues_count` 一致。
- 实际观测到的已应用标签（跨 open/closed issues + open/closed PRs 去重，仅 8 个）：`bug`、`enhancement`、`feature`、`documentation`、`help wanted`、`wip`、`need review`、`approved`。工作流里还出现但本批未观测到应用实例的标签：`changes requested`（`pr-status-labels.yml`）、`format`（`auto-format.yml`，打上触发自动格式化并推回）。MCP 只读工具无 `list_labels`，以上不是官方全量标签表。
- Open issues 标签分布（53 个中）：`enhancement` 18、`help wanted` 4、`bug` 3、`feature` 3、`documentation` 1、`wip` 1（与 #453 重叠）；其余 23 个无标签。
- Open PRs 标签：`need review` 50、`wip` 15、`approved` 3、同时 `wip`+`need review` 1；Draft 15 个。年龄（以 2026-10-14 12:38 UTC 计）：>30 天 24 个，>14 天 44 个，近 7 天 9 个。最老：#218（89 天，BAGEL vLLM-Omni T2TI）、#257（80 天，config recipe contracts）、#301/#300/#298（72 天）。
- Open PR 作者集中：@ruiling-smartbear 14、@CjhHa1 11、@leviking98z-rgb 7、@nussejzz 5、@cboss6 4。仅 3 个 open PR 带 `approved`：#372、#362、#311。

### 9.2 Open issues 抽样（按创建时间倒序，前 60 实际 53 个全量）

#591 [enhancement] [RFC] One rollout sample-observation path；#590 [bug] Qwen3-Omni GSPO: first-update ratio != 1 with old_logp_source=replay；#588 [Benchmark] MiniMax-H3 T2VA: UniRL vs veRL-Omni on 8xH20；#587 [enhancement] [RFC] Validate full-parameter RL training for MiniMax-H3 T2VA；#581 [bug] FSDP2 collective mismatch when only some DP ranks skip a 0-token AR micro-batch（见 §10）；#580 [enhancement] [RFC] Single-owner loss normalization；#570 [RFC] Which multimodal rollout/training extension should UniRL prioritize?；#567 cosmos3: H20 cuBLAS workarounds are gated to torch 2.10/cu128 and never run on pinned torch 2.13/cu130；#564 Monitor full-vocabulary AR policy entropy without regularization；#561 Add bounded dynamic sampling for synchronous AR GRPO；#556 [RFC] Opt-in token-level rollout correction for replay-anchored AR GRPO；#551 [enhancement] [Feature] TempFlow-GRPO: per-step branching with noise-aware weights；#537 [enhancement] [RFC] Mixed-resolution rollouts, long-tail scheduling, sibling batching, train/inference consistency；#529 [RFC] Sibling group-affinity routing for AR rollout；#525 [enhancement] refactor(vllm-omni/hi3): reuse FusedMultimodalCondition.concat for DiT capture padding（已由 PR #569 Fixes）；#523 train(unified): UnifiedModelTrainStack skips eval-mode anchor freeze and prepare_part；#522 train/stack/planner: underscore helpers are imported and re-exported；#520 [bug] fix(hi3): batched chat-template tokenization encodes only the first sample；#516 [enhancement] [Feature] MixGRPO-Flash；#515 [enhancement] Frozen teacher resolution can fail on one rank and leave others hanging；#512 [enhancement] Re-evaluate hard-coded foreach=False in default AdamW；#507 [enhancement] [vLLM-Omni] AR rollout ignores ARSamplingParams.seed；#488 [RFC] Complete decay schedulers and explore TempFlow-GRPO / Flash-GRPO；#484 [RFC] Optional AISimulate integration；#483 [enhancement] [RFC] Trainside optimization roadmap；#481 RFC: rollout memory and context budget for agentic RL；#473 [Tracking] Audit and retire vLLM/vLLM-Omni patches for 0.28/0.29；#470 [enhancement] [RFC] Support DyRef and multi-image model inputs；#463 [enhancement, help wanted] [RFC] NVIDIA CUDA MPS for colocated reward workers；#458 [enhancement] Part.control: one untyped root bag, no stage ownership；#455 [enhancement] feat(async): expose coherent rollout-manager depth, lag, phase metrics；#453 [documentation, help wanted, wip] [RFC] Community refresh of main-repository documentation（21 评论，open issues 中评论最多）；#450 [RFC] Text Embedding Precomputation & Decoupled Loading for Minimax-H3；#443 [RFC] Unified text-encoder and conditioning cache for GRPO rollout；#435 [Roadmap] Qwen3-Omni post-training support；#433 [enhancement] [RFC] CI Development；#431 RFC: UniRL Experimental Plugin Framework for Train–Inference Parity；#423 [enhancement, help wanted] [RFC] Validate composite reward capabilities before eager model loading；#419 [help wanted] reward: no way to score a component without feeding it into the gradient (monitor-only rewards)；#416 OrcaRouter as an optional OpenAI-compatible provider；#414 [RFC] Unified benchmark result schema and CPU-safe CLI；#412 Reward collapse when using SGLang on Hunyuanvideo1.0+Pickscore；#399 [RFC] Ordered Ragged ImageSet / ImageSets Primitives；#391 [Roadmap] MiniMax-H3 critical path；#368 [help wanted] [RFC] Extend UniRL's on-policy distillation support (Welcome Contributing!)（18 评论）；#262 [Tracking] AR rollout prefix-cache exposure；#190 Should I support TP-aware weight sync for SGLang colocate rollout?；#170 [Question] SGLang rollout can't use TP/EP for large AR models (Qwen3.5)；#147 PPU Run Support for This Repo；#127 [feature] Video RL Roadmap；#93 [feature] Agentic RL Roadmap；#86 [Tracking] Add PPO value critic and GAE support；#25 [feature] UniRL Roadmap Q2 2026。

分布观察（仅陈述）：open issues 作者高度集中，CjhHa1 占 20/53；`help wanted` 仅 4 个（#453、#423、#419、#368）；明确写 “Welcome Contributing!” 的仅 #368。

## 10 硬核 issue 三件套核查（用户点名）

### ① Checkpoint LoRA：无对应 issue，找到的是 open PR #599

- Issue 编号：未找到。`search_issues` 用 “LoRA checkpoint bias resume export / LoRA bias / checkpoint resume LoRA” 仅命中已关闭且主题不同的 #327、#155、#27。Open issues 53 个标题中无 “LoRA bias / checkpoint LoRA” 条目。PR #599 正文无 `Related Issue` / `Fixes #` 段。
- PR 覆盖：PR #599 `fix(checkpoint): preserve trained LoRA biases on resume and export`，作者 @LingYi-Liang，创建 2026-10-14 07:58 UTC，状态 open，标签 `need review`，非 draft。
- PR 内容（正文摘要）：`bias=all/lora_only` 时 adapter checkpoint 静默丢 trained parameters；修复为 torch 与 DCP 两种格式都 save/restore，并在 PEFT 与 merged-model 导出中保留；旧的不完整 bias checkpoint 改为明确失败；`bias=none`、frozen-teacher exclusion、EMA adapter 行为保持。Test Plan 称 CPU/Gloo 验证通过，GPU/NCCL 与 rollout sync 未测。
- 修复状态：未修复（未合入 main）。
- 邻近已合并项（不同主题，勿混淆）：PR #575 `fix(lora): reject frozen teachers whose weights differ across ranks`（Related to #515，2026-10-13 合并）；PR #579 `fix(checkpoint): resume a partially initialized AdamW`（Fixes #499，2026-10-10 合并）。

### ② FSDP replay：Issue #581 open，PR #598 draft 覆盖，未合并

- Issue #581 `[bug] FSDP2 collective mismatch when only some DP ranks skip a 0-token AR micro-batch`，作者 @MinhaoLi0318，创建 2026-10-11，状态 OPEN，标签 `bug`，0 评论。正文机制：若干 AR 算法在 0-token micro-batch 时在 `compute_loss_and_backward` 中提前返回（列出 `grpo.py:115-116`、`cppo.py`、`dppo.py`、`drpo.py`、`ppo.py`、`gspo.py`、`sft.py` 等）；仅当所有 DP rank 同跳才安全，否则 FSDP2 的 all-gather / reduce-scatter 序列错位。附 2-rank gloo CPU 复现（单侧有 0-token 行时对端报 `gloo::EnforceNotMet`，3/3 次）。正文称 #580 §2 / phase 3 规划由 stack 决定空 micro 的处置，本 issue 为跟踪直至该 phase 落地；来源为 #566 review 的 follow-up。
- PR 覆盖：PR #598 `fix(train): keep empty AR rows in FSDP replay and backward`，作者 @ruiling-smartbear，创建 2026-10-13，状态 open + draft + 标签 `wip`，正文明确写 `Fixes #581`。方案（正文）：训练对齐前对空 response 行插入 token 0 且 `loss_mask=0`，保持 replay/backward 同步且不贡献 loss；GSPO/PPO 对全掩码 replay 输出也 backward。Test Plan 称 CPU/gloo 与 2×H100（Modal）验证通过，未跑完整 HI3/BAGEL 训练或多 rank CUDA PPO。
- 修复状态：未修复（未合入 main）。

### ③ RoPE FP32：Issue #595 已关闭，PR #596 已合并，已修复

- Issue #595 `[Bug] meta-init leaves RoPE inv_freq in BF16 on Qwen3 / Qwen3-MoE recipes with a BF16 master (position-dependent logprob drift)`，作者 @MinhaoLi0318，创建 2026-10-13，状态 CLOSED，标签 `bug`。正文机制：`meta_init_transformer: true` + `model_precision: bf16` 时，RoPE 频率缓冲 `inv_freq` / `original_inv_freq` 落入 BF16（`finalize_meta_init` 的 blanket `transformer.to(dtype)`、以及 `restore_init_state` / `recover_rope_inv_freq` 以 `tgt.dtype` 回写），而非 meta 路径（`from_pretrained(dtype=bf16)`）与 vLLM 均保持 FP32；角度误差随 position 增大，导致 trainer logprobs 漂移。附 CPU 复现脚本思路与 affected recipes。标题说的是 BF16 vs FP32 的 inv_freq，与“RoPE FP32”指同一问题；未找到其他 open 的 “RoPE FP32” issue。
- PR 覆盖：PR #596 `fix(meta-init): preserve non-persistent buffer dtypes across meta-init and VeOmni wrap`，作者同为 @MinhaoLi0318，创建 2026-10-13，正文明确写 `Fixes #595`，merged_at = closed_at = 2026-10-14 12:11:49 UTC，即已合并；对应 main 提交 `d8a82c60`。修复要点（PR 正文）：`build_meta_init_transformer` 在 `torch.set_default_dtype(dtype)` 下实例化；`finalize_meta_init` / `veomni_parallelize` 仅 cast 参数与 persistent buffer；`restore_init_state` 按捕获值的自身 dtype 恢复；删除 `recover_rope_inv_freq`；在 `unirl/models/README.md` 的 Gotchas 记录该不变量。验证称 #595 复现脚本在 FSDP/VeOmni 路径下 `inv_freq` 保持 FP32 且 logprobs bit-identical。
- 修复状态：已修复（已合入 main）。
- 邻近 RoPE 历史（勿混淆）：PR #207 `refactor(meta-init): unify the meta-init build/restore path (fixes garbage RoPE on VeOmni / sharded-FSDP)`（已合并，garbage RoPE，非 dtype 问题）；PR #89 BAGEL 的 fp32 RoPE + RMSNorm；PR #600 删除了 BAGEL 的 `_install_rope_fp32`（称 vLLM-Omni 已改用 MoTRMSNorm，与 #595 无关）。

结论：三件套中，RoPE 已修复；Checkpoint LoRA 与 FSDP replay 均已有 open PR 覆盖，均不构成新的空白点。开任何 PR 前必须再检索 overlapping open PRs。

## 11 贡献规范与风险

### 11.1 必须遵守的文件（均在上游 main，已验证）

- `AGENTS.md`（约 119 行）：§2 Simplicity First：最小改动，不做未要求的抽象/灵活性/错误处理。§3 Surgical Changes：一行 docstring 上限（`check-docstring-lines` hook 强制，无 allowlist）；真正的约束/数值/bit-identity 要求写进最近的 `README.md` 的 `## Gotchas`，不写多句注释块；只清自己改动产生的孤儿代码。§5 Contribution Hygiene：开 PR 前查相关 issue、open PR 与关键词防重复；已有 open PR 覆盖同一修复时不开重复 PR；不提低价值 busywork（单 typo、孤立 style churn、机械清理）；AI 辅助需在 PR 中声明、附重复工作检查与测试命令+结果；`tests/` 树已按政策删除（#99/#267），不得以任何名字重建；一次性验证 harness 只跑、不提交，命令+结果写进 Test Plan。§6：本地 skills：model bundle（`.claude/skills/development/add-model-bundle/SKILL.md`）、PR（`pr-workflow/SKILL.md`）、PR review（`pr-review/SKILL.md`）、README（`readme-guide/SKILL.md`）。§7 Pre-PR Review Gate：实质性代码/配置变更在 `git push` / `gh pr create` 前必须完成 pr-review skill，且该 commit 的记录为 `Status: pass` 才可 push。
- `.claude/skills/development/pr-workflow/SKILL.md`：开 PR 前读 PR 模板 + `pr-body.yml` + `semantic-pull-request.yml`；审全部分支 diff（非仅最新 commit）；标题用 Semantic PR 允许的类型；Body 必须含真实内容的 `## Summary` 与 `## Test Plan`（否则 `PR Body` 检查失败）；提及兼容性风险、AI 辅助、重复工作检查。
- `.claude/skills/development/pr-review/SKILL.md`：pre-push review 程序——全分支 diff 审查、作者自查（含 `getattr` / `except` / `.get` / `deepcopy` / `.item()` / `empty_cache` / `print/TODO` 等 smell scan）、独立 review 须由不同模型/不同上下文的子代理执行、验证（`SKIP=no-commit-to-branch pre-commit run --from-ref $base --to-ref HEAD`；需 GPU/多 rank 的检查未跑则必须写 `Not run; reason: ...`，不得谎报通过）、审查记录写在 `.git/unirl-review/<HEAD>.md`，首行 `Status: pass` 仅当无 open 的 P0/P1/R finding。
- `.github/pull_request_template.md`：Sections 为 Summary / Related Issue（`Fixes #123` 或 `N/A`）/ Test Plan（精确命令或 `Not run; reason: ...`；训练/rollout 变更需含模型、recipe、硬件、GPU 数、数据集/checkpoint）/ Compatibility / Risk / Reviewer Notes / Checklist（已审 diff、已更新测试/文档/配置或说明原因）。
- `.pre-commit-config.yaml`：hooks 含 pre-commit-hooks（yaml/toml/ast/大文件/私钥/尾空格等）、ruff check + format、以及 3 个本地 guard：`check-recipe-targets`、`check-experimental-boundaries`、`check-docstring-lines`（另有 `check-core-dependencies` 在 PR 正文提及）。CI 的 lint 工作流即跑 `SKIP=no-commit-to-branch pre-commit run --all-files`。
- 安装（`INSTALL.md`）：uv + `pyproject.toml`；Python `>=3.12,<3.14`、Linux x86_64、NVIDIA GPU；每 venv 只选一个 engine extra（`vllm` 与 `sglang` 在 `[tool.uv] conflicts` 中互斥）；`train` / `infer` 不拉 rollout 引擎；`dev` = pytest + ruff + pre-commit；`dataset-prep` 独立于 engine extras。`pyproject.toml` 关键 pin：`torch==2.13.0+cu130`（两引擎栈同款）、`sglang[diffusion]==0.5.19`、`vllm==0.28.0` + `vllm-omni==0.28.0`、`transformers>=5.12,<5.13`、`diffusers==0.40.0`（uv override）、`ray[default]>=2.9`、`hydra-core`、`peft>=0.20`；ruff `line-length = 120`，`unirl-reward-service` 与三处 vendor 目录排除在 ruff 之外。

### 11.2 CI（`.github/workflows/`，8 个文件，均已读关键内容）

- `lint.yml`：push 到 main + PR 到 main 时跑 pre-commit 全量。无 pytest / 无 GPU 测试 CI（`.pre-commit-config.yaml` 注释亦称 “this repo's CI is lint-only”）。
- `pr-body.yml`：PR 非 draft 时强制 `## Summary` 与 `## Test Plan` 含可见非注释文本。
- `semantic-pull-request.yml`：标题类型仅允许 `fix / feat / docs / test / refactor / build / ci / chore / perf / revert`。
- `pr-status-labels.yml` + `pr-review-event.yml` + `pr-review-status-labels.yml`：自动同步 `wip / need review / changes requested / approved` 标签。
- `auto-format.yml`：仅当 PR 被打 `format` 标签时跑 pre-commit 并推回格式化提交。
- `publish.yml`：手动触发（`workflow_dispatch`）才发布 PyPI，不会自动触发。

### 11.3 DCO / CLA（未找到证据，需区分“未找到”与“不存在”）

- 无 CONTRIBUTING 文件：`CONTRIBUTING.md` / `CONTRIBUTING.rst` 在上游均不存在。
- 无 CLA 文件：`CLA.md` 不存在；`search_code` 对 `CLA` 的命中均为误报，无 CLA bot / CLA 流程文件。
- 无 DCO 证据：`search_code` 对 `Signed-off-by` 返回 0 命中；工作流中无 DCO 检查；`list_commits` 显示合并提交的 committer 为 `web-flow`（GitHub 网页合并），未见 Signed-off-by 要求。
- 结论标注：以上为“在仓库文件与工作流中未找到 DCO/CLA 要求”，不是“组织层面一定没有 CLA”。Tencent 组织级 CLA（如有）无法从本仓库文件验证，需提交前在 PR 页确认有无 CLA bot 评论。

### 11.4 PRs 概览与外部贡献者合并速度

- Closed PRs（MCP 返回的前 200 个，按时间倒序近似 2026-07-25 起）：第 1 页 100 个中 `merged_at != null` 90 个、未合并关闭 10 个；第 2 页 100 个中合并 76 个、未合并关闭 24 个。
- 外部 vs 内部划分是近似口径，需谨慎解读：MCP 无法可靠区分“Tencent 内部作者”与“外部作者”。用 CODEOWNERS 名单 + 高频作者作为 core-proxy，其余为 non-core；个别作者可能实为协作方而非纯外部。MCP 的 `comments` 字段对 PR 返回 null，故无法用 MCP 统计 review 评论数 / maintainer 回复延迟。
- Closed 第 1 页（约 2026-08-27~10-14）：core-proxy 合并 79 个，中位 2.8 天，均值 5.9 天，最长 38.3 天；non-core 合并 11 个，中位 10.0 天，均值 10.7 天，最长 30.8 天，未合并关闭 0 个。Non-core 合并实例：#536/#530、#479、#477、#448、#432、#409、#406、#405/#402/#401。
- Closed 第 2 页（约 2026-07-25~08-24）：core-proxy 合并 70 个，中位 1.5 天；non-core 合并 6 个，中位 0.9 天（样本小，含 #346、#276、#272、#338/#335/#332）；non-core 未合并关闭 6 个（@morluto #340/#336/#334/#331、@NancyFyong #271/#270，均为 2026-07-28~08-05 的同期 PR 批，未验证关闭原因）。
- 综合表述（仅描述样本，不推断制度）：最近一页 non-core 合并中位约 10 天、慢于 core-proxy 的 2.8 天；但更早一页 non-core 合并中位仅 0.9 天，样本小且作者集中，不能得出“外部更慢”的稳定结论。另有 69 个 open PR（含 24 个超 30 天）表明 review 积压真实存在，积压主体是 core 作者的 PR。
- Maintainer 互动（可验证的间接证据）：`.github/CODEOWNERS` 明确：fallback `*` 为 @CjhHa1 @leviking98z-rgb @celve；按目录细分 owner，且每条规则至少两名 owner。PR 状态由工作流自动打标签。PR 正文中有 maintainer 跟进提交的记载：如 PR #585 正文列出 maintainer 的 4 个 follow-up commits 及两轮独立 review。未验证：maintainer 在 PR 下的实际评论内容、review 轮次、request-changes 比例。

### 11.5 候选观察（仅列观察，不下“值得提”结论，不认领）

1. #368 `[help wanted] Extend UniRL's on-policy distillation support (Welcome Contributing!)`（2026-08-13，18 评论）：仓库中唯一明确写 “Welcome Contributing!” 的 open issue，且为 RFC 形态。任何动作前需先读其 18 条评论确认是否已有进行中的实现（MCP 未返回评论正文，未验证）。
2. #453 `[documentation, help wanted, wip] Community refresh of main-repository documentation`（2026-09-14，21 评论）：open issues 中评论最多，但 AGENTS.md §5 明确反对“孤立 style churn / 机械清理”类 PR，文档类工作需先确认 #453 的分工与接受标准（未验证评论内容）。
3. #419 `[help wanted] monitor-only rewards` 与 #423 `[enhancement, help wanted] Validate composite reward capabilities before eager model loading`：仅有的另外两个 `help wanted` open issues，均为 RFC/能力缺口描述，无现成实现路径（未验证细节）。
4. #525 / PR #569：#525 由 PR #569 覆盖（正文写 Fixes #525），但 #525 在 open issues 列表中仍出现——可能为 MCP 快照时序或 PR 未合并所致；若考虑邻近工作，需先核对 #569 的实时状态（未验证，MCP 中 #569 仍 open、`need review`）。
5. 积压与规范成本：69 个 open PR + 24 个超 30 天 + 仅 lint CI（无测试 CI）+ 强制的 pr-review 独立审查记录，意味着外部 PR 需自带强验证证据（CPU/GPU harness 结果、`Not run` 诚实标注）才易被接受；同时 AGENTS.md 的重复工作检查要求开 PR 前必须检索 overlapping open PRs。

## 12 与用户方向的对照

用户方向：WWW 2027 投稿《Learning to Survive the Beam: Boundary-Aligned Training for Semantic-ID Generative Recommendation》，核心是边界对齐训练；更广的兴趣是多模态 RL、能 scaling 与泛化、能真正落地。

UniRL 对口的地方：

- 统一模型（AR + diffusion 双算法、共享 optimizer 步）与 Prompt-Enhancer（AR 改写 + 扩散奖励）是最接近“生成式推荐/多模态生成 + RL 对齐”的两条线：`UnifiedModelTrainer` 的双算法反向累积、PE 的 composed rollout，都是现成的组合范式。
- 三自研算法的信任域思想与“边界对齐”有概念亲缘：Flow-DPPO 的 KL mask、DRPO 的光滑二次正则、CPPO 的位置/累积预算，都是在“推多远才安全”上做文章。区别是 UniRL 的边界是策略与 behavior 策略之间的散度边界，用户论文的边界是 beam 搜索中的语义 ID 生存边界——类比可用，公式不可直接搬。
- Agentic（多轮工具使用、组归一化 advantage 应用于全部 assistant turn）与 Async（staleness budgeting）是工业落地侧最有参考价值的两块：前者是长程任务 RL 的标准做法，后者是吞吐与新鲜度的权衡。

不对口 / 需警惕的地方：

- UniRL 的重心是图像/视频生成 RL（扩散侧模型占矩阵大半），文本生成式推荐不是其目标场景；不要把扩散侧的 SDE/KL 结论直接套到离散 token 的推荐序列上。
- 框架极重：Ray + FSDP + 多 rollout 引擎 + 奖励服务，本地复现成本高。只读学习时以算法与契约为主，不要试图全量跑通。
- Fork 落差大：任何基于旧 fork 的判断都需以上游状态为准。未来贡献分支必须从上游 `main` 建，个人笔记绝不进上游 PR diff。

## 13 未验证清单

- 三篇自研论文（Flow-DPPO / DRPO / CPPO）的实验结论；任一 recipe 的端到端可跑性（矩阵自述“包 + recipe 才算 runnable”，本次未执行）；v0.2.0 tag 的精确 UTC 时间戳（发布页仅显示 “08 Oct 08:43”，未断言秒级时间戳）。
- 未逐行读算法、trainer、distributed 的实现文件；§5 数据流来自各模块 README 与 entrypoint 源码，未做运行时/GPU 验证。
- `benchmarks/`、`experimental/`、`datasets/` 下各子目录、`CPPO/`/`DRPO/`/`FlowDPPO/` 教程正文、`examples/` recipe 总数、`.gitmodules` 内容未逐一核查。
- `rollout/engine/vllm/` 在目录中存在，但 `rollout/README.md` 六引擎枚举未单列它，其地位未验证。
- 全量标签表（MCP 无 `list_labels`）；PR 下的 maintainer 评论正文、review 轮次、request-changes 比例（MCP PR 的 `comments` 字段返回 null）；#368 / #453 评论区内是否已有进行中的实现；组织级 CLA 是否存在；外部 vs 内部作者的精确划分（§11.4 的 core-proxy 为近似口径）。
- Fork 状态：fork main `80b24995`（2026-07-14）已由主 agent 直接验证；本报告全部取自上游 `main`。

## 14 下一步（未授权，仅记录）

- 若要找贡献点，先读 #368 的 18 条评论确认是否已有进行中的实现，再决定是否碰 on-policy distillation。
- 若要同步 fork，用浏览器 discard 方式把 `main` 同步到上游 HEAD（个人笔记在 `notes/learning` 分支，与 `main` 分离，不会被 discard 影响；操作前仍需确认）。
- 不碰：#599（Checkpoint LoRA）、#598（FSDP replay）、#596 已合并的 RoPE——均已有覆盖/已修复。

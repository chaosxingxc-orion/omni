# LoRA 更新后，如何定位训推概率偏差

当 rollout 使用 vLLM，训练侧用 HF 对实际采出的动作重新评分时，若同一策略的两端概率已经不同，RL 的概率比就混入了工程偏差。先在发生错误的 checkpoint 和请求上解释第一个不同的运算，再恢复训练。本次基线是 Qwen3-Omni-30B-A3B-Instruct、thinking OFF、NPU BF16，同机两套 TP4 推理与 FSDP8 训练分时执行。

## 这次为什么冷请求通过，热请求却失败？

原失败请求有 5,531 个提示 token，生成 11 个动作。最大偏差发生在第 6 个动作：单 token logprob 差为 0.042679，整段差为 0.041136，两者门槛都是 0.01。独立实例的冷请求通过。用另一段录音预热，证实有 2,176 个前缀 token 命中后，同机复现了偏差。缓存改变了剩余 token 在计算批次中的位置，这提供了区分“输入不同”和“相同输入算出不同结果”的实验条件。

| 待证假设 | 如何隔离变量 | 实测及判断 |
|---|---|---|
| O 与 QKV 的 LoRA 分片装反 | 从实际 worker 读回 A、B，按各自并行方向逐 rank 对照 checkpoint，并核对 O 汇聚后的通道顺序。 | Q/K/V 复制 A、沿输出切 B。O 沿输入切 A、复制 B。两套实例、48 层共 3,072 个分片全部匹配。保留原布局。[分片与算子回放](p-rollout-o-row-replay.json) |
| 偏差来自 O 基座或上游输入 | 按绝对 token 位置对齐冷、热公共后缀，分开记录完整输入、基座、LoRA A、LoRA B 和合并结果。 | 第 30 层（从 0 计数）的 3,355 行输入、权重及基座输出相同。LoRA A 的 26,840 个 BF16 输出中有 9 个不同。固定同一输入矩阵，只循环移动行再还原，原生 linear 有 3 个元素改变。原位重复为 0。分组乘法在实测冷热窗口得到相同结果。证据支持批内行位置影响舍入，未确认芯片内部具体选核机制。 |
| 修好 O 就能消除最终误差 | 只替换 48 层 O 的 LoRA A，重跑同一热请求。随后检查首个仍有差异的分支。 | 第 30 层 O 已测边界恢复一致，整段差仍为 0.041136。第 31 层入口及 Q/K/V 基座一致，四个 rank 的 Q/K/V LoRA A 都出现差异。最终 Q 只在 rank 3 不同，随后经 O 汇聚传播。[只修 O 的反例](p-rollout-o-grouped-control.json) · [分支证据](p-rollout-lora-branch-localization.json) |

修复把训练和推理两侧 48 层的 Q/K/V/O **LoRA A** 统一为分组 BF16 矩阵乘。HF 保留 FP32 主参数和 BF16 autocast 边界。在实际使用的两个输入宽度上，输入梯度与主参数梯度均与原生反向对照逐值相等。热请求修复前后采出**完全相同的 11 个动作**，两项概率差均降为 0。冷、热请求彼此可以采出不同动作。

## 下次遇到类似故障，先查什么？

1. **先确认两边在给同一个动作打分。** 保留原 HTTP 请求与响应、checkpoint 和实际 adapter 摘要、采样参数及概率定义。对齐音频预处理、展开后的 prompt token、mask、位置和采样动作前缀。核对预测位置是否偏一位。temperature 决定采样分布，验收比较的是给定同一历史的实际动作概率。身份、动作或口径不一致时，先修数据或接线。
2. **按触发条件分流。** 仅更新态失败，先查导出、加载后的 A/B、缩放、dtype 和切换 adapter 后的缓存失效回执。仅热路径失败，保留 KV/prefix cache，以同一请求做冷缓存、同录音重复、跨录音部分命中三种对照，读取实际命中数。仅跨机失败，先做同机同权重对照。只有反向或参数更新不一致时，才优先比较梯度累加/归约、裁剪和优化器状态。
3. **让程序找输入相同而输出首次不同的边界。** 先按计算依赖看层入口、残差和归一化，再拆投影的基座、A、B、合并。按绝对位置与 TP 通道重组对齐，记录分块长度、实际操作数和输出。某个 rank 的差异可在汇聚后出现在全部 rank。若输入已不同，沿依赖向前找。若融合内部不可见，记为观测缺口。第一个被探针看到的位置不一定是全模型首差。
4. **用最小反例证明原因，再扩展同类修复。** 先原位重复，再一次只改行位置、批形状或算术实现。插入探针后要核对动作及概率仍复现原问题。某层通过后重跑原失败请求。整模仍超限就重新定位，已有证据一致的层不必逐层重做。
5. **分别签收数值校准和训练更新。** 令每个实际动作的差为 Δₜ = log p_HF(aₜ | hₜ) − log p_rollout(aₜ | hₜ)。每个请求在 HF train 和 eval 均须满足 maxₜ|Δₜ| ≤ 0.01、|ΣₜΔₜ| ≤ 0.01，且全词表 top‑1 无翻转。首个失败请求、所在批次和历史成功样本都要过。下一次真实更新另看新策略采样、各 rank 的 optimizer 回执、Clock 及新 checkpoint，诊断动作不进入训练集。

## 合作团队如何复用

采用已验收的[工程发布 142943af9](https://github.com/chaosxingxc-orion/omni/tree/142943af9)，保留其[训练和推理源码清单](p-rollout-source-lock.json)。双方的 attention 文件按各自部署角色核对，两端的 LoRA A 算子则须匹配同一份源码。相同镜像名称不能证明挂载的源码相同。

1. 从 [profile 示例](../../../tools/system_rl/runtime_repairs/qkvo-bf16.profile.json) 把 `values.LORA_ROW_GEOMETRY` 合并到本地部署 profile。训练用[续训模板](../../../tools/system_rl/runtime_repairs/BOUNDED-CONTINUATION.template.json)读取它。推理准备配置的 `lora_row_geometry` 使用同一个对象。该配置覆盖第 0–47 层，适配器秩为 8、两个输入宽度为 2048 和 4096。更换秩、宽度、设备算子或并行拓扑时重新验证相应边界。模型、镜像、checkpoint、受控请求材料与本机部署路径由接入团队提供。
2. 在仓库根运行下列命令。第一条只在 CPU 上比较已准备的两端源码。第二条复核实际回放与 HF 的逐 rank 结果。它消费已经封闭的材料，不派发推理。生成材料的步骤见[推理实例与回放说明](../../../tools/system_rl/instruct_activation/README.md)。本地 `RUNTIME_ROOTS` 是 JSON 文件，包含 `training` 和 `rollout` 两个部署目录。`CHECKPOINT_SHA` 为被恢复 checkpoint 的 `COMMITTED.json` 摘要。

   ```bash
   python -B tools/system_rl/runtime_repairs/verify_runtime_sources.py \
     --repo . --lock docs/operations/partner-handoff/p-rollout-source-lock.json --roots "$RUNTIME_ROOTS"
   python -B tools/system_rl/runtime_repairs/review_lora_candidate.py \
     --packet "$PAIRED_JSON" --config "$HF_CONFIG" --hf "$HF_RESULTS" --checkpoint-sha "$CHECKPOINT_SHA"
   ```

   检查第一份报告的 `source_matches` 和第二份报告的 `all_passed` 均为 `true`，并核对第二份报告的请求数与动作数。

3. 正式续训指向包含锁清单的完整发布提交 `142943af9db15c23693d12cab676a3cf450c0476`。准备阶段核验源码，运行时再核对两路实际 worker 的算法、层范围及源码摘要。源码一致、预检通过、数值验收通过和优化器更新成功各有独立回执。

本次验收覆盖 16 次请求、191 个动作，其中定点冷热对照为 2 次请求、23 个动作，原失败批次为 8 次请求、98 个动作，历史回归为 6 次请求、70 个动作。[逐请求验收记录](p-rollout-qkvo-regression.json)。三组在两种 HF 模式下的两项概率差和 top‑1 翻转均为 0。该证据对应严格 step 1。不能外推到后续权重、跨机部署或 DEV 收益。同一原批 8 请求的平均 HTTP 耗时从 82.80 秒变为 86.58 秒（+4.6%）。每种配置仅一次且带观测，生产并发吞吐需单独测量。

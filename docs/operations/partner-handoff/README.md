# Omni System RL 合作交接 · P 节点 LoRA 训推校准

新增：[LoRA 更新态训推偏差排查与复用指南](lora-parity-playbook.md)。包含原失败请求的根因、排查顺序、两端配置及验收命令。

两套 TP4 已在独立推理主机部署并启用 KV 和 prefix cache，每个 rank 读回 384 个严格首步 LoRA 张量。两实例对同一条真实录音各完成一次请求，共 20 个实际动作。训练主机从同一 checkpoint 恢复，8/8 rank 正常结束；训练和评估前向的最大单 token 和整段 logprob 差均为 0，top-1 无翻转。两条诊断耗时 105.58 / 107.66 秒，含适配器装载和全词表观测。训练仍为 1/3，尚未测量常规 rollout 吞吐。 [跨机复核回执](cross-host-active-parity.json)。

严格确定性 P 首步的同一份八卡 checkpoint 已完成同机更新态校准。四卡/八卡的 384 个 LoRA 张量、梯度及 Adam 更新逐值一致。两套 TP4 在旧版请求排空后切到新权重，清除旧缓存；两条录音共 6 个新权重请求、70 个动作，HF train/eval 与 vLLM 的单 token 差、整段差和 argmax 翻转均为 0。本次未执行连续训练或 DEV 评测。 [验收回执](updated-lora-parity.json)。 [配置与运行入口](../../../tools/system_rl/instruct_activation/README.md)。

以下为此前的实验记录。

从 `~/workspace/omni` 接手。内部路径中的 `jev-system-rl` 是历史目录名。P、S1、S2、D 的业务定义、实际训练程序和验收记录都在 Omni。合作方只需 Omni 源码与本项目的受控运行材料。
 
**当时的接手点是八卡 step 1 的 checkpoint。** 八卡从冻结模型独立完成一次 P 更新，并用新进程恢复了 FP32 LoRA、Adam、scheduler、Clock 和 RNG。四卡首步用于独立复核，不能与八卡首步累计。原跨机推理服务已释放；同机训练与推理按原 guard 分时使用。

| 核验对象 | 已观察到的结果 | 接手时如何使用 |
|---|---|---|
| 更新前概率和首步 loss | 同一批 2 条录音、8 条轨迹、86 个动作。四卡与八卡各自的同动作 logprob 与冻结行为一致，全局 loss 都是 0.6874986887 | 复用已封闭的样本及动作掩码 |
| 梯度和发布权重 | 四卡与八卡的梯度范数分别为 8.51779747、8.51530647。两份 BF16 发布文件的 LoRA A 全部 3,932,160 个元素相同，B 的 2,752,512 个元素中有 33,517 个不同，最大绝对差 0.00002002716 | 逐张量比较同一批次的更新前参数、归约后梯度、Adam 矩及实际更新量 |
| 恢复 | 八卡新进程 8/8 rank 恢复 step 1，checkpoint 清单 SHA-256 为 `7c2951657f448e95625791fc4aad0eca5c591d762498f91d2d9837e4e826a71e` | 此为原谱系。四卡包仅含 BF16 发布权重 |
| 更新后整模 | 释放前两套 TP4 均读回 384 个张量并使旧缓存失效。同一条真实录音产生相同的 10 个动作。原始最大单 token / 整段 logprob 差为 0.102120 / 0.156538，第 0 层 Q/O 修复后为 0.108583 / 0.079442，均有零次 top-1 翻转 | 两项门槛均为 ≤0.01，且全词表 argmax 无翻转。这是旧配置的失败回执；同机修复结果见下文 |

在同一主机，推理对 48 层 Q/O 基座投影使用已验证的分组 GEMM，保留非零 LoRA 的原有算术。两个独立 TP4 实例各处理一次原失败录音并自然生成 10 个动作，随后恢复原 FSDP8 checkpoint 做同动作评分。训练及评估前向的最大单 token 差、整段差均为 0，全词表 argmax 无翻转，8/8 rank 通过。分母是 **1 条录音、2 个请求**；此次只覆盖一个长度；不同长度与版本切换见页首。

在独立的四卡、八卡首步复验中，按 rank 重组成全部 384 个 LoRA 张量。两边初始 FP32 参数相同。LoRA B 初始为零，因此首步 LoRA A 梯度为零。B 的裁剪前梯度有 2,425,128 / 2,752,512 个元素不同，以四卡为参考，相对 L2 差 0.895%；最大绝对差 0.007019，在第 3 层 K 投影。FP32 实际更新量的相对 L2 差为 9.64%。两边各自按真实梯度范数、clip=1、AdamW 一二阶矩复算，参数最大绝对误差分别为 3.08e-12、3.14e-12 以内。差异已出现在 Adam 之前；当前还不能区分单条反向、累加顺序和跨卡归约。默认配置本身不足以归因到单个算子。该复验没有推进正式 Clock。见[逐项数值回执](first-step-update-comparison.json)。

对同一批数据启用严格确定性计算后重新执行原训练：四卡和八卡的 384 个 LoRA 张量、共 6,684,672 个元素，初值、裁剪前梯度、Adam 两个矩、FP32 更新量及更新后参数均逐值一致，两份 BF16 发布权重的身份也相同。原始配置的差异与反向执行模式相关；本次是全模型切换，未单独证明某个算子为唯一原因。严格模式产生独立首步；更新态整模复验用的是原八卡首步发布件，两个结果不能合并成同一条连续训练谱系。

第 0、1 层均以相同输入和权重重放，定位到 Q 基座投影在 560 行提示分块上的 BF16 GEMM 差异。第 0 层 O 基座也复现同类现象。HF 读取 FP32 master 或实际 BF16 导出得到相同动作概率。扩展验收时，沿用已通过的 48 层 Q/O 配置，在真实音频、有 KV/prefix cache、自然解码的服务上按新候选实际动作复算。已有 adapter 切到下一版仍需独立记录卸载、加载、读回与缓存失效。

## 在 CPU 复核历史四卡首步

本包包含交接说明、只读脚本和三份回执：[首步训练](p-first-fsdp4-loss.json)、[首批数据](bounded-p-train-handoff.json)、[冻结策略对齐](configured-actor-parity.json)。该四卡实验的 Omni 源码基线是提交 `aa4e202015a9b3cd95f3704ba7bb2e0641c3b252`。从项目方领取封闭运行副本 `first-loss-accepted/`；其中含本地配置及权重，单独受控交付。只复核结果无需镜像、模型或 NPU。

在交接包目录执行（将最后一个参数换成实际收到的目录）：

```bash
python3 verify_first_step.py --receipt p-first-fsdp4-loss.json --run-root /path/to/first-loss-accepted
```

预期打印 `PASS: ranks=4 step=1 rows=8 tokens=86 loss=0.68749869 grad_norm=8.51779747 adapter_bytes=13420384`。脚本核对训练配置、退出回执、四份 loss、四份结果及权重的 SHA-256，同时检查动作是否漏算或重复。所需结构：根目录的 `TRAIN.local.json`、`hot/EXIT.json`、各 `hot/rank-N/{LOSS,RESULT}.json`、`hot/step-1/lora_adapter/adapter_model.safetensors`。推理侧领取**整个 adapter 目录**，包括 `adapter_config.json`。缺文件补原件；摘要不符先核对版本，无需重跑模型。校验通过仅证明收到的是已验收的那次运行。

## 历史四卡首步的设置与结果

| 项目 | 实际设置与结果 |
|---|---|
| 模型与卡组 | Qwen3-Omni-30B-A3B-Instruct，thinking OFF。物理 0–3：vLLM TP4 推理；物理 4–7：HF/FSDP4 训练，容器内逻辑卡号 0–3。两端 BF16，LoRA 主参数 FP32。首步复用已封闭的 TP4 新动作；两侧同时满载的吞吐尚未测量。 |
| 训练人口 | 从 19 条合格 TRAIN 记录按固定顺序取 2 条原始录音，每条新采 4 次，共 8 条轨迹、86 个动作 token。gold 意图在前置召回列表中才准入训练。这个小批用于验证计算与更新。 |
| 策略与奖励 | 温度 1、top-p 1、不截 top-k。正确意图排第 1/2/3/未命中，分别得 1/0.75/0.5/0。两组奖励是 [1,0,1,1] 和 [1,1,1,1]：一条因额外的 Markdown 围栏被原解析器判为协议错误；另一组优势全零。两组都保留。更新前平均 reward 为 0.875。 |
| 优化器与损失 | AdamW，学习率 1e-5；LoRA rank 8、alpha 16、dropout 0，48 层文本注意力 q/k/v/o。先在组内计算优势，再对每条轨迹的有效动作贡献求和，最后除以 **8 条轨迹**。本步 KL 与辅助项均为 0。读 `RESULT.json → objective.components.total`：**0.6874986886978149**；四个 rank 的梯度范数均为 **8.517797470092773**。单 rank 的 `actor/pg_loss` 不是全局 loss。 |
| 对齐与导出 | 冻结条件：冷缓存 87/87、共享前缀 86/86，训练前向与 TP4 同动作 logprob 精确一致；本训练批次 86/86，最大单 token 和整段差均为 0。最终 4/4 rank 正常退出，导出 384 个 BF16 张量、6,684,672 个元素、13,420,384 字节。文件摘要与张量身份分别取训练回执的 `adapter_sha256` 和 `publication_identity`，二者用途不同。 |

首步的判据包括正常退出和导出成功。早期失败运行也有 loss 回执；不要混用其梯度或把独立重启累计成连续更新。该四卡首步只交付发布权重。八卡恢复及更新态结果见页首。

## 在哪里改、如何独立复现

Omni 公开仓从 `tools/system_rl/runtime_repairs/` 进入；内部 Omni 工作目录下从 `delivery/system-rl-transition-v1/jev-system-rl/rl-ready-window-configuration/` 进入。固定脚本分工如下：

| 入口 | 用途 |
|---|---|
| `publish_retained_cohort.py`、`pack_actor_fields.py` | 通过原消费者校验真实轨迹并封闭训练字段，关联请求、参考概率、Clock 和数据版本。已验收字段包可以直接复用。 |
| `stage_bounded_actor_step.py`、`launch_retained_hf_fsdp.py` | 从成功退出的 HF 参考运行准备独立首步，在已有 guard 下核对镜像、卡组、空间并派发。 |
| `bounded_actor_step.py`、`source/events.py` | 通过 Omni 训练入口调用 verl 的 `DataParallelPPOActor.update_policy`；单行去 padding 后送入模型，分数按原动作掩码映回，汇总全局 loss。 |
| 封闭源码包内 `parents/jev-system-rl/implementation/{entry,lora_training_state,publication,newprocess_replay}.py` | 接连续训练、checkpoint 与发布的工作入口。已有八卡假设须在四卡拓扑实测。这些入口尚未用当前产物完成续训验收。 |

**仅需复现首步时**，准备原参考 bundle、字段包和原镜像。训练镜像 ID 为 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`；TP4 推理镜像为 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`。校准实现绑定了源码与镜像的组合，不能仅替换同名镜像标签。合作环境用本地部署配置绑定现场地址，下面命令消费**已解析好的本地 JSON**：

```bash
# 从 Omni 公开仓根目录运行；TRAIN_STEP_CONFIG 指向团队自己的配置
python3 tools/system_rl/runtime_repairs/stage_bounded_actor_step.py --config "$TRAIN_STEP_CONFIG"
```

配置六项：`host` 是训练机 SSH 别名；`run` 是未使用的小写运行名；`source_bundle` 是**训练主机上**含 `HF-ACTOR.local.json`、`tree/` 和成功退出回执的冻结参考运行；`accepted_fields` 是**启动客户机上**的 `ACCEPTED-FIELDS.pt`（SHA-256：`e84a16fffeb2cd230117fb2a85da39117b8b4c2f56169ebf72a3d22f831146e0`）；`dataset_sha256` 取数据回执的 `pins.dataset`，对应 `datasetV2/COMPLETE.json` 的文件摘要；`devices` 为 `[4,5,6,7]`。配置中的 boot、原 guard、镜像、存储必须属于实际部署。热状态落本地持久盘，共享模型/数据/源码只读；由已验收的独立客户端归档。复现会写入新运行目录。此入口接受初始 Clock 和 policy-only 首步，不能循环调用来续训。

## 从首步到连续训练：并行工作与合流条件

1. **推理侧：验证更新后的权重真正生效。** 排空旧请求，记录旧策略身份、缓存失效回执和新 adapter 的实际载入身份；KV/prefix cache 保持启用。用真实语音采新动作，保存请求 ID、版本、token 和行为概率。HF 使用同一更新权重和动作复算。通过条件：最大单 token 差和整段 logprob 差各 ≤0.01，全词表 argmax 无翻转，且依据实际服务回执证明未复用旧策略缓存。
2. **训练侧复核八卡更新。** 在原 actor 的 optimizer step 前后记录本地 FP32 LoRA 分片、归约后的梯度和 Adam 两个矩，按参数名与分片位置还原。以同一初始权重、动作掩码和奖励比较四卡与八卡的差异。推理侧消费部署用 BF16 文件。
3. **合流：连续训练第 2、3 步。** CPU 提前准备新的 TRAIN 输入。用对应版本的 TP4 重新 rollout，再由冻结 reference 对本轮新动作评分；数据消费者封闭整组后，由恢复的优化器推进。交付同一谱系的三步全局 loss、分量、梯度、步号、轨迹/token 分母、策略与数据摘要、运行耗时和发布回执。三次 loss 不要求单调。
4. **业务评价。** 完整 DEV 8690 条上报告 P@1、P@3、平均位置奖励及有效回复率。选模后冻结，在完整 TEST 13078 条上比较直接调用模型、冻结 TFRL 系统、系统加 RL LoRA。训练准入条件不删减评测分母。

遇到故障先查已知边界：设备初始化失败，核对物理与逻辑卡号；概率有差，先核对权重版本、原始音频、padding ID 151643、动作 mask 和三轴位置；反向后缺全局 loss，检查 HCCL 是否仍对 FP64 直接做 all-reduce，已验收实现是收集各 rank 脱离梯度图的标量再求和；导出为空，核对 FSDP 所有者与 384 个 LoRA 张量。冻结的 48 层全宽投影与按绝对位置分块的注意力可直接复用，只有同条件复现数值偏差才重新定位算子。

# Omni System RL 交接：P 节点四卡推理、四卡训练

**接手点：冻结 P 策略已完成训推对齐，第一个真实优化器更新已通过，更新后的 LoRA 已导出。** 推理同事现在可以用它验证服务重载；训练同事可以并行接入四卡 checkpoint 的保存与恢复。该 BF16 adapter 未保存 FP32 主参数、Adam 矩、scheduler、Clock 和 RNG，不能据此直接续报第 2 步。

项目仓库是 [omni](https://github.com/chaosxingxc-orion/omni)，内部工作目录为 `~/workspace/omni`。P（意图排序）、S1（槽位类型判定）、S2（槽值填充）、D（最终决策）的任务定义、知识卡、轨迹、奖励和评测均由 Omni 定义。本次交接的首步 loss 是 **Omni 的 P 节点**在真实语音轨迹上的结果。

训练层复用了 `multimodal-jev` 集成的 verl actor、FSDP 模型加载及 LoRA 参数处理组件。内部路径中的 `jev-system-rl` 是这段集成历史留下的目录名。下文所有源码入口均位于 Omni 仓库或本次封闭运行包内，接手和复核无需另行检出上游仓库。

## 第一次接手：先在 CPU 复核

本包包含交接说明、只读脚本和三份回执：[首步训练](p-first-fsdp4-loss.json)、[首批数据](bounded-p-train-handoff.json)、[冻结策略对齐](configured-actor-parity.json)。Omni 执行源码基线是提交 `aa4e202015a9b3cd95f3704ba7bb2e0641c3b252`。从项目方领取封闭运行副本 `first-loss-accepted/`；其中含本地配置及权重，单独受控交付。只复核结果无需镜像、模型或 NPU。

在交接包目录执行（将最后一个参数换成实际收到的目录）：

```bash
python3 verify_first_step.py --receipt p-first-fsdp4-loss.json --run-root /path/to/first-loss-accepted
```

预期打印 `PASS: ranks=4 step=1 rows=8 tokens=86 loss=0.68749869 grad_norm=8.51779747 adapter_bytes=13420384`。脚本核对训练配置、退出回执、四份 loss、四份结果及权重的 SHA-256，同时检查动作是否漏算或重复。所需结构：根目录的 `TRAIN.local.json`、`hot/EXIT.json`、各 `hot/rank-N/{LOSS,RESULT}.json`、`hot/step-1/lora_adapter/adapter_model.safetensors`。推理侧领取**整个 adapter 目录**，包括 `adapter_config.json`。缺文件补原件；摘要不符先核对版本，无需重跑模型。校验通过仅证明收到的是已验收的那次运行。

## 实验设置与验收边界

| 项目 | 实际设置与结果 |
|---|---|
| 模型与卡组 | Qwen3-Omni-30B-A3B-Instruct，thinking OFF。物理 0–3：vLLM TP4 推理；物理 4–7：HF/FSDP4 训练，容器内逻辑卡号 0–3。两端 BF16，LoRA 主参数 FP32。首步复用已封闭的 TP4 新动作；两侧同时满载的吞吐尚未测量。 |
| 训练人口 | 从 19 条合格 TRAIN 记录按固定顺序取 2 条原始录音，每条新采 4 次，共 8 条轨迹、86 个动作 token。gold 意图在前置召回列表中才准入训练。这个小批用于验证计算与更新。 |
| 策略与奖励 | 温度 1、top-p 1、不截 top-k。正确意图排第 1/2/3/未命中，分别得 1/0.75/0.5/0。两组奖励是 [1,0,1,1] 和 [1,1,1,1]：一条因额外的 Markdown 围栏被原解析器判为协议错误；另一组优势全零。两组都保留。更新前平均 reward 为 0.875。 |
| 优化器与损失 | AdamW，学习率 1e-5；LoRA rank 8、alpha 16、dropout 0，48 层文本注意力 q/k/v/o。先在组内计算优势，再对每条轨迹的有效动作贡献求和，最后除以 **8 条轨迹**。本步 KL 与辅助项均为 0。读 `RESULT.json → objective.components.total`：**0.6874986886978149**；四个 rank 的梯度范数均为 **8.517797470092773**。单 rank 的 `actor/pg_loss` 不是全局 loss。 |
| 对齐与导出 | 冻结条件：冷缓存 87/87、共享前缀 86/86，训练前向与 TP4 同动作 logprob 精确一致；本训练批次 86/86，最大单 token 和整段差均为 0。最终 4/4 rank 正常退出，导出 384 个 BF16 张量、6,684,672 个元素、13,420,384 字节。文件摘要与张量身份分别取训练回执的 `adapter_sha256` 和 `publication_identity`，二者用途不同。 |

首步的判据包括正常退出和导出成功。早期失败运行也有 loss 回执；不要混用其梯度或把独立重启累计成连续更新。当前没有更新态推理对齐、断点续训、DEV 收益或稳定端到端速度的结论。

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
2. **训练侧：产生可恢复的首步。** 在原训练入口接入实际 FSDP4 的分片保存。从冻结基座开一条明确标识的新训练谱系，在 step 1 保存 FP32 adapter、Adam 矩及步号、scheduler、Clock、已消费组和各 rank RNG。封闭后用新进程恢复，核对参数、状态摘要与步号。仅有 BF16 发布权重不足以完成此验收。若新首步的发布身份与本次不同，后续推理验收和采样都须绑定新身份。
3. **合流：连续训练第 2、3 步。** CPU 提前准备新的 TRAIN 输入。用对应版本的 TP4 重新 rollout，再由冻结 reference 对本轮新动作评分；数据消费者封闭整组后，由恢复的优化器推进。交付同一谱系的三步全局 loss、分量、梯度、步号、轨迹/token 分母、策略与数据摘要、运行耗时和发布回执。三次 loss 不要求单调。
4. **业务评价。** 完整 DEV 8690 条上报告 P@1、P@3、平均位置奖励及有效回复率。选模后冻结，在完整 TEST 13078 条上比较直接调用模型、冻结 TFRL 系统、系统加 RL LoRA。训练准入条件不删减评测分母。

遇到故障先查已知边界：设备初始化失败，核对物理与逻辑卡号；概率有差，先核对权重版本、原始音频、padding ID 151643、动作 mask 和三轴位置；反向后缺全局 loss，检查 HCCL 是否仍对 FP64 直接做 all-reduce，已验收实现是收集各 rank 脱离梯度图的标量再求和；导出为空，核对 FSDP 所有者与 384 个 LoRA 张量。冻结的 48 层全宽投影与按绝对位置分块的注意力可直接复用，只有同条件复现数值偏差才重新定位算子。

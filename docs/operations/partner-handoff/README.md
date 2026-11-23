# Omni System RL 合作交接 · P 节点 LoRA 训推校准

P 知识库 v2 的动态 Top‑5 命中 106,018/107,060（99.0267%）。四套 TP4 筛完全部 106,018 条，均有有效 P 结果。正式训练集封存并由固定镜像读回 2,949 条：真实错例 1,290、真实命中 422、合成错例 659、合成命中 578。5,000 是采样上限，缺额来自真实错例 399 条和合成错例 1,652 条。正式续训控制器已从 Clock 3 接续；前三个正式 loss 待实际参数更新确认。 [脚本与配置入口](reproduction/p-screen-release/README.md)。

Clock 3 的双 TP4 训推验收已通过：补验的五条真实录音请求与独立 TP4 回放匹配，八卡训练和评估前向的单动作及整段 logprob 最大差均为零。本轮正式训练尚未提交参数更新。完整 DEV（8,690 条）基线准备曾因旧账本所在卷余量不足退出，原窗口已按零提交结清，保留原预算、请求账本与互斥锁后接续。新窗口的远端 CPU 计划在编码前因依赖缺失退出，原批次留待对账恢复。正式训练消费 0/2,949，新 loss 0/3。

本轮启动时，正式环境引用的探针路径指向较新的工作区版本，恢复锁定的探针与同包依赖后，在固定镜像中通过摘要和加载检查。部署配置仅提供 SSH 目标时，租约命令已能使用固定镜像的 SSH，但端口转发因缺少执行路径而在发单前退出。[端口转发修复](formal-provider-tunnel.patch)按原租约的环境规则选择 SSH，保留部署参数。原断点核验并接续同一租约后，缺少精度证明的推理实例收到的 5 个探针请求均返回 HTTP 200；此处只确认传输与服务响应，数值及参数更新另验。

正式评测可配置 `heldout_lifetime: iteration_deadline`，让 DEV/TEST 内层程序继承外层任务期限。原模板会在累计运行 43,200 秒后耗尽，即使单窗口正常封闭也不能继续。此配置保留窗口、容量、未知请求对账和成本约束，缺省沿用原模板。固定镜像已在完整 8,690 条 DEV 的配置绑定和原校验入口对照通过，没有执行评测推理。见[实现补丁](heldout-program-lifetime.patch)。

P 筛样的可达性报告按真实/合成和正确意图召回排名估算全池错例，历史优先派发的错例单列，避免将优先队列的高错例率外推到整个 TRAIN。使用[报告补丁](screen-population-yield.patch)提供的 --all-domains；报告不改变采样配额、不触发运行。

合成错例稀少时，可在前述补丁之后应用[有限总体上界补丁](screen-synthetic-error-bound.patch)。在装有 SciPy 的 CPU 环境，从脚本目录运行：

```sh
python p_screen_yield.py --config "$SCREEN_CONFIG" --progress "$SCREEN_PROGRESS" --output "$BOUND_REPORT" --synthetic-upper-confidence 0.95
```

仅使用原随机派发顺序中已全部返回的固定前缀。以各个 2 的幂和全量为观察点，校正多次观察，再按不放回抽样求单侧置信上界。一次实测：前 2,048 条中有 68 个错例，37,705 条合成候选的**错例总量**上界为 1,708（95%），配额为 2,311。解释依赖均匀洗牌和固定策略下样本的结果稳定；此报告不触发停机。

正式 P 窗口从配置指定的同一只读发布路径进入，供训练侧打开的源文件可按角色发布到逻辑目录。原路径在 CPU 上复现投递失败；修复后生产者与训练侧文件摘要一致。恢复模块沿用 Clock 3 实际使用的版本，原缺失接口已在固定镜像复验。见[入口与投递补丁](formal-actor-source-namespace.patch)。这些是启动检查；实际正式更新以 optimizer 回执为准。

正式 rollout 可用配置项 `formal_rollout.official_index_cache_root` 复用官方 TRAIN 标注索引。指定 cohort 之外的本地持久目录；原索引按源文件身份独立建库并复核所取行。见[对应版本的实现补丁](formal-train-index-cache.patch)。固定镜像、两域共四条标签的独立进程实测：每窗口建库 4.45 秒；首次预热 4.40 秒，后续读取 0.16 秒；重建读取从 103,543,164 字节降至零，26,578,944 字节的 SQLite 文件不进入窗口清单。耗时仅为 CPU 标注绑定。

正式采样版的续训启动器若从开发树的 `source/` 目录找不到回执模块，可使用[发布目录修复补丁](formal-actor-release-paths.patch)。补丁适用于含正式 actor 分支、启动器 SHA-256 为 `a4dec2f4cedbfc75eb60f40d10bad93a760bdb7988dbc7216103e86e6f6b8fcb` 的版本；先用 `git apply --check` 核对。模块从配置的 training release 根读取，原训推源码双重摘要检查保持生效。固定镜像还需能只读访问发布仓库引用的 Git 对象库。恢复 Clock 3 的只读启动检查已通过，未执行参数更新。

当前 TP4 decode 基线是 `tp4-batched-decode-v2`。新实验从该锁继承；复现旧精度实验保留原版本。见[性能回执与适用范围](../p-rollout-inference-performance.md)。

P 节点在同一严格确定性 FSDP8 谱系完成三次 optimizer 更新。每步使用两组 TRAIN 录音和 8 条轨迹，三步分别有 86、98、91 个动作，全局 loss 分别为 0.6874986886978149、0.8127936070072792、0.8128709792945301。每步 8 个 rank 正常退出。第 3 步 checkpoint 保留原 Adam、scheduler、RNG 状态，Clock 为 3，累计消费 6 组。 两套 TP4 均已在线，依次在常驻基础模型中热切到第 2、3 步 LoRA。第 3 步切换确认卸载旧 adapter、重置缓存，每实例四个 rank 各读回 384 个张量。两条新权重语音请求共 20 个动作。从第 3 步恢复的 FSDP8 在训练与评估模式下，最大单 token 差、整段 logprob 差及全词表 top-1 翻转均为 0。正式 TRAIN 和 DEV 评测仍待完成。 [三步与部署验收](../system-rl-training-topology.md)。

新增：[LoRA 更新态训推偏差排查与复用指南](lora-parity-playbook.md)。包含原失败请求的根因、排查顺序、两端配置及验收命令。

两套 TP4 已在独立推理主机部署并启用 KV 和 prefix cache，每个 rank 读回 384 个严格首步 LoRA 张量。两实例对同一条真实录音各完成一次请求，共 20 个实际动作。训练主机从同一 checkpoint 恢复，8/8 rank 正常结束；训练和评估前向的最大单 token 和整段 logprob 差均为 0，top-1 无翻转。两条诊断耗时 105.58 / 107.66 秒，含适配器装载和全词表观测。当时训练为 1/3，尚未测量常规 rollout 吞吐。 [跨机复核回执](cross-host-active-parity.json)。

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

训练 Docker 的宿主内存上限由部署配置的 `TRAIN_MEMORY_GIB` 提供。已解析配置使用整数 `training_memory_gib`，建议设为 1280。启动器要求该值大于 1024 GiB，创建容器后读回 Docker 的字节限制，再启动训练进程。

配置中的数据和运行项包括：`host` 是训练机 SSH 别名；`run` 是未使用的小写运行名；`source_bundle` 是**训练主机上**含 `HF-ACTOR.local.json`、`tree/` 和成功退出回执的冻结参考运行；`accepted_fields` 是**启动客户机上**的 `ACCEPTED-FIELDS.pt`（SHA-256：`e84a16fffeb2cd230117fb2a85da39117b8b4c2f56169ebf72a3d22f831146e0`）；`dataset_sha256` 取数据回执的 `pins.dataset`，对应 `datasetV2/COMPLETE.json` 的文件摘要；`devices` 为 `[4,5,6,7]`。配置中的 boot、原 guard、镜像、存储必须属于实际部署。热状态落本地持久盘，共享模型/数据/源码只读；由已验收的独立客户端归档。复现会写入新运行目录。此入口接受初始 Clock 和 policy-only 首步，不能循环调用来续训。

## 从首步到连续训练：并行工作与合流条件

1. **推理侧：验证更新后的权重真正生效。** 排空旧请求，记录旧策略身份、缓存失效回执和新 adapter 的实际载入身份；KV/prefix cache 保持启用。用真实语音采新动作，保存请求 ID、版本、token 和行为概率。HF 使用同一更新权重和动作复算。通过条件：最大单 token 差和整段 logprob 差各 ≤0.01，全词表 argmax 无翻转，且依据实际服务回执证明未复用旧策略缓存。
2. **训练侧复核八卡更新。** 在原 actor 的 optimizer step 前后记录本地 FP32 LoRA 分片、归约后的梯度和 Adam 两个矩，按参数名与分片位置还原。以同一初始权重、动作掩码和奖励比较四卡与八卡的差异。推理侧消费部署用 BF16 文件。
3. **合流：连续训练第 2、3 步。** CPU 提前准备新的 TRAIN 输入。用对应版本的 TP4 重新 rollout，再由冻结 reference 对本轮新动作评分；数据消费者封闭整组后，由恢复的优化器推进。交付同一谱系的三步全局 loss、分量、梯度、步号、轨迹/token 分母、策略与数据摘要、运行耗时和发布回执。三次 loss 不要求单调。
4. **业务评价。** 完整 DEV 8690 条上报告 P@1、P@3、平均位置奖励及有效回复率。选模后冻结，在完整 TEST 13078 条上比较直接调用模型、冻结 TFRL 系统、系统加 RL LoRA。训练准入条件不删减评测分母。

遇到故障先查已知边界：设备初始化失败，核对物理与逻辑卡号；概率有差，先核对权重版本、原始音频、padding ID 151643、动作 mask 和三轴位置；反向后缺全局 loss，检查 HCCL 是否仍对 FP64 直接做 all-reduce，已验收实现是收集各 rank 脱离梯度图的标量再求和；导出为空，核对 FSDP 所有者与 384 个 LoRA 张量。冻结的 48 层全宽投影与按绝对位置分块的注意力可直接复用，只有同条件复现数值偏差才重新定位算子。

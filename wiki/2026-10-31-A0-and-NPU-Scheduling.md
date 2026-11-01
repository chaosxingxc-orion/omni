# A0、真实数据工程基线与后续优化

更新：2026-11-01 00:46 UTC。真实500工程基准、Fold B完整1080条首轮收益和 Fold A 完整1080条 A0 结果已锁定；首轮 construction625/625 官方评分完成。K4 vote 相对 lock 下降0.4040pp，按预定门槛停止当前家族。

## 当前结论

真实500条的三组工程比较均已完成推理和官方评分。**本轮TP4-128最快：8卡、2实例、23.05分钟、2.766秒/条**。TP8-256耗时38.85分钟，吞吐没有收益，不追加384。选择依据是完整500条墙钟耗时；单轮结果不证明全局最优或统计显著优势。

**完整Fold B首轮收益已锁定。** 1080/1080条A0官方SLU-F1为**73.2225%**、Intent为**85.3704%**；相对同名单Direct lock提升**28.1046/7.3148个百分点**。SLU-F1配对95%区间为**[+25.7315,+30.5277]个百分点**。历史427条、工程候选新增300条及接续353条逐条来源核验后统一评分；不平均子集F1。详见[完整锁定记录](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-a0-full1080-baseline-lock-zh.md)。

**Fold A A0 完整结果也已封存。** 1080/1080 条真实录音、2160 次模型调用、SLU-F1 **73.3139%**、Intent **85.7407%**、Scenario **89.5370%**、Span-F1 **64.4628%**；8 卡、2 个 TP4 实例，生成耗时 2773.60 秒，整机吞吐 **2.568 秒/条**。输入/输出 tokens 为 13,239,312/3,507,701，处理音频 6169.057 秒。详见[Fold A 完整结果](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-fold-a-c2-verified-result-zh.md)。

## 系统轨迹与 LoRA 参数学习的可行性

**评估结论：8×64GB Ascend值得验证注意力LoRA，但尚未训练、没有新增精度收益。** 本地Qwen3-Omni约31.72B参数，BF16权重59.08GiB；文本注意力rank16约1337万可训练参数，仍需分片、反向激活及训练/推理权重同步。verl已有Ascend、MoE和LoRA相关支持，但本次核查未找到Ascend+Qwen3-Omni音频+LoRA+在线RL的完整已验证组合。

正确轨迹SFT需要可靠监督；GRPO不要求预先全是正确轨迹，但同一输入下需要奖励可区分的当前策略采样。当前seed-only K4实际使用相同提示/证据、仅改变种子并共享P；可作为固定P条件下的S学习资料，不能冒充独立全流程P/S组，也不能跳过策略概率及版本核验直接送入在线GRPO；全错/全对二值奖励组没有组内任务优势。优先验证S求解阶段adapter，保留P和知识配置，然后比较SFT与GRPO；冻结模型TFRL单独保留。训练资源效率和收益需实测，不能用A0推理2.766秒/条外推。

详见[资源、轨迹要求、兼容性和分阶段方案](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-11-01-verl-lora-feasibility-zh.md)。本文为可行性评估，没有变更部署或启动权重训练。

## 精度、效率与资源

| 配置 | 样本完成量 | 资源 | 生成封存耗时 | 墙钟秒/条 | SLU-F1 | Intent |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TP4-96 | 500/500 | 8卡、2实例 | 24.21分钟 | 2.905 | 72.5851% | 86.4% |
| TP4-128 | 500/500 | 8卡、2实例 | 23.05分钟 | 2.766 | 72.6877% | 86.2% |
| TP8-256 | 500/500 | 8卡、1实例 | 38.85分钟 | 4.661 | 71.1400% | 85.8% |

三组均已FINALIZED，剩余0，ETA为0。计时含任务准备和生成封存，不含部署加载及CPU评分。96→128耗时下降4.765%、吞吐增加5.004%。256相比128耗时增加68.51%。保留所有质量结果，不将吞吐选型解释为质量等价结论。

| 配置 | 生成调用 | 输入tokens | 输出tokens | 分配卡时 |
| --- | ---: | ---: | ---: | ---: |
| TP4-96 | 1000 | 6,189,399 | 1,632,704 | 3.227 |
| TP4-128 | 1000 | 6,194,007 | 1,636,795 | 3.074 |
| TP8-256 | 1000 | 6,158,192 | 1,617,376 | 5.179 |

三组断点成本均与封存manifest一致；每条P/S各处理一次音频，总处理音频2947.169秒。96/128的FIT均1000，服务端1000次生成均stop，无length/abort/error/repetition，抢占0；自然前缀token命中分别约46.01%/45.97%。256服务端计数现已核对：1000次均stop，无length/abort/error/repetition，抢占0；输入/输出tokens与manifest完全一致，前缀token命中46.274%。FIT目录各有1000个json/jsonl记录，尚未逐份重验其语义。

**8卡分配不等于8卡算力持续满载。** 96只有3个十分钟硬件采样点，逐卡AI利用率均值14.0%–36.3%；尚不能声称持续满载。以有效吞吐和实验效率为约束，CPU准备、评分、分析与NPU执行异步。

## 数据与测量口径

从SLURP train Fold B的1080条按固定种子无放回抽取500条，500个ID和500个音频哈希均不同。17:59再次复核三个执行目录，dataset文件哈希完全相同，媒体路径均登记为真实录音；[逐配置核验记录](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-real500-dataset-identity-audit.json)保存计数和哈希，不公开原始数据。跨配置共用名单用于配对比较；配置内不重复样本制造并发。 “相同500条”指各配置共用500条不同真实录音，绝非重复同一条合成样本。抽样不读取答案、正确率或历史完成状态。

TFRL construction 使用另一份真实无放回 roster：H1 为313条、H2为312条，合计625条唯一的 `slurp:train` 录音，两个半组无重叠。每条候选共享真实音频和上下文是配对归因设计；这不增加独立样本数，也不能把候选调用成本当作自然分布成本。正式精度、token 和成本只从真实请求收据汇报；合成负载只保留为容量压测记录。完整规则见[真实采样数据与成本估计口径](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-real-sampled-data-policy-zh.md)。

真实音频→检索/上下文→本次P感知→本次S求解→解析/断点。允许自然公共前缀缓存，不重放历史P答案。此前合成轨迹只保留为压力诊断，撤回据此锁定48并发及外推真实成本的结论。后续真实K4采样计划也已移除合成吞吐、token及卡时预测，保留精确调用数和音频量；耗时须由对应真实流程测量。原计划保留归档，修正有独立哈希记录。

500条是每组整机总数，96/128/256指每实例最大并发序列，`max_num_batched_tokens=2048`。当前抽样来自 Fold B 子集，不保证代表完整训练集或 test 的音频时长、上下文长度及任务长尾；训练分布工程测量不代替最终13078条测试集评估。后续消耗预测须用对应真实智能体流程的调用、token 和延迟测量，不能从重复输入或两调用基准直接外推。

## 单条样本耗时拆解

下表为完整500条同轨迹连续区间均值，三组均无计时缺失；单位秒。

| 环节 | TP4-96 | TP4-128 | TP8-256 |
| --- | ---: | ---: | ---: |
| 准入→P请求开始 | 353.34 | 281.82 | 474.31 |
| P请求 | 183.46 | 227.90 | 436.93 |
| P结束→S请求开始 | 3.32 | 3.14 | 1.24 |
| S请求 | 207.94 | 243.44 | 457.43 |
| S结束→样本完成 | 0.22 | 0.30 | 0.15 |
| 单条准入→完成 | 748.28 | 756.60 | 1370.05 |

TP4-128的2.766秒/条是整机吞吐倒数，756.60秒是单条平均延迟。请求时间包含服务排队、音频、prefill、解码及通信，尚不能分离归因。256单条延迟p50/p95为1429.74/2078.47秒。阶段与嵌套计时不能重复相加。

TP4-128平均每条2次生成、输入12,388.014和输出3,273.590 tokens。后续RL若增加分支、轮次或上下文，必须单独核算成本，不能直接套用两调用效率。详见[完整计时及结果](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-real500-first-progress-zh.md)。

## 系统精度基线与配对比较

| 系统 | 样本范围 | 官方SLU-F1 | 当前结论 |
| --- | --- | ---: | --- |
| Direct lock | Fold B完整1080 | 45.1178% | 完整结果已重新校验 |
| Direct aa | Fold B完整1080 | 45.6633% | 保留重复结果 |
| Direct ab | Fold B完整1080 | 45.2940% | 保留重复结果 |
| 旧A0 lock | 完成子集427 | 74.8657% | 不与Direct1080相减 |
| A0完整结果 | 1080/1080，来源合并评分 | **73.2225%** | 首轮Fold B收益锁定 |

Direct三次原bundle、官方结果、1080条名单均已再次通过校验，无新增推理或评分；历史服务凭证归档的5个查找入口已恢复，原始字节不变。A0完整评分后已完成按语句聚类的10000次配对bootstrap，Direct lock为主参考，aa/ab均保留。置信区间描述样本不确定性，不代表完整A0重复运行方差；不据此作测试集结论。

旧完整五分支约17调用/条，两阶段lock仅2调用/条。历史8707次生成中1448次未进入完整样本断点，全部成本保留。427条上L1/L2相对lock未表现稳定收益；预算减少不能直接等同墙钟提速。详细轨迹与分支证据见仓库既有结果文档。

## A0接续与恢复

**353/353条接续已生成封存并正式成功退出，剩余0；完整A0共1080/1080条来源齐全。** 8卡、2个TP4实例，每实例最大128序列。接续生成耗时1086.34秒（18.11分钟）、3.077秒/条，ETA0。完整断点记录706次生成、4512866输入/1195443输出tokens；断点无损坏，服务计数的独立对账尚待完成。单样本准入至完成均值617.56秒、p50 651.02秒、p95 861.65秒，与整机吞吐分开报告。完整1080条官方评分与配对核验均已成功。随后K4经上下文修复和来源合并，625条已完成；当前家族因收益门槛未过结束，没有新的推理任务运行。

登记校验失败源于账本中的大写`SLURP`未匹配精确carrier键`slurp`；已修正文字标签，复用已有准入、供给和服务，通过登记及完整预检后接续。未重复部署、准入或已完成推理；失败现场保留。19项恢复/队列测试通过。完整1080评分与配对分析的独立CPU队列已重新绑定，600秒读取终态。

## Training-free RL：完整625条结果

| 方法 | 完成量 | SLU-F1 | Intent |
| --- | ---: | ---: | ---: |
| lock | 625/625 | 67.2040% | 77.12% |
| aa | 625/625 | 67.0475% | 77.12% |
| K4 vote | 625/625 | 66.8000% | 77.44% |

**vote-minus-lock为−0.4040pp，低于预先登记的+1pp门槛。当前家族停止，两个后继队列均正常结束为stopped-family，第二轮未启动。** H1为−1.2455pp，H2为+0.4074pp；完整结果由官方逐条计数汇总，不平均子集F1。不能事后挑选略高的s4替换投票规则，也不能用另一名单上的A0分数直接相减。

H1完整313条由修复后310条和3条历史上下文完整记录合并并完成七列评分；H2完整312条七列评分完成。新生成共622条，有效结果625/625，剩余0，当前家族ETA0。原失败25条中22条错误上下文不进入有效结果，全部历史消耗保留。

| 生成阶段 | 新生成完成量 | 资源 | 启动到封存 | 吞吐折算 |
| --- | ---: | --- | ---: | ---: |
| H2 | 312/312 | 8卡、2个TP4实例、128序列/实例 | 59.19分钟 | 11.383秒/条 |
| H1接续 | 310/310 | 同上 | 57.56分钟 | 11.141秒/条 |

单条平均延迟分别2571.05秒、2454.97秒；整批耗时排除封存后的CPU评分。**当前无新推理任务运行，8卡服务驻留不代表算力满载。** 下一方法输入、方案及准入尚未完成，队列衔接仍有空档。

## 调用成本与轨迹价值

有效625条共4375次生成：625次感知、2500次候选求解、1250次aa/lock对照。每条恰好7次，没有额外长循环；四候选共享同一次感知与原始上下文。成本仅覆盖有效结果，历史失败尝试和FIT另记。

有效625条共4375次生成，输入34,929,975、输出9,164,770 tokens；推理tokens占输出98.23%。每条恰好7次调用，没有更长循环。

| 用途 | 调用数 | 输入tokens | 输出tokens | 输出占比 |
| --- | ---: | ---: | ---: | ---: |
| 感知 | 625 | 2,401,845 | 1,146,586 | 12.51% |
| 四候选 | 2500 | 21,685,420 | 5,272,426 | 57.53% |
| aa/lock对照 | 1250 | 10,842,710 | 2,745,758 | 29.96% |

291/625条（46.56%）的四个候选规范化答案完全相同，平均每条1.816种答案；候选解析失败0。这里比较规范化结构，不宣称语义等价。正向credit样本消耗1,083,480个输出tokens，其余8,081,290个不能全部称为可避免浪费。

全部4375次调用均完成成本归属；历史3条的21次调用未合入请求阶段计时，其余4354次有对应计时，缺失明确保留。该总数只覆盖有效625条，不含历史22条错误上下文及其他未完成请求、FIT成本。51项正式容器测试通过，全量审计退出0，0新增模型/tokenizer/评分调用。

[机器可读结果](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-tfrl-first625-summary.json)与[完整成本审计](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-tfrl-recovered-cost-audit.json)保存来源哈希。

68/625条获得正向credit（10.88%）；99条对照不稳定，337条无超越基线收益，121条无intent正确候选。经验库真实完成：H1为35示例/0规则，H2为32示例/1规则，合并库64示例/1规则；176次tokenizer请求、0新增生成。经验库产出不等于收益，不绕过停止条件。

## 完整答案聚合筛查结果

固定规则、+1pp SLU且Intent不下降的门槛均在首次评分前登记。完整625条一次官方CPU评分完成：SLU-F1 **67.7474%**、Intent **77.44%**；相对lock **+0.5434pp**，相对原vote **+0.9474pp**。未过门槛，停止该候选，不降低标准、不搜索新阈值，旧K4家族仍停止。

新增模型/tokenizer调用0，复用已有候选；这不表示部署成本为0，方法仍依赖四次候选求解。无独立系统增益或新效率结论。67项相关测试通过。[完整结果](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-tfrl-whole-answer-result.json)。

## 当前状态与下一阶段

Fold A A0 已终态完成 1080/1080：2,160 次真实调用，生成封存 2,773.597 秒（2.568 秒/条），SLU-F1 73.3139%，Intent 85.7407%；完整成本和 checkpoint 凭证已封存。最近一次远端服务核验为 2026-11-01 00:01 UTC，两个 TP4 服务容器仍在运行；该检查未确认活跃生成或逐卡利用率，不能据此断言 8 卡空闲或持续满载。

odd40 仍处于真实运行时准入阶段。625 条真实 construction 的供给审计已完成（625/625 变化、0 回退），但尚未产生 B 候选答案、FIT、生成、评分或系统收益。独立首阶段供给已完成；下一步完成真实 FIT、登记、恢复验证和准入检查，再将 625 条真实任务放入后台队列。CPU 评分和成本审计与 NPU 推理异步执行，观察周期约 600 秒。

首轮 odd40 必须相对同轨迹 A 达到官方 SLU-F1 +1pp 且 Intent 不下降；否则停止，不搜索阈值。方法冻结并完成独立 Fold A 配对确认后，才运行完整 13,078 条 test。

## 候选与条件裁决运行时验证

条件裁决已接入共享运行时、调度、请求门控和成本追踪，并新增整答案选择凭证。真实 H1 313 条、H2 312 条配置实际编译及加载通过：候选部分 4 次，A/B 不同再增加最多 1 次；同轨迹 A 和 A/A 保留原提示。正常生成 2,500–3,125 次，含现有单次补答回退的预算上限 6,250 次；FIT 计划上限 3,125 次，尚未登记或消耗。254 项测试通过，覆盖原始音频传输、四/五调用分支、选 B、非法选择回退、原始响应来源和裁决输入复验、完整样本断点保存恢复，以及未知网络结局不重试。测试使用模拟传输，新增真实模型、tokenizer 和评分调用均为 0。[完整编译凭证](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-11-01-odd40-method-compilation.json)。

独立运行/FIT 身份已建立，旧任务的准入凭证、供给和服务会话未继承。H1 313 条、H2 312 条的独立首阶段 Perception 上下文已完成 **625/625**，并经共享运行时实物检查：两半 `identity_ok=true`、`seed_ok=true`。这只完成首阶段供给；Solver B 和裁决上下文仍需使用当次真实 Perception 输出构建。新增配置/绑定及相关运行时测试 **70 项通过**，公开仓版本同样通过。新增模型和 tokenizer 调用均为 **0**。[供给绑定凭证](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-11-01-odd40-supply-binding.json)。

逐调用持久化已接入 odd40 的共享模型请求边界：发送前独占发布并 fsync 请求意图，收到响应后先保存原始响应字节，再解析及写 trace；新目录项也同步落盘。已收到的异常响应保留；只有意图没有响应时标记 UNKNOWN，日志读取器不重发。候选A/B无法解析时跳过裁决，保留原A并保留样本记录；缺失响应仍报错。该功能目前完成**记录与读取**，尚未接通跨进程接续、成本合并或未知请求对账，因此不能宣称自动续跑已完成。

本次改变了运行时源码，上一轮供给的身份/种子通过记录只代表当时版本；启动前须重新绑定并证明供给输出不变。真实推理仍为0/625、模型/tokenizer新增0，暂无新效率或ETA。下一步接通已知响应复用和原轨迹成本归属，验证部分分支中断恢复后，再完成真实FIT、登记和准入。

实现分支604项测试通过、1项跳过；公开仓594项通过、1项跳过（两树原有测试集合不同；跳过项为Linux上的Windows junction检查）。覆盖真实执行图与模拟传输、进程直接退出、响应trace失败、未知裁决结局、异常响应和并发重复写入；新增真实模型调用0。[验证凭证](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-11-01-odd40-contact-custody-validation.json)。

封存seed-only K4请求与官方评分审计现已完成：625条、3750次Solver请求来源与相同输入/种子策略通过核验；s1–s4二值奖励混合组100条（H1 52、H2 48），全错363、全对162。可优先使用已有混合组进行共享P的S阶段TF-GRPO适配，按半组隔离学习/开发；尚未执行经验更新或证明收益。新增模型、tokenizer、评分调用均0。

8 卡持续满载目标尚未达成。

## 不同证据方案的CPU准备结果

真实construction625供给审计完成：625/625条供给变化、0条回退、每条平均重合10/20条，查询语句与知识池重叠0。采用共享检索器排名，未把ID排序误当检索排名；新增模型/tokenizer/评分调用均0。5项审计规则测试与11项整答案交付测试通过。这不是新精度结果。

新方案固定为原始证据答案 A 与替代证据答案 B，答案相同时交付 A，不同时最多一次受约束 A/B 裁决；非法选择回退 A，不拼接实体。odd40 供给策略已完成 625 条 CPU 审计，0 回退，真实运行时仍需完成候选配置登记、FIT、准入和断点验证。A 兼作 lock，保留 A/A，正常预算为每条 4–5 次调用；旧 K4 与完整答案 medoid 方案保持停止。[设计与完整审计结果](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-10-31-tfrl-next-method-plan-zh.md)。


[当前任务状态、决策与下一阶段计划](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-11-01-task-status-decisions-next-plan-zh.md)


## 有界综合策略 H2 终态

312/312真实开发样本完成，936/936请求收到响应；8卡、2个TP4实例，生成5.238秒/条。learned相对lock的SLU差值为+0.6729个百分点，但Intent下降1.6026个百分点，bootstrap区间跨零，按预定门槛不晋升。成本、1872份journal和终态归档已核验。

[完整结果](https://github.com/chaosxingxc-orion/omni/blob/main/docs/results/2026-11-01-dense-h2-full-result-zh.md) · [成本](https://github.com/chaosxingxc-orion/omni/blob/main/docs/results/2026-11-01-dense-h2-terminal-cost-timing.json) · [归档核验](https://github.com/chaosxingxc-orion/omni/blob/main/docs/results/2026-11-01-dense-h2-terminal-archive-verification.json)

## 当前工程效率状态（2026-11-02）

181 当前只有两个 TP4 服务驻留，覆盖 8 卡；没有活跃实验，因此不能把驻留服务计作八卡有效利用率。已落地的 CPU 准备优化是 [runtime manifest cache](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/src/speech_aware_evidence_acquisition/data/runtime_manifest_cache.py)：它把已验证真实 roster 的媒体身份、SHA-256、时长和采样率固化为可共享、原子发布、篡改即拒绝的 manifest，内容不含标注或提示词。该优化已推送 `main` 的 `195d5940`，尚未用真实 500 样本做 cold/warm preparation 端到端测速，因此暂不报告加速倍数。

下一阶段先在固定镜像内完成 TFRL-v2 bank 的真实 tokenizer 计量、receipt hash 和 4 行 smoke，再把 CPU bank distillation、评分和成本归因接入独立 ready-queue 后继 job，避免 NPU 等待 CPU；smoke 通过后才安排 1,080 行 Fold B。方法冻结且 D/A/best-T 的 runtime、预算、owner allowance、exposure 和 detached identity 齐全后，才运行完整 13,078 条 test。监控维持 600 秒轮巡或明确终态事件，后续每个 NPU 任务继续报告完成量/总量、秒/条、卡数、实例数、逐卡利用率、成本和 ETA。

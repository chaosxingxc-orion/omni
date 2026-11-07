# TFRL 当前任务状态与下一阶段计划

观测窗口：2026-11-07 04:24–04:34 UTC（洛杉矶时间 11 月 6 日 20:24–20:34）。运行数字是快照，不是实时倒计时。

## 结论

TFRL 的 CPU 工程基座已经具备继续准备的条件，但真实 NPU 轨迹采集还没有形成可运行的 successor 队列。现在不应重复加固已经可用的服务和队列框架；应把已有 K001/K002 的 held 资产通过正式配置入口登记为新的候选，完成准入后接入空闲的 B 实例。紧凑 Slot 已生成完整 1080 条；A 的 Decision 后继于 04:30:42 UTC 经正式队列启动，04:37 后已确认真实模型请求，最新 running64。B 服务健康但空闲，八卡尚未持续有效利用。

## 已完成进度

- 固定 Qwen3-Omni-30B-A3B Thinking BF16 推理服务已运行：两个 TP4 实例、上下文 65536、`max_num_seqs=128`、`max_num_batched_tokens=2048`、显存利用率 0.92、SHM 多模态缓存、prefix cache/chunked prefill 开启。
- 完整 ON（P OFF → Slot ON → Decision ON）Fold A 1080 条已完成并通过正式配对核验：Intent 84.0741%、官方 SLU-F1 70.9703%、精确槽位 F1 61.4446%、Strict 587/1080；合同合法 1077/1080，仍有 3 条空答。
- ON 已作为当前精度工作基线，OFF 保留为效率和回退参照。测试集 13078 条仍只用于最终完整汇报，不参与调优。
- TFRL 的 GEPA、ACE、Reflexion、TF-GRPO 配置和反馈字段已分别整理；真实数据集、H1/H2 划分和来源哈希已固定，禁止重新抽样或用合成样本估算成本。
- K001/K002 各 625 条已核实 CPU stage/prepare；原执行报告称 register 完成且存在注册归档，但当前执行 release 的可读注册闭环尚未核实。两者均为 `held`、`generation_authorized=false`、模型调用为 0；这代表可审的输入资产，不代表已采集轨迹。

## 当前 NPU 实验

| 实验 | 资源与设置 | 进度 | 当前判断 |
|---|---|---:|---|
| A：紧凑 Slot 提示，P OFF / Slot ON / Decision ON | TP4，4 卡，客户端并发 64 | 1080/1080 | 04:24:41 observer 确认完整生成、3258 observed contacts、0 in-flight；generation-released 已产生，评分待正式验收 |
| 已完成：紧凑 P 提示 | 原 4 卡 1 实例，已释放 | 1080/1080 | CPU score-ready；完整合法覆盖、官方指标与配对本轮未审，不能宣布收益 |
| B：P 操作辨别提示 | 原 TP4，4 卡 | 583/1080，53.98% | 02:03:44 退出 1；后继 Slot 尚未启动，无有效 ETA |
| ON K2 K001/K002 | 计划 B 实例执行 | 每成员 0/625，0 新模型调用 | 625 个独立样本、两次 rollout，共 1250 样本次；尚未准入 |
| P thinking ON 对照 | CPU 准备，未占 NPU | 0/1080 | request-only 预算修复已合 main；fresh stage/prepare 完成，register 仍失败 |
| Decision 条件证据 | A，TP4，客户端并发64 | 0/1080 | 04:30:42 正式启动；04:37 后真实请求确认，最新 running64；完成样本数待下一次600秒快照 |
| C96 | A 后继 | 0/1080 | binder 等待 Decision，不计作在跑实验 |

04:21 服务快照：A 端 38 个请求运行、等待为 0，KV 使用约 29.43%；B 端服务当前运行和等待均为 0。当前没有证据证明八张卡都持续达到 AI Core 80%，因此不能把“两实例在线”表述为“八卡满载”。

## TFRL 基座现状与真实阻塞

已经具备的部分包括：固定镜像、外部配置启动、真实数据来源绑定、原生 FIT、断点文件、历史响应和成本留存、600 秒轮巡、CPU 评分与 NPU 生成解耦，以及 GEPA/ACE/Reflexion/TF-GRPO 的独立输入字段。

当前 K2 的直接工作是完成新候选准入和实际 B lane 的旧失败任务交接。04:30 后复核纠正了此前“B resident receipt 缺失”的判断：runtime/session receipt 已存在，旧原生 preflight 容器确实 exit0，绑定当前 B PID1397553/8202。旧报告只在 rails-preparation 目录寻找文件，遗漏实际 resident 目录。live profile 标签采用 canonical JSON SHA；两个文件仅 model_fragment 名称不同，推理几何相同，不能将 raw SHA 与 canonical SHA 直接混比。

实际 B 运行使用 tp4-lane-merge 的 lane locks，并非此前引用的宿主全局 flight.lock。当前 lane lease 仍绑定失败 P 实验，需要核对1763个已观察响应后通过既有终态清算入口交接；不能删除锁或重放请求。独立 Terra CLI 正在执行这一任务，尚未交付成功的 K2 admission/queue，不能声明已启动。

## 需要决策的内容

目前没有必须暂停工作的用户决策。建议沿已确定顺序执行：

1. 保持 ON 作为精度工作基线，继续保留 OFF 作为效率参照。
2. 不等待完整评分再派发 NPU；先完成 B 上 K001/K002 的 successor 准入，释放 8202 的空闲资源。
3. TFRL 第一轮先采集同策略 ON K2，再比较 GEPA、TF-GRPO、ACE 和 Reflexion；不能把旧 OFF K2 或单次 GEPA 诊断当作 ON 训练轨迹。
4. 并发先沿已配置的 C96 做单变量验证；只有真实终态显示吞吐提升、质量和合法覆盖不退化，才考虑 C128 或调整引擎参数。

## 下一阶段异步任务图

### CPU 任务

1. 在共享 guard 下审计旧 failed queue 和 lease 终态。
2. 将 K001/K002 的 held 资产通过正式注册入口写入 successor readiness、candidate 和 experiment 配置；复用已生成的注册归档和供给，不伪造身份、不改内容哈希、不重新采样。
3. 绑定当前 B 的 profile、runtime receipt、session、PID/端口和服务启动时间，重新运行只读 preflight。
4. 编译 K001/K002 的 generation/FIT/launcher pins，设置 600 秒监控并保留断点接续键。
5. 独立完成 A 的 Slot 评分与 B 的失败原因归档；评分不阻塞 NPU 生成。
6. 统计每个节点的样本数、秒/条、模型调用、prompt/completion tokens、FIT 成本和失败/恢复成本。

### NPU 任务

1. 当前 A 继续运行到终态，不重启、不重复已知请求。
2. K001/K002 准入后立即进入 8202 队列，两个成员按依赖图排队，使用固定 ON 配置和断点续推。
3. 维持输入就绪，避免节点交接时断供；先验证 C96，再评估是否到 C128；不要同时改变提示、知识库、预算和引擎 geometry。
4. 每 600 秒或收到明确终态事件更新运行快照，报告“已处理/总量、秒/条、卡数/实例、running/waiting、KV、ETA 和成本”。

## TFRL 的科学顺序

第一轮先使用 completeON 的固定 625 条 construction roster，补采至少两个显式 seed 的 ON 轨迹；同分全错组标记为 `no-learning-signal`，不能强行制造优势。GEPA 先用于根因反馈和候选策略，TF-GRPO 需要真实同策略多 rollout 后再计算相对奖励，ACE 只能提交有来源的卡片差分，Reflexion 只能保存绑定真实失败和后继尝试的 episodic memory。任何候选都要在固定开发回归集上完成配对评分，再决定是否进入 Fold B 确认。

TFRL 不改变模型权重；后续 LoRA 属于独立阶段，仅更新主模型适配器并冻结音频塔、视觉塔及既有对齐投影。所有改动通过配置文件和脚本进入固定工程基座。

## 精度、失败与效率解释

| 已核验 Fold A 系统 | Intent | 官方 SLU-F1 | Strict | 合法回答 | 墙钟秒/合法条 |
|---|---:|---:|---:|---:|---:|
| 历史 P→Solver | 85.7407% | 73.3139% | 本轮未核 | 本轮未核 | 不作跨配置估算 |
| 完整覆盖 OFF | 85.0000% | 65.9217% | 517/1080 | 1080/1080 | 6.043 |
| 完整覆盖 ON | 84.0741% | 70.9703% | 587/1080 | 1077/1080 | 10.617 |

ON 比 OFF 的 SLU 高 5.0486 个百分点，但少 10 个正确意图，单位合法样本耗时约 1.76 倍；距离历史 P→Solver 仍差 2.3436 个 SLU 百分点。ON 是已选工作基线，不是最终已对齐方案。相对旧 k8 ON，完整 ON 的 SLU 变化为 -0.2493pp，配对区间跨零；覆盖改进不能解释成稳定精度提升。

B 的失败日志明确为 perception_role 的 `candidate 0 unresolved_fields must be a non-empty string`。对应空白归一化修复 0ba6a4cb 已合 main，但生产旧 release 没有自动获得修复，不能据此宣称已恢复。此路径属于程序协议兼容问题。此前完整 ON 的三条空答则是两条输出空白退化至截断、一条 Slot 重复空 filler，原生重放未证明存在完整合法答案被错误丢弃；应与本次程序错误区分。

P ON 的新阻塞是 register 报 `registered probe program, material or recovery changed`，不是当前服务不支持 7168 输出。只修配置/注册对新增节点预算的真实绑定，不改引擎或绕过校验。

此前紧凑 Slot admitted→finished 平均 675.23 秒、P95 911.48 秒；最新完成样本 cost.latency_seconds 汇总均值 664.72 秒，属模型联系耗时口径，不能代替端到端拆解。11.02 秒/条是并发吞吐倒数。04:21 服务采样 A running38/waiting0、KV29.43%，B running0/waiting0；A 已进入尾部。逐卡 Core 单点为 77/55/32/42/0/0/0/0%，不是持续平均；尚未达到八卡持续80%的证据标准。prefix cache 计数已有真实命中，服务生命周期比例约31%，不能当作本实验专属命中率。引擎128是并发序列上限，2048是每次调度token预算，均不是实际样本batch。

完整 ON 生成成本已核：3265 次调用、输入35,005,713 tokens、输出4,973,031 tokens。紧凑 Slot 的 1061 条已完成样本成本：3201 次调用、输入 34,475,103 tokens、输出 5,184,760 tokens；04:22:05 attempt 账本累计 3244 calls，包含与完成样本不同的在途/尝试口径，二者不得相加。当前 FIT 成本与失败/恢复成本尚未独立结算；失败请求和历史恢复费用继续保留。当前最大确定性资源损失是 B 空闲；先恢复有效任务，再用同策略窗口比较 C64→96→128，不承诺加速倍数。

## 交付与下一阶段验收

主线程负责规划、评价与整合，独立 Codex CLI 子进程负责配置实现和结果分析；禁止协作型子智能体。每项子进程必须交付 report.md、result.json 和 final.txt，状态消息不算完成。B 队列审计子进程已经交付完整报告，结论是服务健康但启动身份绑定未闭合；后续 successor 实现子进程因宽范围扫描且没有完整交付已终止，不能将其计作修复成果。

下一阶段先完成 B 合法接续及 A 生成后继，然后并行收紧凑 P/Slot 的覆盖与配对结果、修复 P ON 注册、准备教师输入。GEPA 可复用既有封存反馈先做 CPU/本机教师候选准备；TF-GRPO 的组内相对反馈必须等 ON K2 真正产生。ACE/Reflexion 同步配置准备，共用评估器，各自独立基础模板；本阶段不声称四种算法都已运行。只保留固定知识库内有来源的知识卡解释/呈现改进，不让教师新增标签或无依据知识。

目前无须用户重复授权。后续若出现 SLU 明显改善但 Intent/成本显著退化且无法按既有目标取舍，再提交具体配对结果供决策；执行注册、文件权限和队列接续由工程处理。

## 本轮复核后的执行重点

- A 的 Decision 条件证据后继存在真实 queue，状态 waiting，依赖当前 Slot 的 generation-released；CPU 评分不是依赖。C96 binder 仍 waiting-container，未启动。生成容器存活不等于后继启动成功，最终以真实模型请求确认。
- B 的直接工作是完成当前 resident 身份登记和旧任务终态交接。已知 live profile 与候选 profile 哈希不同，先比较有效配置和哈希计算口径；服务 PID/启动时间及 runtime/session receipt 应由既有入口绑定。不能把哈希差异直接判作模型故障，也不能手工改哈希绕过。
- K2 使用同一 625 条真实 construction 样本、seed 1001/1002，共 1250 样本次，当前新调用 0。计划正常生成 3750 calls；generation 总上限 7500、FIT 上限 7500 均是预算，不能报为已消费。不同 seed 不保证形成有区分度的反馈，必须核查实际输出与奖励方差。
- 先恢复 B 的有效任务供给，不扩展通用加固范围。若 K2 身份登记仍阻塞，可使用已准备且能合法接续的节点对照填充 B；不能让所有实验依赖单个 K2 工程任务。所有依赖在占用推理资源前完成。
- 质量主线仍为 100% 有效回答、追平 Fold A 历史 P→Solver；TFRL 输入与适配器并行准备。候选按知识卡呈现、召回消费、节点提示分别标记，单独比较 P/Slot/Decision 输出，保留正向修改后再组合验证。
- TF-GRPO 等待同策略组内反馈；GEPA、ACE、Reflexion 可先准备各自模板和真实失败上下文。统一评估指标、开发集、父策略与成本账本，不强行共用不同算法的动态采样。教师通过本机外部 API，当前没有新教师结果可宣布。
- C96 优先检验同策略吞吐，收益存在再推进 C128。当前 C96 候选为 OFF，不能拿其速度直接证明 ON 同比例加速；ON 若采用新并发需做同配置对照。关注有负载窗口的 AI Core 均值、tokens/s、样本吞吐和节点 P95，不能用结束阶段单点代替稳态结论。
- 当前无需用户重复授权；工程接续不是研究决策。若出现稳定 SLU 收益但 Intent 明显退化，则提交配对错例、置信区间和成本差异再讨论取舍。LoRA 暂不进入本轮主线，全量测试待开发基线确定后执行。

## 本轮实际执行与同步

主线程收到紧凑 Slot 的 generation-released 后，通过既有 configured_job_queue once 触发 Decision 后继，启动容器 ecbd46b72571ed13255942dd3f924c0a5b056bc5a14c2dfd70632ae237b1cc79，04:30:42 UTC running。首次普通用户调用因0600配置无读取权限退出，随后用已有owner身份执行成功；未改文件权限、未跳过校验。

Decision 启动耗时已定位：runtime build 18.75秒、prepared verification 71.34秒，随后实际进程在读取模型权重分片做校验。这是启动CPU/存储成本，不能混入模型推理耗时。当前不重启；后续应将可复用资产验证前移到占用实例前，且不凭缓存跳过真实身份核验。

独立CLI并行任务：Terra负责B K2配置化激活；Sol负责紧凑P/Slot覆盖与正式配对分析。两者禁止协作子智能体及重复生成，必须交付完整文件。当前尚无新精度结论。

GitHub SSH直连超时、HTTPS写入缺认证；复用本机已有SSH代理配置后main2fa2b1d6、Wiki fe569c7已推送成功。两仓已设置本地core.sshCommand复用该代理，避免以后重复排查或误报远端已同步。

后续确认：Decision 已进入真实 vLLM 生成，先观测 running33/waiting27，随后 running64。紧凑 Slot 的最终容器退出1原因明确为生成后的官方评分批处理失败，非1080条生成失败；既有CPU工具处理评分，不重发模型。B旧P lease原生清算容器exit0，1763 observed contacts、新模型调用0，归档旧lease和reconciliation；K2正在配置化admission，K001 admission 已实际执行但退出码1；错误为 `registered configuration changed; rebuild registration before admission`，尚未创建workspace或模型调用。

## K2 最新准入状态

固定 release `b322be85` 已生成并验证，B resident/runtime/session 凭证已用原生 preflight 核验，旧 P lane lease 已清算。K001 的正式 admission 在配置组合校验阶段失败，错误是注册配置与当前 admission 配置不一致；这不是服务故障、数据损坏或模型调用失败。当前重点是比较注册时使用的 `data/repo/model fragment/session/experiment manifest` 与 admission 的实际映射，修正外部配置后从 held 资产继续。现阶段 K001/K002 仍为 0/1250 NPU 样本、新模型调用 0。

当前资源快照：A/8200 的 Decision 任务 running=64，B/8202 running=0；因此仍不能宣称八卡满载。B 的有效负载接入被 K2 admission 阻塞，A 任务不受影响。

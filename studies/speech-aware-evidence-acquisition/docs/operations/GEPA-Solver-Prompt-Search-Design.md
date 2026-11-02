# GEPA Solver 消费提示搜索：最小真实实现设计

交付：2026-11-02。设计已完成，未修改仓库源码、未 SSH、未运行模型或正式测试。所有数量是建议的预注册上限，未产生 GEPA 实验结果。

## 结论与边界

实现 `gepa_solver_prompt_search_v1`：冻结一个通过支持性审查的知识版本、原始 P 与检索/排序/供给、解码、输出格式及模型权重，只改变 Solver 的“如何使用已供给知识”指令区块。真实搜索必须包含当前父候选的新 Solver 轨迹、TRAIN 反馈反思变异、同 minibatch 子候选实测、实例级 Pareto 采样与完整内部选择集实测。一次重写或重放51组历史反馈不构成 GEPA。

首版实现论文不带 Merge 的 GEPA；只开放一个模块，跨模块互补的 system-aware merge 无有效应用空间。保留父候选 DAG 和 merge 字段，但 `merge_enabled=false`。不要把两段提示拼接当作论文 Merge，也不要为了实现 Merge 拆出原本没有独立执行语义的假模块。

当前应先完成知识构建对照再冻结知识。无通过审查的知识候选时，可以准备实现与数据协议；用空知识运行只能标为独立“基础 Solver 指令优化”实验，不能声称验证知识消费增益。

## 已核对原文与作者代码

- 论文固定版本：[arXiv 2507.19457v2](https://arxiv.org/html/2507.19457v2)，§3、Algorithm 1–2，Appendix D.1 Algorithm 3–4。Algorithm 1 明确 feedback/Pareto 划分、父候选 minibatch 新轨迹、反思修改、严格改善后完整 Pareto 集评分、最后返回内部选择集平均分最高的候选。
- 作者代码固定审阅 commit：[`8c7b697a71977980afc7537dac0881e784eef99f`](https://github.com/gepa-ai/gepa/tree/8c7b697a71977980afc7537dac0881e784eef99f)。检查 `src/gepa/strategies/candidate_selector.py`、`gepa_utils.py`、`proposer/reflective_mutation/reflective_mutation.py`、`core/engine.py`、`strategies/batch_sampler.py`、`proposer/merge.py`。
- 已下载原文和相关源文件用于本次核对：`/tmp/omni-gepa-original-source/`，不是已安装运行的依赖，也没有声称审计整个仓库。正式实验需同时记录论文版本、上述代码 commit 和本项目适配 commit；不能跟随 main 浮动升级。
- 当前作者实现的 `remove_dominated_programs` 是按平均分由低到高删除“所有获胜实例仍有其他幸存覆盖者”的冗余候选，再按获胜实例频次抽样。不能用一般的两两数值向量支配或平均分 top-k 静默替换。建议对该固定实现做小型等价测试；记录 `pareto_selector=gepa-8c7b697-cover-prune-v1`，并明确这是本文及所选作者版本的实例获胜集合选择，不是新提出的 Pareto 算法。
- 作者代码默认严格 minibatch 改善；论文分别报告 GEPA 与 GEPA+Merge。因此首版禁用 Merge 是清晰的实验变体，不构成假装完整 Merge 复现。

## 数据划分、历史接触与隔离

1. 固定完整 H1 TRAIN 313 个 query 作为搜索母集，不能仅从51个奖励不同组挑选内部选择集。51组/204条历史轨迹属于知识构建证据，不等于新的当前候选 rollout。
2. CPU 按 query identity、源 utterance identity、音频内容哈希建立重复连通分量，同一个 query 的全部轨迹与相同音频必须同侧。核对 H1/H2/test 的重复关系；跨侧碰撞导致协议预检失败并报告，不能悄悄删除最终完整测试条目。
3. 建议内部选择目标80个 query，其余约233个用于反馈；按真实 intent（必要时既有 scenario/slot 关系）确定性分组分配，seed=20261102。不能构造 intent×slot 笛卡尔积。若重复分量无法恰好80，输出最接近且不超80的完整分量选择并锁定实际名单，母集总量与原因同时记录；以下成本公式使用实际 N、V、b。
4. 一次性写 `partition.json`：母集来源、每侧 IDs/音频 SHA、分量归属、真实计数、seed、分配算法版本、全部内容哈希。所有方法复用此稳定名单。反馈 minibatch 按冻结随机顺序遍历，首版最多12轮、每轮4条，不补齐尾部重复样本。不能为吞吐或预算估算复制样本。
5. 本轮知识构建曾接触全部 H1，包括后划入内部选择集的样本。该选择集只能称 TRAIN 内搜索选择集，不能称未见验证；`knowledge_exposure_ids` 显式记录。GEPA 本轮反思器仅获得 feedback 侧新轨迹与标签，不能收到内部选择集逐样本答案/错误或 H2 错误。选择集数值分数只给 CPU 搜索器，反思 prompt 中也不放选择侧轨迹。
6. H2 完整312条只用于冻结后筛查。不得把 H2 标签、逐样本错误回送反思器；多次方法筛查带来选择偏差，需保留全部记录，另行冻结确认协议。test 13078全量在方法和采用规则冻结后评估，绝不作为搜索或停止规则输入。
7. Solver 的执行输入只有原任务输入、已封存 P 输出、同一冻结知识供给与当前指令；评分参考答案在 CPU scorer 侧，不能因为它属于 TRAIN 就把答案渲染给 Solver。反思器可看到 feedback 侧真实答案和差异，用来归纳规则，但最终提示禁止 query-ID→答案表、复制训练样本答案或嵌入原始轨迹；语义审查判断是否只是记忆化。

## 评分、选择与变异的确定协议

**搜索标量。** 首版复用 `rails/tfgrpo_learning.py::intent_slot_reward`：0.5×intent_exact + 0.5×exact_slot_F1，空槽位分母为0时 slot_F1=1。使用官方评分器产生的 intent/TP/FP/FN，再计算该预注册适配奖励。奖励只是搜索代理，不叫官方 pooled SLU。单独保留 strict success、Intent、官方 SLU及实体错误；最后科学报告以官方 pooled 指标与配对差异为准，不对单样本F1求均值后冒充 pooled SLU。

**初始化。** 候选c0为冻结知识下当前 Solver 指令，完整运行内部选择侧，得到按固定顺序排列的 V 维真实得分向量。每个候选记录指令内容SHA、完整系统SHA、知识与P供给SHA、父ID、生成请求/响应/FIT/预算来源；候选池只有完整选择侧评分者。

**每轮。**

1. 对每个选择样本i建立 `F_i={c : score[c,i]=max score[:,i]}`。取并集，按固定作者实现消除冗余覆盖者；使用冻结 seed/RNG 状态，按剩余 `F_i` 中出现次数抽样父候选。存下父ID、全集分数哈希、集合、RNG前后状态和实际随机选择，确保可重建。
2. 取 feedback 侧下一组真实 b=4 query。以当前父指令执行新 Solver rollout；相同候选/样本/所有输入与解码身份完全一致且有完整原生响应的既有搜索观测可直接复用，引用原请求，不能再次计为新增调用。历史其他策略的答案不能作为当前父的得分。
3. CPU官方评分和结构化差异产生 `feedback`：原意图/槽位、预测、缺失/额外/错误值、父P与知识使用轨迹、数值奖励、严格成功状态、数据来源。若4条全部达到奖励1，记录 skipped-perfect，不发无依据变异；该轮上限被消费，不能自动追加第13轮。
4. 一次 `gepa_reflect_mutate` 模型调用同时输出诊断与新指令区块。严格JSON：父候选SHA、模块名、仅 feedback query 的证据引用、诊断、`solver_instruction`。原指令框架中的输出schema/工具与占位符全部CPU固定，变异器只返回文本区块，不能替换系统角色、模型、标签空间、知识库、检索或P。建议变异文本上限2048 tokenizer tokens；完整反思输出cap4096，实际reasoning/answer预算由注册配置绑定。
5. 无效JSON/越界引用/FIT超限等按照已知成本封存并停止本次搜索，禁止自动修复重发。相同指令SHA属于已知无变异，记录 rejected-noop、继续下一个已授权轮次，不创建重复候选或虚构零成本成功。
6. 以新指令在同一b条执行真实Solver，父子共用完全相同P、知识供给和解码。仅当平均代理奖励严格增加时进入内部选择评估；不因措辞更好、模型自报更高分或同分就接受。完整 V 条评分与来源验证结束后原子加入候选池；部分向量不是0，也不能进入Pareto。
7. 12轮、实际调用/tokens/时间任一硬预算先到即停止新增请求。最终输出完整选择向量平均奖励最高者（相同均值固定选择最早候选ID），保留c0作为可能结果。若仍是c0，报告搜索未找到改善，不发布伪造的新版本。保存全部拒绝候选和成本，不能只报告赢家。

**Merge后续。** 多个真实模块开放后才另设 GEPA+Merge：两个非祖孙候选、有共同祖先、相对祖先有互补的模块变更、逐模块按原文组合，完整记录双父和共同祖先，去重既有组合；同样先比较minibatch，再完整选择集。不在此版本写一个未用的“智能合并LLM”替代作者system-aware crossover。

## 有限预算与有效资源使用

推荐初始研究配置：N≤313、V≤80、R=12、b=4、最多13个已测候选、merge=off，单个Solver/反思completion cap4096，上下文49152、额外reserve1024，原生包装prompt必须≤44032。P的具体输出配置保留基线并单独绑定，不借搜索修改。提示区块2048上限是新预设容量，非实测。

若需要一次全新的P输入库，逻辑调用最坏为 `N + V + R*(2*b+1+V)`；取313/80/12/4得到1461：313 P、1136 S、12反思。若已有身份完全匹配的封存P，可复用并单列313的历史实际成本，而不是再次执行。实际只用到部分feedback query时可以惰性采集P，此式仍是上限。已存在的原生raw-vLLM fallback若保持每逻辑请求最多两次完成，则原始生成硬上限2922；必须在注册中显式绑定逻辑/原始两种计数，不能以1461掩盖fallback成本。反思无自动格式纠错重试。

建议独立上限：总输入24,000,000 tokens、总输出6,000,000 tokens、租用8卡墙钟4小时（32 allocated card-hours）。这是停止预算，不是ETA、预算会全部用尽或NPU忙碌时间估计；与ACE/Reflexion的51组预算不等价。几何上限允许的token总量可大于这些预算，故实际可能在12轮前终止。预发每个并发批次前按所有在途请求的真实prompt+completion上限原子预留，完成后按真实账本结算；fallback也须预留。无法承担完整下一阶段时不开始该阶段，保留当前最佳完整候选。到时间停止新dispatch，已发请求继续收集终态、记账与seal，不将deadline变成unknown重试理由。

真实成本拆分：继承知识/P/上游研究历史、搜索新增、筛查H2新增；总体按原生请求ID去重。报告完成S或P样本/目标、候选完成V/V、反思轮数、模型calls/input/output/audio、wall seconds、秒/条、8卡/2×TP4（以实际receipt为准）、allocated卡时、等待时间和ETA；未测不填推算值。原生FIT单列次数与耗时，不当作模型生成tokens。

同轮Solver批次使用共享Flight现有并发/队列，单轮反思保留依赖；不能为提高8卡利用率增加重复样本或随意并行多个反馈依赖轮。CPU评分可在生成释放后执行，但下一轮必须等真实评分来源完整。独立有价值任务可由已有队列交接，不另起私有启动器。

## 与现有Flight的接入：真实缺口

已读当前 `rails/tfgrpo_runtime.py`、`reflexion_runtime.py`、`ace_runtime.py`、`tfgrpo_consumption.py`、`learning_candidate_release.py`、`scripts/analysis/tfgrpo_h2_paired.py` 及 Flight dispatch。当前ACE/Reflexion是有限顺序反思调用；GEPA还需要动态候选的真实Solver评估与决策状态，不能只在strategy表加字符串。

建议最小组合方式：CPU搜索状态机串起现有正式Flight评估作业和正式学习反思作业；所有模型请求仍经Flight、FrozenCoreClient、唯一共享guard、已注册原生FIT和contact journal。唯一部署入口保持 `scripts/environment/configured_job_launch.py`，扩展受验配置kind/共享模块dispatch，不新增 GEPA shell launcher、HTTP客户端、评分器或账本。每阶段先从已封存前驱输出构造不可变配置，实际prompt哈希才能注册；预注册的是有限阶段槽位与全局预算，不能在c0时伪造尚未存在的c1 prompt。状态机不是独立租约体系。

建议状态：INIT→SEED_SELECT_EVAL→PARENT_PICK→FEEDBACK_PARENT_EVAL→REFLECT_MUTATE→FEEDBACK_CHILD_EVAL→[REJECT或SELECT_CHILD_EVAL]→COMMIT_CANDIDATE→下一轮→FINALIZE_SEARCH。每条转移绑定前驱stage seal、scores SHA、candidate SHA、全局预算快照，使用原子写+fsync；恢复不重新抽父或抽minibatch。

- **实际Solver执行**：复用生产P/S节点与消费模板渲染、既有供给渠道及官方scorer；P缓存必须有独立traced原生来源，不能把完整原始trace塞进Solver的prepared-text提示冒充真实部署消费。需补只运行已封存P之后S的受验上下文入口；若现有接口无法证明等价，首个版本可用完整P+S Flight重评、扩大实际成本，不应绕过真实Runner。
- **反思执行**：复用 `MeteredExperienceCaller` / `ExperienceLearner._call` 的真实模型调用链。新增精确GEPA schema、有限轮次stage identities与从已观测前驱重建prompt的validator；其样本锚属于feedback侧。原生FIT测完整包装，预算gate在generation intent之前。
- **提示候选类型**：当前 `tfgrpo_consumption.expected_text()`要求非空`items`经验且固定SEPARATOR，`learning_candidate_release`与H2只认ACE/Reflexion。GEPA输出是指令区块，不能装成经验条目，不能单加 `EVALUATION_METHODS` 就宣称接通。新增严格 `candidate_kind=solver_prompt` verifier与模板区块绑定；保留原经验校验不放松。
- **H2归因**：GEPA效果主要比较“同一冻结知识+优化指令”对“同一冻结知识+初始指令”，共用一次新P，312×(P+S+S)=936逻辑调用。现有 learned/lock 若lock不含该知识，不能用于单独归因GEPA。另需总体lock对照时显式第三个S臂，312×4=1248，分别报告提示优化增益与整个知识系统增益，不混淆。新H2消费候选仍为未采用，不自动部署；原+1pp SLU且Intent不降仅是筛查门槛，报告区间与全候选选择影响。

## 恢复、来源与发布

恢复先验证旧进程终态/boot/共享guard租约，所有尝试原生journal全局unknown=0后才允许任何新请求。一个候选分支有unknown不能只跳过该分支继续其余分支。保留FIT、intent、response、请求SHA、得分、失败与成本；用 `NativeContactReplay` 的全部联系人观察屏障并核验确切输入身份。

状态转换后崩溃：从sealed模型响应及CPU评分重建确定状态；请求响应已知但bridge落盘失败时补确定性派生文件，不能重发。无响应的intent保持unknown等待核验；没有intent且无真实请求证据的阶段可在授权预算内首次执行。评分中断重做CPU官方评分即可。候选已完整选择评分但尚未入池时从完整来源完成原子commit；池中不能出现两份同一SHA。已完成search发布中断必须支持从最终seal重建release，不得再次反思或重新搜索。

最终 `search-release.json` 绑定完整候选表/祖先、分区、知识/P/供给、模型/运行时、全部stage seal、计费去重表、冻结赢家与选择规则、候选类型、语义审查和原始请求/FIT来源。内部选择标签/原始轨迹不入公开repo；公开只发方法、hash、预算与聚合结果。保留继承研究成本但别将未实际消费的旧融合错误算成该方法新增。搜索结束或c0获胜属于合法终态；未知请求存在时只能held，不能生成“已知完整”release。

## 实现文件与必要测试

建议新增：

| 文件 | 职责 |
| --- | --- |
| `rails/gepa_search.py` | 纯CPU分区验证、候选/DAG、作者选择算法、批次/RNG、接受规则、预算与确定性状态转移 |
| `rails/gepa_runtime.py` | 共用Flight的阶段绑定、反思请求重建、Solver评估来源、native FIT、全局预算与恢复 |
| `rails/solver_prompt_candidate.py` | 指令区块schema、不可变模板边界、prompt release/消费校验 |
| `configs/learning/examples/gepa-solver-search-v1.json` | 完整有限预算和未填真实release的明确DRAFT，不伪装可运行 |

已有文件定点扩展：`core/flight/runtime.py`的来源/dispatch；`rails/tfgrpo_runtime.py`的精确schema/stage与已有caller；`scripts/analysis/tfgrpo_final_job.py`的共享strategy准备/注册接口；`scripts/environment/configured_job_launch.py`的配置dispatch；`learning_candidate_release.py`及H2 stage/job/paired/read的候选类型、同知识双S臂与来源验证。不要复制ACE整个启动脚本成GEPA脚本。实际Solver缓存/导入入口须在实现时以现有node/runner契约定位，不能在设计稿中声称已经存在。

正式固定镜像测试必须覆盖：

1. 同query、多轨迹、重复音频连通分量不跨分区；H2/test泄漏、失真名单、伪造标签来源拒绝；只向反思器暴露feedback，向Solver零标签。
2. 手算三候选×四样本矩阵验证每实例winner集合、全同分、互补赢家、冗余覆盖删减、固定seed采样及作者固定版本等价；平均分greedy替代会使测试失败。
3. 父子同batch严格提升才入下一阶段；同分/退化/全满分/重复指令；不完整选择向量不入池；赢家c0合法；公式预算按实际V、batch尾部计数。
4. 真实消费路径只改变指令区块，P、知识、供给、输出schema/解码保持同SHA；继承上下文不能带gold；同知识H2双S身份正确，原经验候选路径不受影响。
5. 实际包装后超FIT、输入或输出预算不足、在途并发原子预留、fallback raw计数、时间到不新发；故意把离线token计数设小仍被native FIT拦截。
6. 原生请求发送前/后、响应已知落盘前/后、CPU评分中、pool commit前后、final release前后的崩溃恢复；任一分支unknown阻断全局；恢复不新增已知调用、不重抽随机数、不漏历史成本。
7. 集成测试使用现有Flight真实注册/门控/模板渲染/评分与受控模型响应，走至少两个变异轮、一次拒绝与一次完整候选接受，不用mock掉关键validation；伪造candidate+release一起重hash仍须因来源/策略不符拒绝。
8. 最后一个有限真实机制检查：新身份、完整原生FIT+响应+成本和至少两个候选实评。真实机制未执行前，只能报告实现与正式测试通过，不能报告GEPA复现完成或有收益。机制检查不得用test标签，不能将小样本差异称稳定结果。

按功能闭环优先：先交付上述搜索状态、真实评估和来源/预算/恢复必需路径，再运行；不把新通用队列平台、泛化多模块图或吞吐改造作为前置。性能优化随后基于实测热点推进。

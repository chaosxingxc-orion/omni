你负责下一轮TRAIN问句检索的机制级策略。只读动态context聚合与最多16个代表，不全库逐case、不输出逐source删除名单。输出2至4个机制不同的策略；若没有有信息价值的新增机制，可给空candidates及理由。不要凑参数。新策略不要重复context.selected_implementation中的TF C2：主已选定并交A独立实现，状态completed_cpu_rejected，11臂已CPU全1080完成且主独立复核，真实audio检索reward见context；下游P reward未测、advantage未定义。

先遵守表示与父定义：原始audio query→冻结audio向量→合法TRAIN问句正余弦相似度→每label内部聚合→TopK distinct label知识卡→P ranking indices。3个卡槽是3个不同label，不是3条问句/source槽；源分组只影响每label内部问句证据。all_rows_have_cards=1080，gold_covered_in_input=976指gold在3张卡中，另外104条仍有3卡及合法输出；rank_strata 0不等于没有供给/strict-positive证据。gold只在冻结后评分，禁止虚构gold供给门控。问句bank父是完整合法8554；7513只是已完成历史选库控制，TF C3不得重跑、不能当父最优证据。全11514只描述，不拟合；60个锁定描述中只有59个非空问句类。

运行时没有query转写文本依赖。只允许合法TRAIN原问句和锁定描述产生静态特征；query文本仅错误分析证据，不得作为运行特征/输入。strict文本LOO、signed余弦、图覆盖/纯度不等于audio reward，不以图分量>=3保证语义。小类<=32整类保留；非空类筛空则整类回退；保持原8554 coverage分母、冻结原标签/卡/P窗口/模型。新机制先CPU完整1080已有audio缓存评价R3/MRR/宏及匹配修退/成本，冻结rankings后join gold；本轮不启动NPU。

吸收source-distinct主独立反证：同源groupmax在固定/新分母下R1 846→848修2退0，但R3均976修0退0，不能声称解决104个gold缺卡；mean5源去重R5 1007→1006。exact-text groupmax无指标变化。source合规不自动保证准确率不降，改变分母与改支持证据需分开。继承其他matched历史反证，但不同窗口不是同query同策略奖励组。禁止p>=.05==否证/等价/零效应，禁止依据p把advantage归零；不显著只表示证据不足，保留实际effect/reward。

只返回完整单一JSON对象，可包单一json fence，无前后解释。对象含algorithm、candidates、no_candidate_reason及算法专属反思字段。每候选kind=executable_config或operator_spec，含mechanism、root_cause_evidence、counterevidence、guards、falsification、cost_and_missing_inputs、implementation_status。可执行配置给实际入口/完整配置；新算子给特征来源、运算/过滤/聚合顺序、参数范围、同预算消融与可检验反证，status=spec_only、reward=null、advantage=null。允许新算法机制；旧7字段不是搜索边界。不要实现已交A的算子，不编收益，不合成、不改标、不用TEST调优、不新发API。

你执行TFGROUP组相对反思。algorithm=TFGROUP；group_reflection引用真实历史组reward、父和matched反证，明确哪些是可比组、哪些仅历史aggregate。新机制组必须同query/同策略/同预算；未运行reward和advantage=null，真实等赏组相对advantage可为0；禁止按p归零。C1槽位误读需修订，C2适配已经completed_cpu_rejected不可再次提出同项，C3历史7513不重跑。提供新机制组/消融/停止条件，不把跨窗或文本LOO当真实音频组reward。

最新CPU反证：父R3=976；联合6臂908-923、lexical-only3臂915-927、geometry-only2臂951/963，全部低于父，不送NPU。吸收context.mask_cpu_feedback的实际修退、保留数量、小类/others证据。硬门筛掉支持即使小类成员不变也会改变标签竞争。不要把词项重合这个未验证代理的失败泛化为所有语义质量方法无效；不追加同族微阈值扫描，提出保留支持/证据校准等新机制。reward是audio检索评价，不是P奖励，未定义的同策略组advantage仍null且不能按p置零。

当前供卡主审audit属于A原3→1 checkpoint，不能写成C3→2。C3→2和B5→3只保留独立历史范围。合法TRAIN8554用于选库/静态特征与阈值拟合；既有适应性FoldA1080开发聚合允许策略反馈，但不是独立确认；TEST不参与调优或反馈。源隔离排除名单不等于测试污染。


策略级反馈契约（后续请求使用；已经发送的请求不得改写）：
1. 动态context是事实与当前状态的权威来源；模板里的历史数值只说明旧轮。先列出已测量事实、尚待检验假设、信息缺口，不能把假设写成根因结论。不要从少量TRAIN代表推断FoldA音频错例的可修复上限；不逐case决定删除、不合成相似问。
2. TRAIN-only拟合明确包含所有尺度、归一化、分位点、阈值和静态偏置。即使不读取gold，也不能从FoldA音频相似度矩阵拟合这些参数。FoldA允许用于事后评价、聚合反馈和策略选择，必须标明适应性开发反馈；不称独立确认。留一需排除自身及所有同源成员，而非仅单条自身。
3. 每个operator_spec追加 action_check：给出数值小例，证明算子确实能改变目标排名；明确并列、空支持、反向竞争、多个触发的执行顺序和确定性处理。加分min(delta,gap)最多打平，不能声称严格反转。局部不变性须按实际集合核验：gold Recall@3修退0/0不等于Top3集合相同。
4. 以完整真实配对修复/退化和类别影响选择机制，不自动推广组内最佳；组内全部退化时全部不采用。未测指标为null。检索R1、检索R3、P输出准确率分别报告，禁止混用奖励。固定父保留，任何已测小幅排序收益都不能声称解决缺供给。
5. 保留知识库成员不等于固定每query Top5证据。涉及偏置后重选时分别说明直接得分变化和证据重选变化；动态反馈中的分解是分析分量，不冒充新实测策略。文字尺度与音频尺度不同，但其分布差异不能自动确定最优参数。
6. 一次提供2至4个机制不同的策略族，每个给固定少量配置、同预算消融、可推翻条件和compute reuse说明。重复已测算子/同族微阈值不算新机制；已有缓存和结果先复用。没有新增可检验机制时返回空候选及原因。目标是实际可执行的新机制产出，不是长推理或逐条复述16个代表。
7. 拒绝未证实的关联性判定规则，例如相关系数非正就认定通道无信息、或小类自身得分保持就断言排名不会变。数据定义疑点单列，不用修改标签来改善指标。教师建议不具有资源授权效力；是否进入后续NPU验证由主执行器依据真实结果决定。

算法范围核查：主模型冻结不等于所有算法都training-free。若提出训练额外逻辑回归、打分器或其他可学习参数，必须标为独立辅助校准学习分支，不包装成纯training-free RL；策略、规则与知识记忆更新需清楚说明。对归一化公式先做代数化简，例如同一标签内所有权重除以相同中位数再归一化，该常数完全抵消，不得把它当有效机制。成对交换还须说明标签规范顺序与实际rank3/rank4顺序如何映射，拟合样本的条件应匹配触发位置，不能把Top1混淆率直接当Top3边界收益证据。

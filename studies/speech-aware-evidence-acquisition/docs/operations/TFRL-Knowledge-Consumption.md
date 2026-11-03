# TFRL知识消费方案（开发态与运行态）

## 目标与边界

首轮消费侧只优化“如何使用已冻结知识”的 Solver 指令。检索排序、证据筛选各自作为后续独立实验因素，首轮保持冻结。模型参数、P（感知）输出、知识库内容、输出协议、评分器和多模态对齐全部冻结。知识构建侧的新增条目必须先完成来源审计，不能把消费搜索同时当成知识质量验证。

## 开发态闭环

开发集固定使用 TRAIN 难例分区：父候选和子候选在相同有序 minibatch 上运行；诊断器可读取参考答案，Solver 不可读取 gold。每个样本建立 `error_stage`（知识缺失、检索未命中、排序错误、消费忽略、输出协议/程序错误）和证据引用。反思器只能看到 feedback 分区的轨迹和标签，不能看到 selection/H2/test 标签。

当前 v2 候选状态为 `seed_eval → parent_eval → mutate → paired_eval → selection_eval`。反思器只返回受限 `solver_instruction` 文本，不能修改 P、知识、工具、标签空间或 JSON schema。父子必须绑定同一 P、同一供给、同一解码配置；`compile_template_pair` 的 `gepa_pair` 声明和原生 response/FIT/score receipt 是入池前的必要证据。父子 minibatch 平均代理奖励严格增加后才运行完整 selection；同分、无效 JSON、越界、来源缺失均拒绝，不自动重发。

代理奖励只用于搜索（建议 0.5×intent_exact + 0.5×exact_slot_F1）；最终报告使用官方 pooled SLU、Intent、strict success、修复/破坏配对差异和完整成本。GEPA 选择采用固定作者实现 `gepa-8c7b697-cover-prune-ordered-v1`，不以平均分 top-k 替换实例获胜覆盖。

## 运行态配置

运行态冻结最佳候选，按预注册条件路由执行：正常模式只输出必要证据和结果；debug 模式额外写节点耗时、候选证据排序、消费决策、失败分类和请求来源。debug 不改变算法。每个样本最多固定的检索/重排/筛选次数，异常路径和重试上限在配置中声明。缓存只复用内容寻址的 P/知识材料，不跨样本泄漏。

示例配置应声明：`mode=development|runtime`、`observability=normal|debug`、`candidate_kind=solver_prompt`、`partition_sha256`、`feedback_ids`、`selection_ids`、`excluded_ids`、`knowledge_sha256`、`perception_sha256`、`search.seed/batch_size/max_rounds`、`budget.calls/prompt_tokens/completion_tokens`、`runtime.max_retrieval_items/max_retries`。不允许在完整测试运行时修改这些字段。

## 当前真实机制实验

已有固定16条 seed probe 使用 P + gepa/lock 两个 S，只验证接线；不能冒称父子优化。当前 compile_template_pair 已支持 P + parent/child/lock 三个 S，GepaSearchState v2 已要求对应成对证据，但 paired workload 登记、官方读数与真实反思驱动仍需接通。首轮采用“反思前 parent 运行；反思后 fresh P 上 parent/child 同跑”的明确适配，不要求跨阶段 P 导入。反思与最终比较的 P 可以不同；接受判断必须使用后者同 P 的 parent/child 分数，不能用反思前 parent 分数替代。

## 接受标准

1. 每个候选有完整原生请求、响应、FIT、评分和成本；unknown 阻断恢复。
2. parent/child frozen hashes 完全相同，只有 instruction template SHA 可变。
3. selection 完整向量后原子入池，恢复不重新抽样、不重发已知请求。
4. H2 只做冻结后筛查，test 13078 条只做最终报告。
5. 晋升前同时检查收益、成本、协议错误率和多模态回归；无稳定收益保留 seed。

## 一个真实案例的诊断与受控复验

私有封存 TRAIN 1231 第4条 rollout 的 raw 含 food_type 与 order_type，交付后仅剩 food_type，预测意图为 recommendation_locations，参考意图为 takeaway_order。已有固定镜像重放能复现 ti1 类型过滤删除，但历史源 receipt 与统计表绑定尚待核齐，因此当前不能定性为程序 bug。错误意图与交付删槽分别归因：单独关闭过滤的 CPU 诊断不会修复错误意图。

消费候选只增加可泛化指令：“区分寻找餐馆推荐与发起外带订单；以用户当前行为目标判意图，知识仅作为判定依据；逐项保留话语明确表达且符合任务定义的实体，不把低频当作无证据。”这只是待检验假设，不能声称有益。实际新请求须先核验该样本属于固定 feedback；若非反馈侧，仅保留历史审计，另取固定 feedback 中符合该机制的已知难例。child 与当前 parent 共用一个新 P、同知识、同解码；原始输出和交付输出分别计分/留证，识别提升发生在哪个节点。

## 与知识构建侧的交接

交接材料至少包含知识内容 hash、原始证据/反例引用、适用意图与槽位的原始定义、confidence/status、构建预算和已接触样本集合。消费侧输出命中/未命中/被忽略/与话语冲突的分项证据，以及匹配的 request 与评分来源。双方均不把反思器自述正确当成验证。最终主线程执行 K0C0、K1C0、K0C1、K1C1 四格消融，固定真实样本与每样本 P；分别报告构建增益、消费增益及交互作用。

## 算法对照与预算

比较冻结 baseline、同证据 TF-GRPO 经验组选择、Reflexion 反思记忆、GEPA 提示搜索；ACE 主要属于构建线，组合实验须单列。统一官方评测与分区，分别按实耗 tokens/卡时和方法自然成本报告，不用相同调用数伪装等预算。首轮共享 P 成对图的调用上限为 `3V + R*(3b + 1 + 4b + 3V)`；V=80、b=4、R=12 时为3468逻辑调用，若每次允许一次原生fallback，generation预留最多6936。反思 JSON重试不另设隐式机制。此为保守硬界，不是一次必须执行完的任务；先完成1条机制、再固定16条，再扩展搜索，token/时间预算先到即停止新增请求。

配套 JSON 是设计草稿，尚无对应运行入口，不应提交给 launcher。实际轮次预算需按已绑定部署容量、真实 prompt、原生FIT及在途调用原子预留；不复制旧8卡预算到单TP2探索实例。

# TFRL知识消费方案（开发态与运行态）

## 目标与边界

消费侧只优化“如何使用已冻结知识”的 Solver 指令、检索排序和证据筛选。模型参数、P（感知）输出、知识库内容、输出协议、评分器和多模态对齐全部冻结。知识构建侧的新增条目必须先完成来源审计，不能把消费搜索同时当成知识质量验证。

## 开发态闭环

开发集固定使用 TRAIN 难例分区：父候选和子候选在相同有序 minibatch 上运行；诊断器可读取参考答案，Solver 不可读取 gold。每个样本建立 `error_stage`（知识缺失、检索未命中、排序错误、消费忽略、输出协议/程序错误）和证据引用。反思器只能看到 feedback 分区的轨迹和标签，不能看到 selection/H2/test 标签。

候选状态为 `seed_eval → parent_eval → reflect_mutate → child_eval → selection_eval`。反思器只返回受限 `solver_instruction` 文本，不能修改 P、知识、工具、标签空间或 JSON schema。父子必须绑定同一 P、同一供给、同一解码配置；`compile_template_pair` 的 `gepa_pair` 声明和原生 response/FIT/score receipt 是入池前的必要证据。父子 minibatch 平均代理奖励严格增加后才运行完整 selection；同分、无效 JSON、越界、来源缺失均拒绝，不自动重发。

代理奖励只用于搜索（建议 0.5×intent_exact + 0.5×exact_slot_F1）；最终报告使用官方 pooled SLU、Intent、strict success、修复/破坏配对差异和完整成本。GEPA 选择采用固定作者实现 `gepa-8c7b697-cover-prune-ordered-v1`，不以平均分 top-k 替换实例获胜覆盖。

## 运行态配置

运行态冻结最佳候选，按预注册条件路由执行：正常模式只输出必要证据和结果；debug 模式额外写节点耗时、候选证据排序、消费决策、失败分类和请求来源。debug 不改变算法。每个样本最多固定的检索/重排/筛选次数，异常路径和重试上限在配置中声明。缓存只复用内容寻址的 P/知识材料，不跨样本泄漏。

示例配置应声明：`mode=development|runtime`、`observability=normal|debug`、`candidate_kind=solver_prompt`、`partition_sha256`、`feedback_ids`、`selection_ids`、`excluded_ids`、`knowledge_sha256`、`perception_sha256`、`search.seed/batch_size/max_rounds`、`budget.calls/prompt_tokens/completion_tokens`、`runtime.max_retrieval_items/max_retries`。不允许在完整测试运行时修改这些字段。

## 当前真实机制实验

先使用已有固定 TRAIN feedback 侧最多16条做 GEPA 原生接线探测，目标是验证 P/parent/child/lock 三臂、原生 FIT、响应恢复、官方评分和来源账本；不得将探测称为精度收益。真实搜索前必须补齐跨阶段 sealed-P 来源证明；单独 fresh-run 父子会重新采样 P，不能声称同证据消费比较。

## 接受标准

1. 每个候选有完整原生请求、响应、FIT、评分和成本；unknown 阻断恢复。
2. parent/child frozen hashes 完全相同，只有 instruction template SHA 可变。
3. selection 完整向量后原子入池，恢复不重新抽样、不重发已知请求。
4. H2 只做冻结后筛查，test 13078 条只做最终报告。
5. 晋升前同时检查收益、成本、协议错误率和多模态回归；无稳定收益保留 seed。

# 知识消费 GEPA 首轮候选结果与接续方案

## 已取得的结果

四个预注册 mutation 分支全部终态，三个生成合法候选，一个失败。全部使用同一冻结模型、2 卡 / 1 个 TP2 实例、原生 65,536 上下文，沿用固定 TRAIN 反馈来源。每支仅 1 次 native generation；成功支各有 1 次 FIT，FIT 不混记为生成调用。本轮是候选构建，不是精度评测；尚不能宣称任何算法收益。

| 分支 | 结果 | 输入 tokens | 输出 tokens | 模型响应秒数 |
|---|---|---:|---:|---:|
| evidence-use part-a | 成功 | 41,483 | 1,542 | 41.50 |
| evidence-use part-b | 失败，重复生成后截断 | 36,045 | 16,384 | 400.34 |
| intent-slot-reasoning part-a | 成功 | 41,479 | 3,444 | 76.90 |
| intent-slot-reasoning part-b | 成功 | 36,050 | 3,468 | 75.61 |
| 合计 | 3/4 合法候选 | 155,057 | 24,838 | 594.35 |

从首个容器启动到最后一个容器退出为约 20 分 45 秒，包含准备检查、FIT、串行调度与失败清算；不是 594 秒纯推理时间。本次四个候选各用其固定的 2 条诊断反馈，后续计划使用同一 16 条反馈评测集，不能把分支数或 artifact 行数说成已评测样本数。目前配对评测 0/16，尚无可信秒/样本和 ETA。

失败分支已由队列自动清算，后续两个独立分支正常完成。最终 queue 状态为 failed，是因为保留了该失败，而不是其余候选没有运行。不得重新提交已完成调用。

## 失败根因及优化方向

失败输出的 `evidence_query_ids` 进入重复循环：出现 1,014 个知识 value ID 引用，只有 30 个唯一值，个别重复 51 次。最终 `finish_reason=length`，16,384 completion tokens 用尽，JSON 未闭合。这一分支独占本轮约 66% 输出 tokens 和 67% 模型响应时长。

现有校验器已经禁止重复和非法 feedback ID，但这是生成后校验，不能阻止生成阶段浪费。现有提示仅说“cite supplied feedback”，同时上下文存在大量知识 evidence ID；两个命名空间没有对模型明确区分。应在新配置版本里列出唯一允许的 query ID 集合，要求 `evidence_query_ids` 为其非空、不重复子集，并明确 knowledge value IDs 不是 query IDs。可以进一步使用 enum/uniqueItems/maxItems 输出约束，但必须先验证推理引擎支持，并保持完整失败响应与新配置身份。简单扩大 completion ceiling 不解决该重复模式。不能静默修剪旧输出并冒充原生成功。

## 三个候选的可检验假设

1. **evidence-use part-a：证据优先。** 强调提取语音明确表达的实体，不让知识联想替代语音，且不固定选择首个 intent。
2. **intent-slot part-a：联合约束。** 根据选定 intent 的合法槽位限制实体。风险是错误 intent 进一步压掉正确槽位，因此尤其需要统计 strict repairs 与 breaks，不能凭诊断文字直接推广。
3. **intent-slot part-b：保留完整表面形式。** 候选指出 time_zone 规则要求去掉 `time/standard`，但反馈真值保留 `pacific time`。这可能是知识构建的错误规则，也可能有上下文适用条件；已交给构建线核查。消费侧优先保留语音完整值的修改须与构建侧规则修复分开消融。

以上都是模型提出的诊断假设，不能当作已验证的根因。

## paired16 接续阻塞

第一个候选 material 构建已成功，stage 在零新模型调用时被 `require_unspent` 拦截。实查对应 ledger 无注册、runs/contact-journals 为空。原因是准备脚本提前创建并挂载新 run 目录，而 guard 把任何已存在 runs 目录视为已有执行。

已制作 v2 启动配置生成器：CPU stage/prepare/register 只提前挂载 prepared 目录，不挂载未使用的运行目录，复用现有 material，不删除 guard，不修改活跃 mutation 树。v2 尚未完成真实验证：之后出现 Ceph 挂载根 Permission denied，launcher 在检查历史只读来源时退出，未启动新模型调用。因此不能声称该阻塞已经修复。

## 下一阶段 DAG

- 三个成功候选：复用封存 mutation → material → stage → prepare → register/admit → preflight → 同一 16 条 shared-P / parent-S / child-S / lock-S 生成 → CPU 官方评分。
- 每支正常生成预算 64 次；保守 fallback 上界 128 次，FIT 单列最多 64 次。三支总正常生成 192 次，不能把上界当成实际消耗。
- 失败 evidence-use part-b：先完成命名空间与输出约束配置修复，再使用新 experiment ID 做一次受控生成；不阻塞三个已成功候选的评价。
- 三臂读数必须包含官方 intent、槽位、联合严格正确率、repair/break、逐样本因果差分和完整成本。16 条属于开发反馈，不能声称泛化收益；入选策略仍须使用既有固定 selection 集再检验。
- CPU 评分与下一候选 NPU 任务异步运行。当前队列已结束，后续队列尚未就绪，不宣称探索卡持续满载。

## 已实现的配置修复

提交 `fdaa844e` 增加可选 `source.mutation_prompt_policy: explicit-feedback-query-ids-v1`。开启后单独列出允许的反馈 query ID，要求非空、唯一、有界子集，并明确禁止将知识 value ID、content hash 填入该字段。缺省不改变历史提示，因此旧配置与 replay 保持兼容；不裁剪证据、不修改原始失败响应、不减少开发态诊断预算。

固定镜像完成 mutation runtime、job、paired workload 回归：**43 PASS，5.66 秒**。验证了显式 ID 契约、无损证据保持、缺省字节兼容以及未知 policy 拒绝。尚未在新 NPU 分支验证收益，不能说重复失败已消除。修复已审查合入main；后继通过配置选择，不修改旧提示或历史响应。

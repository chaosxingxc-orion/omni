# 节点 Thinking 配置决策

## 已核验事实

2026-11-05 对当前两路 Fold A1080 运行树的真实 contact journal 和请求 payload 做了独立核验。Perception 和 Slot 的请求均使用 `enable_thinking=true`、`thinking_budget_tokens=3072`、`max_tokens=4096`。这与已确认的 Perception 关闭 thinking 约定不一致，因此旧运行只能作为配置偏差和工程链路证据，不能作为 P-off 精度基线。

## 首轮固定策略

新首轮候选将关闭 Perception thinking，并在请求 instruction 中明确：只返回配置的 JSON；输出 1–3 个按可信度排序的合法意图候选；每个候选只提供简短区分依据、已供给的证据 ID 和待确认字段；不输出思维过程，不提取槽位，不代替后续节点决策，不补造音频或知识中不存在的事实。保留 `max_tokens=4096`，避免同时改变输出预算。最终必须从实际 wire payload 验证 `enable_thinking=false`，包括 fallback 路径。

Slot 首轮保持 thinking 开启。Slot 仍需在原始音频、候选意图和知识约束下处理实体边界、实体类型及候选条件化语义；关闭 thinking 可能减少成本，也可能损失边界消歧能力。待 P-off 版本在既有 Fold A1080 上锁定后，再只切换 Slot 的 thinking 开关，使用同一名单、模板、知识和评分口径比较槽位 F1、SLU、截断率及节点耗时。Decision 保持现有配置。

旧 P-on 请求、响应和成本全部保留；P-off 新身份不跨配置重放旧响应。所有结果仍按原 Fold A1080 与 Fold B1080 横向比较，不新增开发集。

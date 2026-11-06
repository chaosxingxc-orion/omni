# 三节点候选协议 v2

v2 是 SLURP 的可选协议，默认 v1 不变。固定输入、模型和知识供给时，Perception 产生候选，Slot 按候选独立抽取实体，Decision 只选候选 ID；程序使用被选 Slot 的结果构造最终任务答案。节点提示可分别优化，变更模板、知识或模型都需要新的配置和运行身份。

## 配置

从已有 `phase-native-module-v1-c2` 基座生成 preparation-only 配置：

```bash
python scripts/analysis/prepare_intent_slot_candidate.py \
  --source <baseline.json> --out <preparation.json> \
  --slot-mode batched --protocol-version v2
```

也可选择 `per_candidate`。生成器保留基座的知识供给与 delivery normaliser，新增 Slot 供给和节点指令；原始基座不变。核心块如下，必须与生成器写出的 v2 policy、模板、响应格式和知识配置一起使用：

```json
{"schema":"saea-intent-slot-pipeline-v2","slot_mode":"batched","max_candidates":3,"candidate_outside_recovery":false,"duplicate_candidate_policy":"group_v2"}
```

生成结果仍需要独立 preparation、FIT 和注册后才能执行；不能热改已绑定哈希的运行。候选集外恢复尚未实现；运行级 `slurp_projection_v3_delivery` 不可与 v2 合用。已有 `native_episode.deliver_normaliser` 可用于派生答案，例如 `saea-deliver-normalise-v4g`，其所需固定统计资产仍受原有准入要求约束。

## 契约与追溯

Perception 的原始响应保留在 trace；所有原始候选的合法 ID 与证据引用先验证，同 ID 按首次出现顺序归组。投影保留每条原始行的区分说明、引用和未决字段，receipt 记录完整来源；不同措辞不自动判为矛盾。原始行数仍受 `max_candidates` 约束。

Slot 请求的响应 schema 使用当次候选 ID 和精确行数；封口还检查每个候选恰好一次。解码约束不保证 ID 唯一，缺失或重复即拒绝，不自动补空实体或重试。Decision schema 只允许当前候选 ID，不能选择集外候选；程序从该候选的 Slot 实体派生任务答案。

计费、请求和原始响应摘要保持现有链路。receipt 分别绑定原始 Decision、派生答案摘要和可选 normaliser 的输入/输出摘要；bundle replay 从原始节点 trace 重算派生答案并重放 normaliser。正常交付与拒绝均保留原始成本和来源。测试与部署环境应验证实际请求 schema、FIT 和 replay，不能把协议可交付性等同于候选召回、知识充分性或最终准确率。


## SDK 节点 rails（v2 opt-in）

使用 `--protocol-version v2 --node-rails` 显式启用。它仍使用同一个 `NativeRoleNode`、同一套 MeteredNativeSpeechRequest 和同一条 Flight/FIT/trace 路径。`before_model_call` 核验当前 sample、role、query state、snapshot 和候选 handoff，并将已经准备的 typed snapshot evidence 绑定到请求；`after_model_call` 只处理无 tool call 且 stop 的最终模型响应。原始响应在 completion sink 中先计费和记录，随后 after rail 做严格解析、同 ID 归组及引用去重、生成下游 handoff，再完成受控 settle。工具调用或非终态响应不会触发业务归并。

Handoff 只含结构化样本判断和锁定 snapshot 的 value/facet ID、正文 hash、scope 与 snapshot hash；模型的 distinction 和解释句不能成为知识正文；经过供给清单验证的引用仍是样本证据引用。下一节点请求消费同一 handoff 的已验证消息，不重新扫库；非法 source URI、跨 sample/role/visit/branch 或旧 snapshot 会拒绝。原始 response hash 与 normalized hash 分开保存，Decision 仍只选择 candidate_id，最终答案由已验 Slot 结果构造。

Slot 的 type/filler schema 没有 mention ID 或位置，因而保留重复实体以维持多重集评分；不判断其是否指同一次提及。不支持知识 index/context 和额外 value renderer 组合；节点 rails 必须显式配置 `answer_only_fallback`：`budgeted-v1` 保留既有有预算 answer-only 恢复，`off` 关闭；生成器选择 `budgeted-v1`。重试使用同一知识材料和首次 FIT 接触，沿已有路径缩小输出预算，每次请求独立计费；现有 FIT 以逻辑 contact 唯一约束，不对 re-ask 单独执行 FIT。非终态按既有有限完成策略处理。固定镜像测试使用 mock response，不能替代真实数据的候选召回、知识相关性、准确率及吞吐测量。

回放通过内容寻址的共享复合快照重验来源、scope、各库 pin、请求和归并摘要。每个样本 trace 和 checkpoint 只记录该快照引用与 pin；共享快照必须随 attempt 保留，缺失或篡改时回放拒绝。归档迁移时必须保留引用路径，目前尚未实现透明重定位。transport 投影的 scope 和请求身份受校验，但 bundle 当前不保存完整 HTTP wire 正文，不能仅凭 bundle 逐字重放 wire。

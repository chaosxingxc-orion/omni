# 三节点知识供给与消费核查

结论：当前 P → Slot → Decision 已配置知识供给和消费路径。不能把准确率下降归因于“没有配置知识组件”。拆分脚本确实删除了旧 `native_episode.consumption_module` 声明，但它是策略编译器声明，不是知识注入的唯一入口。

## 节点实际配置与证据

| 节点 | 供给与消费方式 | 抽查实际请求的知识量 |
|---|---|---|
| Perception | 意图边界检索；原始语音与合法标签表；生成 1–3 个候选 | 8 条意图边界 |
| Slot | 候选条件下的槽位契约、训练示例及继承的意图边界；逐候选抽取实体 | 8 条边界 + 20 条示例 + 6 条槽位契约 |
| Decision（配置名 solver） | P 的候选、Slot 结果与知识；选择完整假设 | 同上，共 34 条 |

上述数量来自 B 已完成运行的一个完整三节点样本，不代表全量平均值。三份请求日志的请求哈希已与输出回执核对。1080 条输出中，651 条包含可读取的 pipeline evidence_bindings；这些记录的 P、Slot 知识清单均非空。其余 429 条不能据此判定未供给，需沿中断/拒绝请求日志进一步分类。Decision 的供给已抽查请求，未做全量请求审计。

供给通过 `knowledge_stores` 的 phase/candidate supply 执行；`phase_tools: {}` 和 `evidence_source: none` 不表示没有知识。请求中的 supplied_evidence 保存正文；history 中的角色消息只是输入的一部分，transport 另行把正文渲染到模型输入，Slot/Decision 配置了候选槽位视图投影。本次核查范围为已存请求、回执和源码路径，未对全部最终 HTTP 请求体作独立回放。

## 旧消费模块是否漏迁

`prepare_intent_slot_candidate.py::derive` 删除 consumption_module 和 prompt_contact_limits，并把 solver 的供给复制到 slot，更换三个节点模板。

核对仓库 Fold A 源配置，旧模块唯一启用的组件为 centroid-boundary-key；其编译后的 intent-boundaries 配置与 B 实际配置完全一致，检索能力仍在。旧 node-context-presentation、knowledge-consumption-opportunity、solver-citation-contract、reviewer-g5-repair 等原本即为 off，不能说此次拆分关闭了这些已启用能力。

当前消费路径是节点模板 + 结构化历史 + 槽位视图。尚不能声称旧模块所有可选 TFRL 功能已迁移，亦不能将当前基线视为已经接入学习到的消费策略。

## 已识别限制与下一步

1. 共享任务说明仍只列 Perception/Solver/Reviewer/Planner，未列新增 Slot；节点专属模板已正确说明 Slot 职责。这是说明不一致，尚无证据证明它造成指标退化。
2. 候选集外恢复关闭：P 若漏掉正确意图，后两节点无法补回；需测候选召回与“正确候选存在时”的最终准确率。
3. Decision 必须原样选择 Slot 实体，不允许修复。因此应单独统计 Slot 抽取错误和 Decision 选错，不能混成知识供给失败。
4. solver 供给直接复制给 Slot，证明接通，不证明内容最适合各节点。后续应按节点检查相关知识覆盖、冲突/冗余、视图完整性和模型使用情况，再进行固定开发集配对实验。
5. 当前 B 正式评分 Intent 46.2037%、SLU F1 50.7539%；相对历史 Fold A 结果显著退化。知识接通不等于消费有效，根因分类未完成前不把下降归因于 thinking 或拆节点本身。

本次仅审计与记录，不热改正在运行的实验，也不改变其科学配置。

# 候选条件化意图—槽位原生基座

## 执行结构与研究边界

默认旧配置仍执行原图。显式 `native_episode.intent_slot_pipeline` 开启 `Perception → Slot → Decision`，Decision 仍使用原生 `solver` 角色和原任务 JSON。模型回答原文直接进入已有封存路径，不改写为伪造的模型输出。合法场景—动作对来自 TRAIN ontology；槽位类型使用全局合法集合，不把观测关联当作意图×槽位的硬性约束。

Perception 返回 1–3 个真实、唯一、排序候选。Slot 逐候选提取完整实体、证据和未解决问题；Decision 选定完整假设，槽值不能静默重写。首版禁止候选外扩展，不按未经校准的概率改变 K。

## 配置与调用预算

```json
{"schema":"saea-intent-slot-pipeline-v1","slot_mode":"batched",
 "max_candidates":3,"candidate_outside_recovery":false}
```

`batched` 对应 `phase-native-intent-slot-batched-v1`，正常每样本 3 次调用。`per_candidate` 对应 `phase-native-intent-slot-per-candidate-v1`，上限 K=3，实际 K+2 次，不补齐候选。首版逐候选串行，跨样本沿用共享 runtime 并发；尚未实现同样本分支并发。

节点模板由 `phase_material` 注入，知识由 `knowledge_stores` 注入，预算由节点配置注入。配置派生复用已有稳定数据和知识 pin：

```bash
python scripts/analysis/prepare_intent_slot_candidate.py \
  --source frozen-two-contact-baseline.json --out pipeline-preparation.json
```

输出是准备输入，需新 preparation/FIT/registration 身份才能排队。禁止热改在途执行树。

## 来源、恢复与失败处理

保留每个候选组的 Perception/Slot request、response、text SHA；最终选择绑定原生 Solver 来源。每条 Slot 请求保留原始音频和合法知识；后分支隐藏前分支模型回答。CarrierHooks 注入合法性检查，通用执行器不硬编码 SLURP 标签。

所有调用走现有 native Flight/FIT/journal。已知响应按请求字节校验回放；未知请求不盲重发。结构错误、非法候选、跨候选结果、缺失槽位组及非法 Decision 都使用现有 seal-refusal 保留并计费，不删除失败样本。单节点长度截断不允许降级成可消费中间结果。

## 新基座评估要求

功能测试通过不等于精度收益。固定同一 80 条开发集比较旧 P+joint S、增强 P+joint S、增强 P+Slot+Decision，必要时补逐候选版本。统计 intent accuracy、官方槽位指标、完整任务正确率、协议失败率、分节点时延、tokens/FIT/实际调用成本及配对不确定性。

在开发集选定图、模板、知识版本与推理配置后冻结，并重新完整评测 TEST13078。旧基座成绩不得转标为新基座结果；测试集不参与选择或调优。此实现阶段真实模型调用为 0，尚无新基座开发集或测试集精度结论。

## 来源与恢复验收补充

每个节点输入带当前样本的 evidence inventory：原始音频 packet ID/SHA，以及实际可见知识 ID/URI/内容 SHA。P 和 Slot 只能引用当前输入列出的来源；未知引用按协议失败封存并保留费用，空引用合法。音频来源只标识完整原音频，不伪造时间戳。

候选准备脚本为已有 slot-contracts 供给启用原 native-candidate-slot-view-v1，增加 Slot consumer。视图从封存的 P 候选及当前分支绑定推导：保留全局指导，只去除与当前候选无关的 intent binding，原知识快照和音频不改。Slot 分支与 P response hash 不符时拒绝。

恢复准入从配置图推导预算：batched=3，per_candidate=K+2（K=1/2/3），不依赖载体名称或3/5预算白名单。每个模式与策略ID一致，角色访问次数/工具/分组配置仍由原配置验证器核验。已有响应与 FIT 复用；第二 Slot 未知时不重发。

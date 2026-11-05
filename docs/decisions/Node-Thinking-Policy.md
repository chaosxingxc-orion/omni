# 节点 Thinking 配置决策

## 已核验事实

2026-11-05 对当前两路 Fold A1080 运行树的真实 contact journal 和请求 payload 做了独立核验。Perception 和 Slot 的请求均使用 `enable_thinking=true`、`thinking_budget_tokens=3072`、`max_tokens=4096`。这与已确认的 Perception 关闭 thinking 约定不一致，因此旧运行只能作为配置偏差和工程链路证据，不能作为 P-off 精度基线。

## 首轮固定策略

新首轮候选将关闭 Perception thinking，并在请求 instruction 中明确：只返回配置的 JSON；输出 1–3 个按可信度排序的合法意图候选；每个候选只提供简短区分依据、已供给的证据 ID 和待确认字段；不输出思维过程，不提取槽位，不代替后续节点决策，不补造音频或知识中不存在的事实。保留 `max_tokens=4096`，避免同时改变输出预算。最终必须从实际 wire payload 验证 `enable_thinking=false`，包括 fallback 路径。

用户已确认两实例立即并行比较同一个三节点图，只切换 Slot thinking：

|实例|Perception|Slot|Decision（代码角色solver）|数据|
|---|---|---|---|---|
|A，4卡|off|off|on|原Fold A1080|
|B，4卡|off|on|on|原Fold A1080|

不再重跑新的联合Solver对照。历史A0完整结果用于同折横向参考；本次两臂用于归因Slot thinking效果。两臂保持相同P/Slot/Decision模板、数据名单、知识与检索、模型和评分实现，仅Slot thinking开关及必要运行身份/端口不同。Slot on沿用3072 reasoning+1024 answer，合计4096；Slot off总输出上限4096，不发送thinking budget。关闭后可用答案空间改变是此策略的组成部分，需在结果中说明。Decision继续thinking on。

Slot共享instruction要求按每个候选分别提取实体、只使用合法类型与有音频依据的值、不补造、允许空实体列表、仅返回既有JSON协议；不改变候选或代替最终Decision。比较槽位F1、官方SLU、Intent、strict、失败/截断率及节点耗时。原Fold A评估与调优后冻结候选，再用原Fold B1080确认。

旧 P-on 请求、响应和成本全部保留；P-off 新身份不跨配置重放旧响应。所有结果仍按原 Fold A1080 与 Fold B1080 横向比较，不新增开发集。

## 实施验证状态

两臂已生成实际基线配置。独立比较确认：模板与原Fold A1080相同，基线层差异为Slot thinking策略和FIT运行身份。两个部署的模型默认输出上限不同，但三个节点均显式配置；用实际生成的基线和模型配置解析后，P与Decision有效参数一致，Slot开关符合上表。此项是配置解析验证，最终adapter/wire及首真实请求验收尚未完成，新实验尚不能报告已启动。

旧任务停止准入后排空较慢：实际队列深度128、执行槽32；已准入但尚未首次调用模型的样本会继续等待并跑完整轨迹。该行为扩大配置切换等待时间。现阶段保留在途账本并安全排空。独立工程进程正在实现通用 `stop_policy: cancel_unstarted`：取消尚未首次提交模型调用的排队样本，已提交样本继续排空；默认策略保持兼容。修复仍在测试，尚未发布或用于当前运行。不能将停止后出现的P请求直接认定为新增样本准入，精确停止观察时间仍缺事件证据。

截至 2026-11-05 06:58 UTC，旧A任务已于06:53:32按 `AdmissionDrained` 退出，并正式核对376次生成全部有响应、服务运行/等待请求均为0。新A的配置化启动预检通过。两臂均通过实际生成配置、模板、模型解码参数到同一HTTP捕获请求的CPU回归；验证包含三个角色的thinking开关、temperature和seed。这仍不代表真实NPU首请求已验收。旧B仍在排空，结算等待器继续运行。

新两臂尚无经过核验的完成样本数、稳定秒/条或ETA。两路各4卡，服务存活不作为8卡满载证据。旧配置偏差实验的成本与轨迹全部保留，不并入新P-off精度基线。

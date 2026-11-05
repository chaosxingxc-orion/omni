# 节点 Thinking 配置决策

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

## 当前checkpoint进度（07:51 UTC覆盖先前快照）

|实验|终态样本/总数|完整三节点|拒绝/其他终态|资源|
|---|---:|---:|---|---|
|Slot off|58/1080|37|20拒绝、1有限输出|4卡、1个TP4、C32|
|Slot on|6/1080|0|6拒绝|4卡、1个TP4、C32|

两checkpoint均无撕裂尾部。Slot off启动至快照约2993秒，粗均摊51.6秒/终态；Slot on约949秒，粗158.1秒/终态。都含初始化与在途工作，尚不足以评估稳态效率或可靠ETA，也不能按此早期差值判thinking优劣。官方评分未产出，完整链路不等于预测正确。

首完整链审计脚本因把终态拒绝行当完整角色链而IndexError退出；生成进程仍运行。已交原工程owner修正只读分类与等待逻辑，保留失败证据，不重启推理。

## 配置与请求验证

两臂共享模板与原Fold A1080，temperature=0、seed=20260913。节点策略及模板已合main（bac92911）：12项边界、36项P策略回归以及两臂实际配置→HTTP捕获CPU验证通过。两个部署的模型默认输出上限虽不同，三个角色显式配置解析后P及Decision一致，Slot差异符合上表。

Slot off真实journal已核三个角色：P/Slot false且输出4096；Decision true、reasoning3072、总输出4096。Slot on首P已核原始请求摘要、false/4096/seed/temp，并存在响应；其Slot和Decision真实参数仍须独立验收。不能把CPU请求捕获当成全部真实推理验证。

## 旧配置与迁移边界

此前真实请求发现P错误开启thinking，已停止准入并保留全部轨迹与成本。旧A、旧B分别正式结算376、377次生成响应；更早B的62次调用单列，重放不重复计费。这些配置偏差运行不并入新P-off精度。

Slot off于07:01:49 UTC启动；Slot on首启动因评分源码__pycache__检查失败，零调用、未建attempt。禁止生成字节码配置修复后，07:35:54启动当前生成进程。

通用stop_policy: cancel_unstarted已合main（3d1750d8），57项相关CPU回归通过：取消尚未首次发模型的排队样本，已发请求排空。该功能尚未部署当前冻结运行。未知请求不重发，新科学配置不跨配置重放旧响应。

P重复候选、Slot漏覆盖和Decision修改实体分别归因，见Intent-Candidate-Grouping。当前两臂不热改。配对评分入口已通过完整历史1080自比较，待两新臂完整封存后产出正式结果；当前无精度收益结论。

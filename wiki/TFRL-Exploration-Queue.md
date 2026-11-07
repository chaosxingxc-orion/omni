# 实验队列与异步接续

更新：2026-11-07 09:16 UTC。三个教师后继已完成持久化配置，主线程核验脚本/配置SHA、前驱映射、无HOLD及真实binder进程。等待不等于生成；当前教师候选尚无本地结果。

|资源|当前实测任务|真实已配置后继|
|---|---|---|
|A，TP4|OFF C96，09:10 checkpoint897/1080|P thinking ON binder存活 → 历史P排序binder存活|
|B，TP4|ON K002，09:10 checkpoint489/625|K001恢复 → Reflexion Decision → Surface → Rescan → ACE|
|第三实例，TP4|P原worker零请求，已安全停止；缓存验证通过，新版本prepare中|B完整625首次交付因知识快照不符、缺缓存、绑定旧前驱退回；返工中，尚不可派发|
|CPU|Decision正式配对/独立coverage分析；175缓存交接；B175配置|与NPU异步，不占生成lease做输入准备|

B的Decision binder为waiting-container；Surface/Rescan/ACE均waiting-source，后者已有脱离SSH的进程。复用既有source_binding：等上游真实bound queue/container，再核验generation-released后派发；未预填假容器ID或release。四个教师候选科学父均为completeON，各项1080，共4320条待评估样本；资源顺序不代表叠加改动。

TF-GRPO必须等待同策略K2有效组内奖励；这项科学依赖不阻挡已准备的独立Reflexion/ACE。CPU评分不阻挡资源后继。输入和知识复用，不重复采样、未知请求不盲重发；600秒或明确终态观察。

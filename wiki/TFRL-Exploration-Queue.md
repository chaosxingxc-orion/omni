# 实验队列与异步接续

更新：2026-11-07 08:35 UTC。三服务可用，两路真实生成；第三路0业务调用。详见[状态与计划](TFRL-Current-Status-and-Next-Plan)。

|资源|当前实测任务|接续状态|
|---|---|---|
|A，TP4|OFF C96，471/1080|既有P thinking ON后继待真实handoff；Decision1080生成已完，CPU评分独立恢复|
|B，TP4|ON K002，352/625|K001恢复已配置等待K002 release；Reflexion Decision binder等待K001 release|
|第三实例，TP4|P任务0/1080；context准备约131秒/次|B175完整625由Codex子进程准备，尚无完整文件交付，不能称已入队|
|CPU|结果评分、教师反馈、FIT配置验证、第三实例阻塞诊断|与NPU异步，不占生成lease做输入准备|

surface、rescan、ACE三个教师候选均已准入，但最新交付明确held-before-queue；补齐真正队列是下一步。四个教师候选各自独立继承同一基线，资源执行顺序不代表叠加策略。

只复用configured_job_queue及现有auto-bind/handoff；以真实generation-released交接，不等待CPU评分。TF-GRPO的组内奖励必须依赖K2有效轨迹，这是科学依赖；教师单节点候选无需等待无关评分。固定Fold A/B1080与construction625复用，未知请求不盲重发。观察600秒或终态。

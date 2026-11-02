# Grounded synthesis 独立对抗性技术审查

审查日期：2026-11-02。范围：公开工作树的 grounded_synthesis_plan/source/execution.py、tfgrpo_runtime.py、关联学习器和状态页。只读 CPU 静态审查；未 SSH、未执行正式测试、未生成模型调用。该独立子任务通过主线程以 gpt-6-astra 模型配置分派；审查结论以实际代码证据为依据。

结论：来源与历史账本、整组确定性分批、已知响应复用已有实质实现；目前尚不能从共享生产入口运行 grounded shard，更不能把 shards 当作最终候选。先修运行接线与实际请求 FIT，再运行有限的已注册 shard 工作；最终综合协议未完成不必阻断明确标为中间产物的 shard，但必须阻断候选发布与精度结论。

## 实际执行前 P0

### P0-1：共享生产入口不支持新 stage，执行器目前只是库接口

证据：grounded_synthesis_execution.py 的 STAGE 为 grounded_shard_update，registration_binding 返回 saea-grounded-shard-execution-v1；tfgrpo_runtime.py 的 STAGES 仅允许 summary/advantage/group_update/batch_update，validate_registration 要求另一套精确 schema 与 sample_id 联系名册。run_registered_learning 始终调用 learner.learn，没有 execute_shards 分支。仅添加 stage 白名单不能完成接入：batch hash、roster anchor、runtime identity、decoding cap、custody pin 和 plan hash 必须贯通。

修复：在独立后继执行树接入明确的 grounded 模式，沿用 FlightRuntime admission、共享 lease、MeteredExperienceCaller 与封存路径；构造不可变 runtime registration 将 shard plan 与 source custody 绑定。不得绕过共享 client 直接给 execute_shards 一个网络回调。验收必须含正式入口到 prepare/FIT/intent/response/cost/seal 的固定镜像集成测试，以及旧普通学习路径回归。现有执行测试使用 mock caller 和 lambda preflight，只证明本地状态机。

### P0-2：原始 messages 的模板计数不等于实际 wire payload FIT

证据：plan._messages/plan_batches 对两条原始 messages 计数；MeteredExperienceCaller.prepare 把它们放入 model_context/source_messages，再经 states_from_contexts、PREPARED_TEXT_INSTRUCTION、contact_instruction 和 client.prepare_request 包装。该实际请求有额外文本和序列化开销；1,024 token 余量不是证明。execute_shards 只检查 learner.preflight 可调用，无法自行证明它是原生 gate，也不核对回调返回 receipt；生产安全依赖接线。

修复：对正式 prepare 后的完整 payload 用原生 build_tfgrpo_fit_gate 验证，并绑定实际 tokenizer、服务 context、输出及 reasoning 参数。任何 shard 超限应在写 intent 之前失败，CPU 重新规划而非截断。测试覆盖 mock FIT 被生产工厂拒绝、cap 不一致、包装后超限、无模型调用。三个原始模板计数不能直接宣传已具备生产可运行性。

### P0-3：新执行路径的恢复必须贯通 native journal，不能把本地成功重放当跨 attempt 恢复

证据：execute_shards 在新工作前先处理所有已有 intent；ExperienceLearner._call 对未知结果只 reconcile，不重发，这一点正确。tfgrpo_runtime.run_registered_learning 也拒绝给已有 receipt intent 新建 client。但 grounded 模式尚无相应生产入口，故当前没有它在新 attempt 下沿 NativeContactReplay/成本封存恢复的证据。

修复：在接线时保留原有“旧进程退出/lease 合法交接/请求来源核验”的恢复协调器，验证三个故障窗：native response 已落盘而 learner response 未落盘；intent 已落盘但 outcome 未知；响应已知但解析或便捷进度文件失败。已知原始响应和全部已花成本保留；未知不重发。停止派发后 drain 已发送 peers，不能把它们从预算抹除。此项是接入验收，不能据此宣称现有 _call 已有盲重发漏洞。

## 候选发布前必须完成，允许在标明中间产物时先执行 shard

### P1-1：跨批最终综合尚未实现，局部优选不保证全局一致

证据：registration_binding 和 execute_shards 明确写 final_consolidation=NOT_IMPLEMENTED、candidate_publishable=False。每批各自收到同一 parent 并输出 ADD/UPDATE/DELETE；不同批可能更新或删除同一旧规则，也可能产生重复、相反适用条件。将操作直接拼接后按顺序应用，会引入源顺序偏差。

修复：先定义有界独立最终 job，输入包括每条 shard 操作的原始 proposal/group 来源、适用条件、冲突边以及可追溯支持；不能只合并已压缩的自由文本。确定冲突处理和总 item/character bounds；保留拒绝/不确定结论。为最终请求单独计数、FIT、注册预算和成本。测试跨批同 id 修改/删除、语义冲突、重复来源与最终超限。没有最终封存候选前不得运行声称“新方法”的候选评估。

### P1-2：有损证据不足以核验具体错误归因；来源 hash 不等于模型看到证据

证据：tfgrpo_learning.grounded_group_evidence 保留 perception history/reference 与 reward/TP/FP/FN/answer SHA，省略检索上下文，也没有提供各 rollout 的完整预测答案。后者尤其影响判断某个槽位操作是否解决了原错误。build_final_messages 虽要求不能推断省略证据不存在，但自然语言约束并非真实性验证。execution.execute_one 只 parse_operations，未结构验证适用条件/支持样本/反例。

修复：在预算内优先保留原始预测 intent/slots 和参考的最小语义差异、必要的检索片段及支持来源；若仍省略，标成不可验证归因。为规则输出增加可机器检查的支持引用、作用域与冲突记录；验证引用来自本批实际证据，并对关键归因做 H1 原始证据核验。不得以 hash 代替语义支持。保留当前有损投影作为显式版本化实验，不能宣称完整轨迹学习。

## 算法与研究建议（不作为持续有效 NPU 队列的阻断项）

1. 与 Training-Free GRPO 的对应：现有链是冻结参数下的轨迹摘要→语义优势→分组经验操作→经验综合→Solver 消费；新的 grounded shard 是对最后综合阶段的上下文工程改造，不是新的组相对奖励估计器，也不是参数 GRPO。单轮候选生成不能证明 RSI；必须记录多个冻结协议轮次、父子经验身份、配对性能与累计学习/评估成本。本次静态审查未独立阅读论文全文，不认证论文逐项复现。
2. 有效信号：状态页说明 174 个全错组中仅 51 个有部分奖励差异，123 个同分。组内差异提供相对信号，但最高 reward 仍可能是错答案；不能把它当正确示范。相同 Perception 下多个 Solver rollout 只反映条件推理差异，无法区分音频感知瓶颈。保留同分组用于诊断，按错误来源决定是否需要新感知/可验证证据，不应仅重复相同随机采样消耗预算。
3. 难例与覆盖：625 条为多样性固定集，并非统计确认 hardset；51 个训练组还是奖励可分性选择后的子集。报告应区分难度、奖励可分性和语义覆盖。按原始意图/槽位实际关系统计，不能笛卡尔积凑分母。补测先确定预算与选择规则，避免把模型碰巧失败当稳定难例。
4. H2 反复开发选择已污染“独立验证”的解释。后续用冻结候选、稳定名单、配对 bootstrap/置信区间和回归门槛评估，明确开发集用途；完整 13,078 test 不参与调优。构建与消费改动分别对照。
5. 309 历史账本：source.load_source 将 FINALIZED 成功综合作为新实验来源，保留其 current usage 加递归历史；native_source_usage 按 run/attempt 去重并与总额相等。此逻辑方向正确，未发现把第 309 次成功调用改成 ABORTED 的做法。仍须在实际产物核对 309、6,032,239 input、761,689 output；本审查未访问私有 native 产物，不能独立复核这些数值。H2 的 936 调用另列评估成本，不混称全项目账本。

## 推荐交付顺序

先完成 P0 接线、最终 payload FIT 和恢复故障窗测试；后继 job 在独立绑定树准备并通过共享 guard 安全交接，继续使用当前队列的有价值负载，勿为等 CPU 文档停服务。随后运行有明确 3 次预算的 shard 中间任务并封存实际成本；CPU 同时完成最终综合协议与来源/冲突核验。最终候选封存后再做冻结配对精度评估。LoRA 生产训练继续暂缓。

审查边界：没有修改源代码，没有执行固定镜像测试，没有 SSH 或改变队列。上述 P0 是 grounded 新路径真实执行的技术条件，不是新增人工审批流程。状态文档应区分“源码实现”“固定镜像测试”“真实 job 注册”“原生 FIT”“真实生成”“最终候选”“效果评估”，避免将其中任一层当成后续层的完成证据。

# GEPA 当前父子同 P 比较与真实反思接线

核查公开树：`a7a87c25`（2026-11-03）。本次仅 CPU 代码审查，未 SSH、未修改产品代码、未运行模型；以下均为实施建议，不能视为端到端验收。

## 建议：先保留 lock，新增显式四调用父子图

最小改动方案是沿用原生 Solver template variants：一次 P 后执行 `parent / child / lock` 三个 S，只有原始 `lock` deliver。正常 **4N** 调用，父子使用同一份 P、同一知识与供应；评分明确比较 `parent` 与 `child`，原始 lock 仅作参照和维持现有交付语义。保留 lock 多花 N 次，但避免先改原生 delivering 模板语义，也避免借恢复机制跨候选注入 P。四调用图验证后，再依据实测决定是否专门实现三调用 `P→parent/child`。

代码依据：`gepa_prompt_material.compile_template_probe` 只发布一个 `gepa` 列；`consumption_module._compile_template_columns` 已能生成多个模板列，但最后固定追加原始 delivering lock。`consumption_module.py` 的 legacy prompt gate 目前只准入单列 `gepa`；`native_recovery.gepa_probe_layout_matches` 也严格要求 c3、`gepa/lock`。所以直接加两列不会合法运行，必须新增有明确 schema 的父子模式，不能放宽所有 c4 图。

新编译入口建议 `compile_template_pair`：分别核验 parent、child 的完整 lineage 与实际模板 bytes，继承冻结知识/P/model/decoding/schema/supply；只允许 `parent,child,lock` 顺序、review off、无工具/分支/重听、P与各S一次。compiled policy 为 c4；program ceiling、native FIT、预算均来自实际图。已有16条原模板 seed probe仍保持P3072/S3072，不能原位改成新模式；下一项新搜索显式绑定P-off/S3072的新契约。

## 状态机必须区分诊断父评分和配对父评分

现有 `GepaSearchState._apply_result` 在 `parent_eval` 保存 `active.parent_scores`，然后 `child_eval` 只接收child分数并直接与旧parent分数比较。仅让新的child run顺带生成parent仍不够，状态机目前不会使用新parent分数。

建议版本化状态 schema，保留旧状态只读：

1. `parent_eval`：在固定feedback batch运行当前parent，保存官方分数、完整轨迹与来源摘要，供反思。沿用单列机制时为P/parent/lock，共3N。其分数称 `diagnostic_parent_scores`；全正确可按既定规则跳过该轮，不宣称全分布正确。
2. `mutate`：只读这批feedback的parent轨迹和真实参考，生成一个受限Solver instruction mutation。所有输出先封存、记账，再验证。
3. `paired_eval`：候选生成后，使用新的共同P运行parent/child/lock；同一ordered batch，4N调用。接受比较只使用本次两臂分数。旧诊断P与新配对P可能不同，报告这一事实；旧parent分数不得进入接受判定。
4. 严格同批改善后才进入完整80条`selection_eval`；完成全名单才入池。最终选择仍是开发选择，不能凭一次随机得分称稳定收益。

这保持GEPA的反馈反思、同批筛查和完整选择集入池，但增加一次parent重评，是本地适配，需在算法文档说明。推荐新增`paired_eval`而非暗改旧`child_eval`结果含义。观察结果应包含准确parent/child身份、ordered IDs、两臂scores及配对执行证据摘要；adapter核验每个样本两臂确实绑定同一P产物、供应摘要和官方score来源后，才允许state.observe。仅样本ID相同不足以证明同P。

## 真实 reflector：复用受控文本调用，不另造聊天客户端

新增小型 `gepa_reflector_runtime`，复用现有 `prepared-context-text` 的注册、共享Flight、native FIT、原生响应封存与恢复框架。当前 `native_contact_custody_enabled` 对该路径只白名单TFGRPO/grounded/ACE/Reflexion prepared schemas，需要增加严格的GEPA注册/源验证分支。旧`stage3_gepa_material`面向历史分组知识delta，不是当前Solver prompt search，可复用来源核验思想，不能直接把它称为本搜索reflector。

反思输入绑定：当前parent模板、契约、feedback batch ID顺序、封存P/parent回答、官方逐条反馈、TRAIN参考、知识/供应来源与请求预算。仅载入feedback；80条selection与H2/test的参考和错误不得进入文本。供应原文与当前parent失败都是真实上下文，不能拿模型自评分替代官方反馈，也不能要求生成任意知识库更新。

输出复用 `solver_prompt_candidate.validate_mutation` 的六个字段：`schema, parent_sha256, module, evidence_query_ids, diagnosis, solver_instruction`。现有验证只要求evidence属于整个feedback集合，adapter还需要求它属于本次已提供batch，并核对mutation来源就是该stage的原生输出。约束外字段、重复JSON键、截断、非有限值、错误parent和越界证据都拒绝；原始输出和失败费用保留，不用静默修JSON或无预算重试掩盖协议失败。语义审查继续拒绝逐题答案硬编码与不受来源支持的泛化。

## 成本与调度

固定16条机制probe：48正常调用，与动态搜索分开计账。

上述最小搜索路径，每轮feedback batch N：parent诊断3N + reflector 1 + paired4N，即 **7N+1**。若完整selection仍复用候选/lock单列图，另加3M（M=80，即240）；seed的selection评估也另记3M。N=16时一轮诊断、反思、配对为113次正常调用，进入selection共353次；这些是图预算，不是实测完成量或ETA。每个图另登记fallback上限、FIT、音频和卡时；reflector的重试也必须有显式注册额度。

`GepaSearchState`当前最低calls校验只是一条样本一调用，不是实际成本模型。native adapter必须按完整图预留calls、prompt/completion token和Flight资源；纳入P、lock、parent重评、反思及失败调用。继承轨迹用inherited_usage，避免重新付费也避免抹去历史费用。pending stage写入后不能重复dispatch；先核验已有响应再observe，unknown保留阻断。

CPU评分/反思材料准备与可独立执行的NPU任务异步；parent反馈尚未产生时，不能假称依赖它的mutation已经READY。后继按固定配置发布，终态/600秒监测，全量测试不进入队列。

## 选择集的额外限制

候选各自fresh P的selection分数仍含感知随机性，四调用feedback配对没有解决跨轮候选池的这一问题。第一版可以明确披露并按预注册预算执行，不能描述为全程同P。最终赢家与参照必须再做同P配对开发回归；若噪声影响排名，再单独设计来源严格的稳定P缓存/导入，而不是放宽`NativeContactReplay`。反复selection属于开发优化，H2也已有开发选择接触；保留未调优完整测试，但现在暂不运行。

## 必要功能测试与交付顺序

1. 编译/准入：真实A0编译四调用图；parent不是原始lock时仍准确核验模板；错顺序、额外臂、deliver变化、P/知识漂移、工具/review均拒绝；旧c3机制和A0恢复回归不变。
2. 原生来源：一次P绑定三个S；模拟在任一S后断开，已知响应恢复不重发；unknown阻断；不能将别的候选或batch响应导入。
3. 状态机：构造旧诊断parent得分低、新配对parent得分高的例子，证明child不能靠旧父分数虚假晋升；ties拒绝、no-op拒绝、名单完整、身份与证据hash漂移拒绝。
4. 反思：实际request/FIT/封存来源与六字段mutation往返；selection/H2标签、batch外证据、截断与错parent拒绝，费用仍完整记账。
5. 评分和预算：全部三S臂完整导出、官方逐条奖励与pooled SLU分开；4N/FIT/fallback/反思全部计费，旧继承成本不重复；预算不足时整阶段不启动。
6. 固定镜像通过后，先在固定feedback小batch实际走parent→reflector→pair；通过才跑80条selection。报告真实候选、两臂结果和成本，随后冻结候选做H2回归，不扩大到全量测试。

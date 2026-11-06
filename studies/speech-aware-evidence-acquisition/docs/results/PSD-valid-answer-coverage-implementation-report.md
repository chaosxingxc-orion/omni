# PSD 有效回答覆盖：辅助引用隔离实现

基于 main `48cb0df4292affd27770fecc5662a5d0540ba14f` 的独立分支，旧生产 `9a50852ef67b964ed7571bb1b08deeb78b2251ce` 未改。实现一项配置化修复，**尚未证明全样本100%有效回答**；未做真实Fold推理、模型、教师或真实FIT调用，未部署、改运行服务或触碰队列租约。

## 根因及范围

已有真实封存诊断显示，seq25 P包含正确首候选，另一合法候选引用了未在本次请求供给的知识ID，整个P拒绝；seq1 Slot将合法类型名字误作未供给合同ID，导致整个Slot拒绝。合法label、全局存在的卡与该请求实际供给的来源是三种不同概念。seq1 P此前已丢失正确候选，清洗不能恢复它或保证实体语义正确。原始响应、固定知识、系统提示、业务字段和候选集仍是唯一来源。

旧9a无main `dd92c65d` 的Slot缺项共享预算补问与已知终态后继能力。main仍对未供给辅助引用整体拒绝；本修改仅面向原有 P→Slot→Decision，未替换Fold A原P→Solver主基线。既有两个reference prompt已由 `ade951ee` 登记，本次未重复准备或改写。

## 配置与语义

NodeWorkload 的 `pipeline` 可显式加入 `"reference_normalizer":"quarantine-unsupplied-v1"`；可与既有 `coverage_recovery` 和 `answer_only_fallback` 同时配置。未配置保持旧行为。新工作负载须独立配置、身份、源码和预算，不能热改在跑release。图配置内容及hash记录在receipt；原生source核验其与实际composed配置一致。

仅在已返回、正常stop的P和Slot响应中，逐个对当前请求inventory校验引用。已供给ID保留，唯一合法别名解析至实际ID；不在inventory或别名歧义的引用从派生视图移除，逐项记录原值、行与位置及 `quarantined`；绝不将其标为供给来源。原始响应、原始响应hash、请求、inventory、合法映射和清洗后hash均保留。SDK交接状态使用清洗视图；正式bundle重算、候选handoff与最终答案重建从原始响应和实际供给重新计算并核对。保留所有业务字段、合法候选、实体、重复实体和不确定项；不是语义正确性证明。

非法JSON、结构、必填项、非法类别、缺失候选、length及未知响应不由该normalizer修复。Slot缺项或无效项可使用原有同attempt有界补问，共享answer-only预算，仍无效时如实失败；length使用既有有界fallback条件。P结构错误、非空length、反复失败、语义错误、上游遗漏候选、服务故障及UNKNOWN仍需后续独立、来源绑定且预算明确的恢复方案，不能用空答案、默认label、随意空实体或未知请求重发补数。当前清洗策略下跨attempt Slot后继因checkpoint祖先清洗重算尚未实现而明确拒绝；旧策略的后继不受影响。100%需先逐类核实真实请求终态，再对可恢复类使用有界真实模型响应、逐次计费和完整来源，对不可恢复类显式报告；目前没有实现或验证可保证100%的路径。

## 测试及pin

通过 `ssh npu2`，指定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`，容器 `--network none --cpus 4 --read-only`，源码只读挂载，临时文件在tmpfs。命令：

```sh
PYTHONPATH=/src/src:/src/scripts/analysis PYTHONDONTWRITEBYTECODE=1 \
/opt/saea-venv/bin/python -m pytest -q --tb=short --show-capture=no -p no:cacheprovider \
/src/tests/unit/test_intent_slot_runner.py /src/tests/unit/test_intent_slot_pipeline.py \
/src/tests/unit/test_intent_slot_workload.py /src/tests/unit/test_native_episode_source.py --basetemp=/tmp/pytest
```

结果 **113 passed，5项依赖弃用警告，34.44秒**。真实OpenJiuwen SDK、真实正式canonical output/trace及handoff verifier；模型传输、音频和tokenizer为测试桩，并非seq1/25原始响应重放或真实采集。脚本构造两类同机制未供给refs，P→Slot→Decision保留模型候选和Slot实体，含同attempt缺候选补问、answer-only fallback、错误必填字段重试成功/失败、旧严格拒绝、bundle篡改及别名歧义。已有UNKNOWN/恢复和旧策略回归包含在相关集，但未声称本策略跨attempt支持。初次相关集暴露旧v1兼容回归，修复后重跑通过。

真实来源 pin：原因报告 `594d2566f0adb3b5fccdac5623f76b9e75c62aeac7fdd07ae8f321e35a050f72`，结果 `9acf557702403578f73f4e4a5a3ff0742cc024e903e6d9ce800f50f62831e999`；恢复报告 `c33942d521ec56c324975066f7e436a8668d4736f28d45bfc473fbe190e090c1`；Sol固定快照摘要 `cd7174f44ae168b1a76acb04d49216d6800bf51b893151c9f4f1f28df1ccdc53`。原诊断快照 `27e75cf2a625657172da37fbedc326b5224cfe47657a200f758c49e1f7766289`，Fold A dataset `4ba55688e80d8d768ec6cfc3ff4bacc9802fb34e4dab54913934015184497215`，roster `1fe7908c31396774d7535fdf37bd791762215c324999d66ed7beb9e7b2f9033d`。测试源码逐文件摘要见同目录 result.json；本地与远端测试副本的9项修改文件hash逐一一致。

## 负载与后续验证

Sol 15:48固定快照：OFF 749/1080完成、715结构有效（95.46%）；ON 425/1080、406结构有效（95.53%）；完成分母不是全Fold，结构有效不等于正确。53个空交付中32为length、21为seal，尚无下游Solver答案；在途/UNKNOWN不得当失败重发。7个稀疏硬件时点AI Core均值约48.36%/58.82%，running约55–64/63–64，waiting 0–5/0；ON Slot输出和reasoning负载明显更长。这不足以认定CPU或NPU单点瓶颈，不能把采样均值外推为连续利用率。

后续以PSD为主流程，以独立身份在相同完整1080名单配对候选，并对齐已封存的Fold A P→Solver历史结果口径，不重跑旧图，统计全分母结构有效率、最终官方正确率、P候选覆盖、Slot实体、隔离引用分布与语义支持、已知终态、每节点tokens、实际calls/FIT、秒/有效条及全部成本。重点调查P 4096截断空白退化；已注册参考提示可比较，不能据两个样本外推收益。最终独立TEST须完整13078条且不参与调优。当前无新有效率、时延、8卡利用率或货币成本测量。

预审修正：P grouping有意保留原始候选与原refs，供原始来源审计；进入Slot的派生候选从清洗结果重建，SDK→canonical output→bundle已覆盖。v1 strict读取已由独立回归验证。恢复runner从真实completion、当前inventory及配置重新计算派生内容并与待封存状态比较；新增篡改回归，保持已知无效后继行为。固定镜像同组重跑 **114 passed、5 warnings、36.93秒**；结构有效统计不代表标签合法或语义正确。本分支尚待主线程合并，未在线部署。

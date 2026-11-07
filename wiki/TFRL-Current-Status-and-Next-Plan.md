# TFRL 当前任务状态与下一阶段计划

更新：2026-11-07 11:10 UTC。09:10/09:20的“两个实例正常运行”已失效；本页原位替换旧状态。

## 当前结论

尚未实现持续满载或稳定TFRL收益。OFF C96已完整生成1080条，CPU评分失败；B K002已完成生成阶段但封存失败。175缓存修复后真实完成320条，之后xgrammar HTTP500导致终止。主线程已处理直接配置问题、释放已核验无未知请求的B租约，并将独立教师候选提前，避免无关K2恢复阻挡算法探索。

## 当前实验与资源

|资源|目的及设置|已确认结果|当前执行状态|
|---|---|---|---|
|A，4卡TP4|OFF C96；Fold A1080；P OFF/Slot OFF/D ON|1080/1080生成封存，3263calls；输入34,910,769、输出2,372,805tokens；生成至release6047秒，5.60秒/条|已释放NPU，CPU官方评分失败待恢复；P ON接续先失败于timing目录、后失败于E0映射，独立执行者已配置真实E0重新启动，尚待业务响应验收|
|B，4卡TP4|construction625 MINUS-R01 ON，K002 seed1002/temp0.7|已生成完整outputs，1887calls全部observed；ABORTED源保留，checkpoint完整性需正式reader终态复核|旧release多出一个宿主pyc，源码7628文件哈希均未变；干净副本与安全结算完成。恢复在PSD归一化策略预检失败，无新恢复attempt；暂由独立教师候选优先使用B|
|第三路，4卡TP4|旧P Fold A1080，C64|320/1080完整checkpoint，964calls/963responses/1unknown；输出tokens下界1,461,985|xgrammar拒绝token后HTTP500、worker退出；未知请求保全、终态结算与配置修复由原Codex会话继续，不盲重发|

服务配置仍为Qwen3-Omni30B-A3B BF16、TP4、ctx65536、max_num_seqs128、max_num_batched_tokens2048。资源已部署不等于持续工作；当前无80%持续AI Core证据。对停止/准备中的任务不报虚构ETA。

## 直接阻塞的根因与处理

1. PON日志目录与E0路径缺失：属于外部运行配置错误。保留两次失败记录，复用科学配置和已准入输入，独立后继修复，禁止重做数据集。PON执行者同时更新后续P历史排序任务的真实绑定。
2. K002发布校验：唯一额外文件为configured_job_queue.cpython-310.pyc。已复制清单内同哈希文件到独立只读发布目录，不修改原失败证据；原生工具核验1887响应全部已知后结算旧租约。contact恢复仍被normalized/length-recovery策略拒绝，需要验证规范化Slot出处、原始contacts与FIT账本；不能删除守卫冒充兼容。
3. 175：tokenizer缓存阻塞已解决并产生320真实结果；新故障是xgrammar HTTP500及一个未记response请求。替代解码后端只通过CPU schema解析，不提前宣称NPU兼容。原执行者继续处理，不以失联为由重复推理。
4. 教师候选：独立Reflexion Decision已真正尝试启动，但模型前因workspace source identity不一致退出。根因是准入在Git目录产生source_manifest_sha256=null，而生成使用固定release的非空hash。主线程已改用同一固定release、保留既有输入重新执行必要准入，首次重准入发现旧coordination输出冲突，保留旧记录并使用独立coordination目录，容器teacher-psd-admit-release-decision-v6正在运行。其余同源候选也需一致处理。此前“已admit”不足以证明能实际生成。

## 完整质量结果

固定Fold A1080：completeON Intent84.0741%、SLU70.9703%、Strict587、合法1077；紧凑P为84.5370%、71.5145%、595、1080。紧凑P保留，质量配对区间跨零。紧凑Slot有实体Strict−7、无实体+20，暂不晋升。

Decision条件证据正式配对已完成：Intent+0.8333pp，95%CI[-0.2778,1.9444]；Strict+0.7407pp，CI[-1.0185,2.5926]；SLU−0.4224pp，CI[-1.8017,0.9497]。修对/改错分别24/15与55/47，收益集中无实体组，独立上游与路由混杂使其不能单独归因于Decision。详见[完整配对结果](PSD-Decision-Official-Paired-Result)。

旧P→Solver TEST13078的Intent86.3894%、SLU74.2334%只作历史参考；新PSD没有正在运行的完整TEST，也未证明稳定超过旧图。合法回答覆盖和正确性分别报告。

## 当前任务图与分工

NPU：A PON修复激活→P历史排序；B独立Reflexion Decision→Surface→Rescan→ACE（每项1080，当前业务生成均0）；175 P安全恢复/资源交接→正确MINUS-R01 B625。K001/K002恢复为TF-GRPO学习前置，不再作为独立Reflexion/ACE的科学依赖。B独立队列启动因准入来源问题失败，固定release准入完成后更新唯一启动配置与后继链。

CPU三个Codex子进程：PON激活、175未知请求/xgrammar恢复、PSD规范化恢复及学习输入。第三项首次只交费用helper，已退回补齐runtime及exporter，不能由主线程兜底或算完成。175 B配置返工已交付正确知识625、缓存与持久等待器，但仍需P负责人提供安全恢复实际描述才能接续。

后续按单节点比较Intent/SLU/Strict/合法覆盖、修对改错、实体分层和成本。GEPA与TF-GRPO使用各自模板，必须基于新PSD真实来源；历史ACE/Reflexion策略迁移还不等于PSD原生学习闭环。FIT并发8/16/32固定镜像39测试通过，真实收益尚未验证，不热改活动科学配置。

## 需要用户决策

无新增阻塞性决策。继续处理直接功能缺陷、100%有效回答与效率；不扩展泛化加固。completeON仍为工作比较基线，保留紧凑P；暂不晋升紧凑Slot与条件Decision，LoRA后置。固定Fold A/B复用，冻结后完整TEST13078，不用TEST调优。

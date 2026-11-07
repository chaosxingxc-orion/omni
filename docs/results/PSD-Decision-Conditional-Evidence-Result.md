# Decision条件证据候选：完整Fold A结果

本页记录人工配置的Decision提示对照，不是后续远程Reflexion Decision候选。2026-11-07只读核验：attempt FINALIZED，独立score handoff为score-ready；复用现有评分结果，没有重推或重评。

|指标|completeON父|Decision条件证据候选|差异|
|---|---:|---:|---:|
|样本|1080|1080|相同完整分母|
|Intent|84.0741%（908）|84.9074%（917）|+0.8333pp（+9）|
|SLU-F1|70.9703%|70.5480%|-0.4224pp|
|Strict|587|595|+8|
|官方逐样本parsed|父独立合法覆盖1077|候选parsed1076|口径需coverage reader进一步对齐|

Strict从官方per_sample逐行计算：parsed、意图正确、精确实体FP=0且FN=0，同时包含合法无实体样本。官方与port身份校验通过，SLU和span差异均0。配对bootstrap与独立coverage reader尚未执行，本次不称统计显著、不把parsed代替独立合法回答覆盖。

结论：意图改善没有带来综合SLU改善，暂不晋升。由于P/Slot是独立生成且父与候选运行在不同实例，本结果不能把全部差异归因于Decision提示。下一步复用现有comparison配置完成同样本修对/改错，区分有实体/无实体及上下游变化；不重复生成1080条。

生成成本：3260次调用，35,069,077输入tokens，4,861,246输出tokens，9315.90125音频秒。完整scores SHA256为01f9432a4945079507cd2b46676818bf6e82875becb55f512d78f8a390451857，原始轨迹保留私有。generation已释放，A正在OFF C96效率对照；CPU分析不阻挡其后继。

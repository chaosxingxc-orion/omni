# Incremental fixed-K insertion screening

`IncrementalTopK` uses frozen score arrays, source groups, per-query excluded groups, legal labels and fixed description ranks. Call `reset(current_source_orders)` after each accepted insertion, then `trial(bank_index)` to screen real TRAIN insertions. `primary_metric` is `question` by default; `union` is also supported. Acceptance and candidate proposal stay in the caller. It does not encode data or call models.

The retained K+1 prefix gives exact question/union hit deltas, label-occurrence changes and repair margins. Score ties use original index ascending. Source exclusion and source deduplication precede TopK; legal label filtering follows it. `verify_reference` compares the prefixes to an existing `source_order/retrieve` implementation and checks aliases, ties and legal masks. Measured scope and limits: [first-pass results](../../docs/results/Recall-Minimum-Question-First-Pass.md).


## 有界因果补偿搜索

causal_depth.py 的 frontier 是已有评分器调用的搜索生成器，policy_contract.py 负责可执行策略校验。模块不读取数据、不编码、不调用模型，也不接管最终接受门槛。现有执行器提供 losses、repairs、lost_classes、successors、test、structural 回调；全局/每起点预算、持久账本及恢复继续由执行器负责。

causal_depth.schema.json 列出动作顺序、层预算、beam、深度、有界非单调和少量起点扩展新增额度的字段。条件纯加下界只作诊断与删除/替换优先排序，不剪掉仍可能先加后删的路径。种子成员允许替换；最终候选仍需通过固定K、来源和逐例稳定性验收。

执行 python tools/recall/test_causal_depth.py 可检查删除后深层新增、原子替换、非单调界、beam/深度及6新增开关的因果夹具。它不是真实样本实验，不能用于性能或召回收益宣称。生产代码与固定镜像验证版本逐字一致；公开夹具仅替换导入路径并移除部署环境guard，远端正式执行器保留原guard。

本轮真实缓存工程验证：4个起点、14个不同状态，删除/替换/两层路径可达；连续与第9状态中断恢复的账本及科学结果相同。0接受，未证明新策略收益。完整研究结果与限制见最小种子结果页。

## 候选发现与最小集回补

candidate_proposals.py 提供冻结分数下的TRAIN候选生成、单插支持/风险边界和确定性有限池选择。proposal-plugin.schema.json 支持旧TopN或全合格TRAIN生成，similarity、incremental_cover、class_round_robin、risk_aware_cover四种排序；默认历史路径保持。来源闭包排除先于检索，非法标签仍占用原始TopK位置。

risk_aware_cover优先安全正收益候选，再按风险事件数、增量覆盖等排序。支持并集不是组合收益：正式接受必须交给实际检索评分器。零支持零风险代表可能先于有风险的正收益代表，这一取舍需要实测反馈。

policy_contract.py 支持显式seed输入与单探索输出schema；gain_or_shrink表示逐例零退且Q15修复，或库严格小于当前接受状态。最终gate和计费在调用方执行，模块本身不保证缩库。frontier的expand_loss_free开关允许继续探索无损状态，调用方仍控制同类删除、深度和非单调额度。历史默认不变；不声称全局最小。

新增功能已在固定镜像通过6项定向验证，其中真实TRAIN单插反事实修复14条Q15且四组零回退；该反事实未晋升库，也不是新教师策略收益。全域profile的候选-query比较量及CPU时间须单列，不能因没有调用Replay就记成零成本。

# S节点意图条件槽位范围审查

结论：槽位类型范围应在模型调用前锁定。当前实现已经锁定枚举，但语义规则没有充分按意图收敛；知识卡与示例冲突属于知识构建问题，不能先归咎于OFF模型。OFF继续作为用户指定基线。

## 当前545样本的实际范围

审查全部1635个真实请求：每个请求只有一个候选意图，allowed_types、该意图知识卡类型和JSON输出enum一致；异意图知识卡0、示例越界0。545个正确意图分支的金标槽位类型均在允许集合内，缺失类型样本0；实际已解析预测越界0。该结论针对当前固定集合，不证明整个任务的知识覆盖完整。

cooking_recipe现有类型是app_name、business_name、cooking_type、date、drink_type、food_type、ingredient。这些关系均有原始TRAIN标注支持，不能只因某个槽少见就删掉。alarm_set有9个类型、datetime_query有6个类型：合法集合不是每次必须填满的表单。

## 真正的知识范围缺口

prepare.py的project_rows只裁剪intent_bindings，保留其余extraction_cues、admissible_formats、missing_behavior、ambiguity_behavior、limitations原样。输出卡虽然冠以单意图名称，正文仍可能带着其他意图的线索与通用规则。比如cooking_recipe的food_type卡仍说明“餐馆前的菜系形容词”以及“餐馆名词归business_type”，但该意图输出枚举根本没有business_type。这是文本语义范围不一致，不是模型跨意图选卡。

真实例子：
- 音量调整：卡称模糊量a bit仍是数量；实际few-shot“turn it up a little”却标空。先审标注与规则冲突。
- 日期查询：相似示例third friday拆成friday/this month，而third monday保留完整长短语。当前query的last friday of the month被OFF缩成month，不能只断言模型不懂长短语。
- 烹饪：drink_type列water为线索，ingredient又说与食物边界不一致；water to boil当前被OFF填成drink_type并漏掉boil。需要明确当前意图下的角色区分。

## 来源复核结论

原始TRAIN的432个关系与当前402个关系之间的30个差异，全部来自原Fold A/B隔离来源：29条语句中14条属于Fold A、15条属于Fold B。原合法构建池9354条语句、37407段录音恰好覆盖402个关系。复现原selector得到完全相同的803条slot支持和474条intent支持，再执行原知识编写函数得到完全相同的54张卡；采样丢关系0、编写丢关系0。

因此此前“全域覆盖缺口”的说法只描述原始TRAIN与构建池的差异，不能解释成工程漏建。保持当前402个关系，不从隔离来源补知识。真正待修的是按意图投影后仍残留的跨意图通用正文，以及卡—示例一致性。详细原来源、隔离命中和复现证据见s-slot-scope-source-review-v1交付。

## 下一步

保留402个合法构建关系与当前OFF基线；优先按intent×slot修正文语义，核卡—示例冲突。原545样本不变，69隔离不进入调优。

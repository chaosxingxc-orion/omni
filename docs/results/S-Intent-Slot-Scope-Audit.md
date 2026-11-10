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

## TRAIN全域覆盖

按原数据scenario + action合成意图，审查本地锁定TRAIN元数据11514条、60类。原intent字段有162条仅含action，不能直接用于统计。

当前54类槽位卡有402个意图—槽位关系；TRAIN实际观测432个。当前关系全部有TRAIN支持，但19个意图共缺30个关系，涉及29条语句。比如cooking_query缺meal_type，原句“what to cook for lunch”标meal_type:lunch；cooking_recipe缺meal_type/event_name。

这些缺失不是当前545错例的解释。也不能自动补齐：其中chili被标为event_name等明显可疑记录，必须核实原标注、已有质量排除和构建来源后确定是合法漏建还是应排除的数据问题。未使用测试数据、未修改数据集或知识卡。

## 下一步修复顺序

1. 预先锁定每个意图的合法槽位集合，并明确关系来源；核查上述30个差异，保留有理由的排除记录。
2. 将语义角色、缺省、排斥与实体组合规则真正按intent×slot构建，移除当前意图不适用的通用线索；不要求所有合法槽位都出现。
3. 同一意图的卡与few-shot执行一致性审查。对于标注本身不一致的模式记录争议，不把少数观察写成强规则。
4. 在锁定OFF与原476学习集上先验证知识修复；69隔离只汇报，不参与策略选择。实体值覆盖哪些词的问题与合法槽位类型集合分别评价。

详细审计与固定基线配置在仓库docs/results中保存；原始样本与请求保留在私有证据目录。

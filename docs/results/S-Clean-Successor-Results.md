S clean后继完整结果与示例消费对照

固定545样本／1635候选条件分支，476学习、69仅报告。只在原gold intent分支评价类型；其他分支用于协议及成本。P、allowed、原音频保持一致。参数冻结，Instruct TP4／c64／out512；S只输出candidate_id、slot_types、unresolved_types。

| 实验 | 学习exact/476 | TP/FP/FN | 学习F1 | 学习非空完整覆盖 | 修复/回退 | 全量exact/545 | held exact/69（仅报告） |
|---|---:|---|---:|---:|---|---:|---:|
| clean基线 | 269 | 459/171/93 | 0.776650 | 253/341 | — | 304 | 35 |
| examples-only | 265 | 455/200/97 | 0.753935 | 252/341 | 13/17 | 301 | 36 |
| organization-only | 271 | 463/210/89 | 0.755918 | 257/341 | 23/21 | 304 | 33 |
| type-projection | 276 | 460/182/92 | 0.770519 | 258/341 | 29/22 | 312 | 36 |

这里B投影276是新clean条件下的实测结果，基线仍为269；历史父276不作为本轮clean基线。B集合净+7、FP净+11，A集合净+2、FP净+39；两者F1下降，不自动晋级。具体版本回退不否定类型/值职责分离或共同区分知识结构。

B保留原示例，仅投影槽值格式；A保留原示例，补共同边界与calendar一致性。局部修复包含减少冗余类型及恢复角色类型，同时泛称、媒体指代、食物/原料、person/relation/business及事件/列表边界仍双向变化。类别只是原注释、供给和行为证据的定位，不是已确认声学或因果结论。

| 学习错误族 | B FP/FN | A FP/FN |
|---|---:|---:|
| temporal_type_or_recurrence_boundary | 17/20 | 25/18 |
| entity_role_person_relation_business_boundary | 34/18 | 38/21 |
| media_reference_or_descriptor_boundary | 19/7 | 21/7 |
| food_recipe_ingredient_or_modifier_boundary | 15/11 | 18/9 |
| place_transport_channel_or_device_boundary | 42/14 | 49/14 |
| topic_name_or_generic_action_boundary | 17/9 | 24/8 |
| other_legal_type_presence | 38/13 | 35/12 |

每个终态实验完整545条case、全部FP/FN类型迁移、所有修复回退、1635原响应SHA与schema复核、逐intent/type及未决补集迁移均已归档。B有8个无效分支、gold545全有效；A有5个无效分支，其中学习gold1条confirmed/unresolved重叠，按失败计，不能当有效空类型。

examples-only：545/545、1635/1635，unknown0/absent0；874.419秒，1.604439秒/样本，4卡/1实例，0.971577卡时，7213465 tokens，剩余0，金额未知。学习gold全剩余补集420/473→427/470，U类型3015→3072；补集变化不当作类型正确奖励。

organization-only：545/545、1635/1635，unknown0/absent0；866.144秒，1.589256秒/样本，4卡/1实例，0.962383卡时，7197306 tokens，剩余0，金额未知。学习gold全剩余补集420/473→423/471，U类型3015→2967；补集变化不当作类型正确奖励。

type-projection：545/545、1635/1635，unknown0/absent0；818.595秒，1.502009秒/样本，4卡/1实例，0.909550卡时，6737134 tokens，剩余0，金额未知。学习gold全剩余补集420/473→429/472，U类型3015→3089；补集变化不当作类型正确奖励。

examples-only已独立实跑：96分支换例、1539原packet复用、原clean snapshot；静态全局TRAIN角色覆盖替换query-nearest，不宣称保留query相关性。多类型示例误诱导已预先登记。不叠organization或semantic候选，不做组合网格。前驱COMPLETE、unknown0、容器退出及guard/服务身份/日志闭环后自动接续。

CPU准备、推理、评分与归档分离；固定镜像预算及1635原生入口验证均通过。后台600秒或终态更新，未知请求不重发，所有历史成本保留。未来候选须具体内容主审及显式ready manifest，不从目录存在推断准入。学习反馈由476独立构建，不向教师提供held聚合；无新增听音结论，不改标注。

examples-only学习gold分支中实际换例50条：exact 20→17，修2退5，TP74→67，FP25→28，FN16→23；其余426条packet完全相同，exact249→248，修11退12。相同packet也有变化，不能把全部差值当换例因果。完整545结果301 exact，学习265低于clean269，不晋级此次具体selector实现。

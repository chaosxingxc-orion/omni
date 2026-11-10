# S 类型存在性：业务角色、词义与标注投影

## 全学习集事实

对 clean-parent 与 clean-strategy 的476学习样本重新按正确意图分支统计，未读69报告样本用于诊断。主可复算入口 code/analyze_role_errors.py，完整聚合 LEARNING-ROLE-ERRORS.json，418条错误观测保存在私有明细；同一样本两臂的错误观测不是418个唯一样本。

| 结果类别 | clean-parent | clean-strategy |
|---|---:|---:|
| 类型集合正确 |269|265|
| 只有混淆提取 FP |119|125|
| 只有漏提 FN |50|50|
| 同时 FP/FN |37|34|
| 协议失败 |1|2|

父FP主要为place_name21、person20、event_name11、change_amount9、device_type8、artist_name8。FN主要为date12、place_name9、business_type7、relation7、media_type6。FP与FN不能仅按标签名映射成同一处提及替换，需要原句、标注和实际卡片共同分析。

## 第一性原理：S 的目标不是一般实体识别

S 的目标为 T_i = {t ∈ L_i : query 在意图 i 的业务定义下表达了角色 t}，L_i 为该意图合法类型集合。这个集合既不是语音里出现的所有实体类别，也不是意图通常需要的字段，更不是D最后能填出具体值的字段。

因此要同时避免两个方向的错误：

- **过度泛化**：某个词是人名/地点/时间条件，就自动增加person/place_name/time，忽略其在当前意图中是完整主题、名称的一部分，或不属于业务定义的类型。
- **过度约束**：没有具体姓名、名称或可格式化数值，就否认已经表达的业务角色；或因为示例中没有共存就强迫两个类型互斥。

多标签可以共存，但共存应有独立角色证据。角色证据不能退化为“所有词义相关的类型都选”；也不能据一条金标的缺失就写成排他规则。S不输出槽值，并不意味着它不需要理解句内语义关系。

## 已逐条查看的机制证据（学习反馈，不作运行规则）

1. news_query 的“give the latest updates on narendra modi”：金标仅news_topic，父输出news_topic+person。实际news_topic卡允许主题中人物有独立槽位，person卡缺该意图的已审角色定义，只剩通用姓名线索。说明泛实体定义与意图角色供给可能不一致；未听音，不能据此证明声音无误或标注绝对正确。
2. calendar_set 的“lily's wedding … with my boyfriend”：金标event_name/date/relation，父输出event_name/date/person。可能同时存在嵌入事件名的人名激活和relation漏识别，但输出没有跨度，无法知道person来自Lily还是boyfriend。不能把FP+FN直接叫同一词的标签替换。
3. audio_volume_mute 的“until i tell you not to be”“until my alarm goes off”：金标无类型，父输出time。实际time意图绑定“how long to stay muted”，可能将任意终止条件扩成time；需要从合法TRAIN和业务定义区分时间表达与事件条件，不能硬禁until/when。
4. takeaway_query 中明说restaurant仍漏business_type；其中一例business_name/order_type已识别。证明存在性问题不全是“隐含表达”，需要检查同意图角色、示例覆盖和模型声学处理；不能以一条文字记录断言工程bug。
5. music_query 的“name of this song … currently playing”金标为空，但play_music的“play that last song”有song_name。不能将两种意图的标注约定一概统一成“this/last song都应提”或“都不提”。用户要求无具体值仍可有类型，应保持该能力，同时明确意图内角色；无法协调的标签争议单独标记，不能悄悄按gold训练反例。

## 可执行的后继方向

类型知识视图和当前共同区分候选继续运行，不把新分析追加入已经冻结的实验。下一轮优先补齐意图角色绑定、共同边界中的独立角色条件及来源；避免通用类型卡越过已给定的业务意图。

教师应输出三类可复用策略：来源支持的角色定义组织；同意图角色差异/共存示例选择；将未裁决标注争议移出运行卡的审计策略。不能输出针对这些样本ID的规则，也不能凭语义常识自行补业务知识。当前教师允许编译的三个意图组以外，只输出待审来源需求，不强行套用别组断言。

运行候选的评价至少同时保留Exact、TP/FP/FN、非空完整覆盖、空集合FP、无具体值类型和未决负担。非空覆盖改善但FP大量增长，应报告真实取舍，不能单靠覆盖或一项加权总分宣布收益。

## TFRL 反馈如何与业务目标对齐

每个候选在固定学习样本上形成逐样本向量 r_j = (Exact_j, type-F1_j, CompleteRecall_j, -FP_j, -FN_j, protocol_valid_j, -unresolved_count_j)。空金标的CompleteRecall不用于增加非空覆盖，协议失败不可折算成合法空。

先按完整性和来源合法性过滤程序错误，再报告候选相对固定父的配对变化与类别保护项。候选比较用向量与Pareto关系保留互补策略，不预先把任意权重称为业务最优。若使用group-relative策略优势，必须等待同批候选的真实rollout，再逐指标在候选组内计算标准化优势；当前只有教师提出策略，不能声称已经完成GRPO更新。样本和候选数不足或同wire波动明显时，保留不确定性，不用更多教师长推理代替真实验证。

teacher的固定模板应讲清操作空间、根因反馈和受保护能力；动态部分只含学习分组、代表轨迹、原卡与来源反证。现有例子仅定位机制，不凭这几条决定整个结构的价值。

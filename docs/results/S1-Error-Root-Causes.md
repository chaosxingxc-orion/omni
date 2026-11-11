# S1错例根因与差异化优化策略

本页是已完成五臂的结果与根因分析，按原位更新维护。私有逐案证据、query和轨迹不公开。报告中的运行状态为生成时快照；最新队列状态见[S1子进程教师与错因驱动TFRL](S1-Local-Teacher-Learning.md)。已测结果与未测策略严格区分。

已交付五个完成臂的全量错误账本：每臂同一6681条TRAIN、66810个Fold、55业务类型，合计128964个错误事件，缺失/多余事件ID均为0。四臂旧索引直接投影，Others由现成配对结果和Fold结果补入，未重建全量请求。历史报告“四臂93049”有加总错误，正确是99518；原证据保持原样，本报告明确更正。

全量记账不等于全量语义定因。继承362条文本triage，本职责明确审读17条、核7个实际供给packet和4个错误Fold响应hash；引用Others专责144个类型—案例对/139条样本的最终审读。按sample_id去重，联合文本覆盖475/6681，另6206条没有上述文本审读；声学听审0，已单因果审定事件0。全部ledger的semantic_cause保留UNRESOLVED，程序性现象标签和文本假说分别记录。Others目的性抽样的类别比例不能外推到全量。

## What：完成实验的结果与覆盖

FP是相对冻结参考的未匹配类型，不自动等同已证语义错误。失败保留分母：baseline有1个length协议事件，不能把失败Fold当正确空；其余四臂完整。每类型计数、gold空/非空、配对修退及来源见METRICS.json、COVERAGE.json、PAIRED-TRANSITIONS.private.jsonl。

| 臂 | TP / FP / FN | C% | R% | P% | B | E0% | exact% | 错误事件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 6065 / 36695 / 379 | 91.8907 | 94.1186 | 14.1838 | 5.4924 | 4.3119 | 3.0834 | 37075 |
| grounding | 5765 / 18851 / 679 | 85.6254 | 89.4631 | 23.4197 | 2.8216 | 12.4771 | 10.4625 | 19530 |
| none | 5472 / 14518 / 972 | 79.7378 | 84.9162 | 27.3737 | 2.1730 | 18.2110 | 13.9949 | 15490 |
| supply | 6283 / 27262 / 161 | 96.5119 | 97.5016 | 18.7301 | 4.0805 | 4.8624 | 4.2359 | 27423 |
| others | 5874 / 28876 / 570 | 87.8916 | 91.1546 | 16.9036 | 4.3221 | 3.2110 | 3.6671 | 29446 |

分母固定：正样本4501、空参考2180、gold类型单元6444。C为正样本所有gold被覆盖且协议完整；R/P为类型单元召回/精度；B=FP/6681；E0为空参考完整正确空；exact为完整集合精确一致。baseline的PR使用可观察集合，C/E0/exact不让失败获正确分，不能把其PR当全请求有效。

| 配对 | rT / lT | aF / dF | 覆盖修复 / 回退 | exact修复 / 回退 |
| --- | --- | --- | --- | --- |
| baseline→grounding | 32 / 332 | 1577 / 19421 | 29 / 311 | 506 / 13 |
| grounding→None | 70 / 363 | 1885 / 6218 | 61 / 326 | 400 / 164 |
| grounding→supply | 547 / 29 | 12301 / 3890 | 516 / 26 | 67 / 483 |
| None→Others | 456 / 54 | 15214 / 856 | 415 / 48 | 21 / 711 |

这些是配对组合干预的观测效应。supply对grounding提高C10.8865pp，原配对95%区间[9.9866,11.8510]pp，同时增加FP8411、exact净减416；Others对None提高C8.1537pp，95%区间[7.2482,9.0971]pp，同时净FP+14358、exact净减690。None相对grounding降低C5.8876pp，95%区间[−6.7484,−5.0616]pp。均不能由单一指标锁winner，也不能把组合结果归到一个组件。

## Why：六类可操作观察，保留根因未决

1. **音频内容与实际语义角色理解。** 音频感知A、角色理解B、映射C是潜过程，尚无原声证据区分。文本见证000740的时间角色漏失后恢复，却同时新增地点；003697在设备语句输出drink_type，错误Fold动态例为0；004437的隐含天气参考仍漏。受影响标签至少date、place_name、device_type、drink_type、weather_descriptor、event_name、person、artist_name、song_name。优先U固定知识/示例，只改变角色判断提示；不要求ASR、长思考、意图路由或值填充。反例是001146在None恢复device_type：不能把门槛/控制变化说成对所有类型单调抑制。
2. **源隔离后的定义供给缺口。** 历史三臂559/6444个gold单元仅类型名，漏86/119/154；其余5885有定义仍漏293/560/818。supply仅3个name-only且全命中，仍有161个完整定义FN。010574核到name-only卡；000740又核到完整date卡和同词示例仍漏，说明缺口真实但不充分致错。标签包括date、list_name、person、place_name、time、media_type、device_type、artist_name、event_name等；完整类型清单附后。保持来源排除，不能恢复被屏蔽query或抄逐例gold。当前不把再次补已有定义算新轴。
3. **示例消费/话题角色投射。** supply高召回伴随多标；000096 date完整卡、4动态例可见，而新增date不在参考。E只把动态max_examples4→1，检验多例负担/关联投射。003697、012282错误Fold动态0反证动态示例是全部原因；仍有静态示例、规则、声音和选择行为等解释。标签包括date、person、place_name、artist_name、audiobook_author、email_address、transport_name。先比较实际changed/unchanged Fold，不能按gold挑例，也不能从输入缩短直接称语义提升。
4. **已来源绑定的边界知识在相关Fold不可见。** 字段与地址值、演唱者与书作者、交通方式与服务名/供应商是不同关系。015754/003953/012282有实际卡、输出与规则定位；部分卡本身已有边界，不是只有类型名。K把现有明确label的来源规则投影给相关Fold，辅助U而不替代语音理解。涉及personal_info、email_address、person、artist_name、audiobook_author、transport_type、transport_name、transport_agency、transport_descriptor。反例003697不属该家族，K不能承诺修；作者/人名及多种交通角色可共存，不硬互斥。
5. **控制呈现/候选判决变化。** None新增丢失363个gold单元，其中319转显式None、44仍保其他业务；标签包含date、person、place_name、time、event_name、device_type等。Others组合又恢复456、丢54，不能简单称阈值统一下降。其实际提示同时换判断措辞、one-or-more框架、候选标签再次呈现和后备符解释；两臂schema均无minItems，语法未强制业务非空。布尔B独立测判决/协议轴，科学parent仍None。Others名称/释义/重复呈现的单因素建议归独立专责，本目录只引用，不重复造方案。
6. **参考粒度、任务契约和协议工程。** 000037/000062/000118及002910/011485等存在文本与参考张力，涉及device_type、alarm_type、artist_name、song_name、movie_name、podcast_descriptor、weather_descriptor。不能未经规范和原声审定就删样本、改gold、以词面黑名单“修复”。泛称、隐含指称、未知值保护仍有效。唯一协议length涉及baseline009803 fold-07（audiobook_author、cooking_type、game_name、music_descriptor、timeofday的候选Fold），当前不能据一个事件解释大规模语义误差。

每类的证据、反例、操作与否证条件详见WHY-WHAT-HOW.md；六类有重叠，不是全体错例的互斥因果分类。未审事件没有被程序规则强行填因。

## Others分层的有限吸收与主审闭环

已读本目录MAIN-REVIEW.md并吸收两段意见。引用独立专责最终SEMANTIC-LEDGER/REPORT/VALIDATION，不重做其139条审读。其统计总量与本账本逐项一致。新增FP15214、移除856、净增14358；全部业务标签另有恢复456、丢54，所以业务净增14760，不能混用。

| 类型 | 冻结cohort gold | TP前→后 | FP净增 | 含义 |
| --- | --- | --- | --- | --- |
| personal_info | 45 | 44→44 | 1180 | 恢复1同时丢1，不能被全局净TP掩盖 |
| game_type | 0 | 0→0 | 1071 | 正类召回不可估计，不能永久压掉此类 |
| app_name | 37 | 29→35 | 892 | 存在真实恢复，也有大量扩张 |
| player_setting | 118 | 107→104 | 590 | 精度与召回双风险需单列 |

负Fold新增FP14877/60429=0.24619事件/Fold，正Fold337/6381=0.05281事件/Fold；这比只报97.7853%在负Fold更能体现暴露基数，但事件率不是出错概率，一Fold可多事件，也不识别因果。样本层新增FP分别为正样本10506、空样本4708。Mixed只有38 Fold，其最多现存124个参考外业务类型也不足解释净FP14358，清掉mixed还会伤6个gold，不能冒作主修复。

元数据支持核验：全TRAIN来源11514行中game_type有1行，原upstream DEVEL2033行中也有1行；均保持原标注，仅汇总支持数，未使用DEV模型表现。此检查未建立冻结DEV1850成员关系，因此“冻结DEV1850中game_type是否有正支持”仍未知；LABEL-SUPPORT.json明确此限制。cohort gold0不等于来源无支持，也不授权删类型/补gold/重抽6681。

## How：四个优化轴与下一轮学习

| 轴 | 改变ω项 | 冻结科学parent | 已核状态 | 新增具体审查/改进 |
| --- | --- | --- | --- | --- |
| U语音角色 | p | supply | 三路已排队等B，结果未知 | 未知值/隐含角色正守卫；标题词不自动生角色；不统一提高阈值 |
| K边界组织 | K | supply | 当前982f3128候选已编译，未全量prepare/FIT/派发 | 明确label、有限源支持、最终预算可见性；旧396不代替391依赖 |
| E示例消费 | R | supply | 全6681 CPU ready，快照未排队 | changed/unchanged分层及静态fallback核算；不把4→1说成全部示例减75% |
| B布尔判决 | q及必要格式p | None | 三路实际运行，快照2006/6681 | required bool完整、严格失败；查输出长度/成本，不把缺key当false |

U真实输入影响66810/66810，ρ=1；E20991/66810，ρ=0.31418949。E动态81327→29178，静态650663→657315，总731990→686493，例证字节70129131→60589693；它不是全样本全例等比例删除。K仅一条音频10Fold核到6个变更，局部ρ=.6不能外推；B没有在本职责核到全量wire影响汇总，ρ保null。字节不是token，L_in/L_out和最终卡时未知项留空。

C全部去示范、R全部去规则仅条件消融；L/I/D/S/T是Others专责的潜在单因素诊断，不计本四轴、不预设组合。新改进项全部NEW_PENDING_REVIEW，不热改已运行/排队的B/U/E/K。

本轮阶段是**已有策略的证据完善**，没有新增实测优化收益：U补角色家族与隐含/未知守卫，K更新当前编译与有限可见性范围，E从14条局部审计更新到全量影响面和fallback核算，B补真实运行及空Fold→单业务的多因素证据，正式候选均未修改。STRATEGY-PORTFOLIO逐轴记录相较上一版的修改与未修改原因。009803只支持B的长度/协议风险，不直接支持认知假说；B的动机另参考14083个None-only→业务、14200个空→单业务Fold及Others审读的恢复/扩张反例，仍不构成单因素证明。

参数θ冻结，搜索ω=(K,R,p,q)。LEARNING-NEXT.json同时给每策略changed/frozen、预期rT/lT/aF/dF及C/R/P/B/E0/exact/成本、实际ρ；所有尚未测得收益均null。ACE本轮在本目录更新教师记忆：修正总数、保动态0反例、更新版本/影响面，未下发学生。组相对搜索单列None/Others观测向量与逐维优势及原始差值，supply下U/E/K候选优势仍null；双臂A约±1丢失幅度，不能跨轮比较，也不能抵消召回损失，不称参数GRPO训练。

U/E/K严格遵循已有SELECTION-PROTOCOL：完整/unknown0；C/R差值配对CI下界≥−0.005；P不降/B不升；再满足B相对降≥5%且ΔB CI上≤0，或exact增≥1pp且CI下≥0；全部类型退化和lost-gold案例审查；TRAIN探索后同parent在DEV1850确认；TEST不调参，最终完整13078。None/Other/B和新诊断不追溯套该scope；若要使用相似门槛，须另作前瞻注册。

## 验证、成本与交付边界

完成6681/6681样本、五臂334050个Fold的现成结果核账，错误事件128964/128964。离线核验包括独立集合差、事件ID去重/预期集、四个原hash、Others十Fold并集、五臂TP/FP/FN与覆盖/exact、55类各项加总、四组配对恒等式及文件pins。BUILD-RECEIPT给CPU用时，约13秒/0.002秒每样本，不包括教师思考时间，不是模型秒/条。本轮无正式模型测试；无API/新CLI/agent/NPU请求，0卡/实例、0新增模型卡时，教师token和金额未知。

| 已完成臂 | 样本完成 | 秒/样本/实例范围 | 实例×卡 | 当前卡时 | tokens | 剩余 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 6681/6681 | 0.653–0.683 | 3×4 | 4.960571 | 34696975 | 0 |
| grounding | 6681/6681 | 0.612–0.637 | 3×4 | 4.664346 | 41268384 | 0 |
| none | 6681/6681 | 0.764–0.805 | 3×4 | 5.861634 | 51534400 | 0 |
| supply | 6681/6681 | 0.630–0.652 | 3×4 | 4.784621 | 110654371 | 0 |
| others | 6681/6681 | 0.718–0.748 | 3×4 | 5.478307 | 65579158 | 0 |

历史baseline另含30失败请求成本，all-attempt5.066417卡时高于当前有效轨迹4.960571；其他祖先/失败成本保持原收据，不因本目录统计重复记账。B运行快照约2006/6681、2.086/2.093/1.960秒每样本每实例、三路各TP4、已观测4.556445卡时、各路剩约2997–3294秒；这是归档时快照，非实时承诺。U快照0/6681、0活跃卡/实例、三路队列等待；E只有CPU准备，K仅编译，推理速度/ETA/收益未知。研究实验当前部署累计可为3×TP4，不能当成单台8卡同时占用12卡。

全部交付仅写此独占目录，未改运行树、gold、队列、guard或其他owner产物，未公开私有query/轨迹。公开式REPORT/WHY只使用简短案例编号和类型统计，完整语句与原输出只在private文件。主线程负责脱敏后整合GitHub main/Wiki，本职责不越目录提交。

私有交付清单锁定文件SHA与有限覆盖；主线程已核验17个交付文件、128964个事件的唯一性及各臂计数。未完成的是全量人工/原声定因、冻结DEV1850稀有类支持核实、候选真实效果、K全量FIT与B全量ρ，不伪装完成。本轮要求的分析文件已交付并通过主审；这不代表系统优化目标已完成。

## 附：程序性现象的具体标签范围

以下标签来自已有索引的非互斥观察，不是新人工根因判断。全55类完整TP/FP/FN和配对修退在METRICS.json；所有未归入更具体现象的事件仍在ledger，semantic_cause不被默认补齐。

**NAME_ONLY_DEFINITION_OBSERVED**（baseline 86, grounding 119, none 154）：app_name、artist_name、audiobook_author、business_name、business_type、change_amount、date、device_type、drink_type、event_name、food_type、general_frequency、house_place、list_name、meal_type、media_type、movie_name、music_genre、news_topic、person、place_name、player_setting、relation、time、timeofday、weather_descriptor。

**FN_WITH_NONTRIVIAL_DEFINITION**（baseline 293, grounding 560, none 818, supply 161）：app_name、artist_name、audiobook_author、audiobook_name、business_name、business_type、change_amount、color_type、date、definition_word、device_type、drink_type、event_name、food_type、game_name、house_place、list_name、meal_type、media_type、music_album、music_descriptor、music_genre、news_topic、order_type、person、personal_info、place_name、player_setting、playlist_name、podcast_descriptor、radio_name、relation、song_name、time、time_zone、timeofday、transport_agency、transport_type、weather_descriptor。

**FN_DESPITE_FILLER_IN_DEMO**（baseline 76, grounding 141, none 203, supply 5）：artist_name、business_type、date、device_type、event_name、food_type、house_place、list_name、meal_type、media_type、order_type、place_name、player_setting、relation、time、timeofday、transport_type。

**LOST_TO_EXPLICIT_NONE**（none 319）：app_name、artist_name、audiobook_author、business_name、business_type、change_amount、date、definition_word、device_type、event_name、food_type、game_name、house_place、list_name、meal_type、media_type、music_descriptor、news_topic、order_type、person、place_name、radio_name、relation、song_name、time、timeofday、transport_agency、transport_type、weather_descriptor。

**LOST_WITH_OTHER_BUSINESS_OUTPUT**（none 44）：artist_name、date、device_type、event_name、food_type、house_place、list_name、media_type、music_descriptor、music_genre、news_topic、person、place_name、playlist_name、time。

**PRIOR_TEXT_REFERENCE_TENSION_CASE_NOT_ALL_TYPES**（baseline 106, grounding 84, none 76, supply 89）：alarm_type、app_name、artist_name、audiobook_name、business_name、business_type、color_type、date、email_address、email_folder、event_name、game_name、game_type、general_frequency、house_place、joke_type、list_name、media_type、movie_name、movie_type、music_album、music_descriptor、music_genre、news_topic、person、personal_info、place_name、player_setting、playlist_name、podcast_descriptor、podcast_name、radio_name、relation、song_name、time、time_zone、timeofday、transport_descriptor、transport_name、weather_descriptor。

## 冻结集合稀有类支持补核

主线程已直接核对固定manifest与gold的sample_id集合及SHA，TRAIN6681和DEV1850成员一致，无重新采样或修改标注。game_type在TRAIN正样本0、DEV正样本1；music_album在TRAIN正样本1、DEV正样本0。这里统计的是类型出现的样本数，未读取DEV模型预测，不是模型性能评估。报告上文“冻结DEV1850支持未知”是早先教师检查范围，现由此补核解决。

稀有类支持不足仍是限制：单个正例只能提供个案守卫，不能证明稳定召回；无正例的召回应为不可估计，不能填写为0或100%。保留类型与原有分母，不用删整类、改gold或重抽开发集制造提升。

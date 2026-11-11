# S1/S2工程与知识基线

## 当前交付与运行状态

S拆成独立于意图的两个阶段：S1分Fold选择槽位类型，S2按选中类型填候选值。P不参与两阶段的候选筛选和知识供给。D保留跨类型消歧、意图与槽位关联和最终组装责任。

已完成可调用的工程模块、配置化知识构建、四个rails、真实openJiuwen适配与离线功能测试。固定镜像29项运行/知识测试通过，另20项评分测试通过，65个实际知识视图（10个S1、55个S2）此前装配通过。测试使用模拟模型响应，不是模型精度实验。用户已授权S1三实例真实验证；数据/评分已交付，三路CPU输入与FIT全部通过并已实际派发。首次30次请求因推理引擎grammar不支持uniqueItems返回HTTP400；已修复、保留after rails去重及原错误账本，新版本三路已完成6681/6681样本、66810/66810请求，终态unknown/absent均为0；6680条样本协议完整，另1条有1个Fold输出不完整。完整原始账本归档与全量语义评分已完成，尚未达到可锁定的模型表现。S2/D/教师和旧实验队列未恢复。

## 两阶段与四个rails

| 环节 | 责任与输出 |
|---|---|
| S1 before_model_call | 使用原始语音，注入当前Fold的类型定义和存在性示例；不注入意图、不输出槽值、不要求固定选中数量。 |
| S1 after_model_call | 解析类型集合、清理重复类型，汇总各Fold并集；保留跨Fold候选，不做互斥选择。单Fold失败、空结果、截断分别记录。 |
| S2 before_model_call | 每个选中类型独立请求，加载该类型定义、内部词表、值规则和完整同类型多提及示例；不装入其他类型的比较信息。 |
| S2 after_model_call | 形成类型与候选值列表，保留重复提及、未知值null、原响应和来源；格式修复留审计，不能凭意图过滤。 |

S1输出slot_types，S2输出values，程序根据请求身份补充type和来源。S2可以返回空列表表示未找到支持；它与有类型但值未知的null不同。失败不伪装成空答案。D收到全局候选集合及未完成分支；跨类型候选是否对应同一语音提及需D判断，本版不编造跨请求时间戳或提及对齐。

## S1并发与Fold划分

当前clean-v7 TRAIN有11460句、55种槽位类型；此前401是历史意图—槽位关系数，不是独立类型数。以训练集相同规范化槽值的跨类型关系，加上配置的语义重叠关系构建冲突图；采用确定性、容量受限的图着色，让易混淆类型分处不同Fold。

第一版10个Fold，每组5–6类；226条配置或观察冲突边，组内冲突边为0。它不是向量空间正交性的证明，也不是模型无混淆的证明。不会强行把跨Fold候选合并成一个标签；D负责最终决策。

| Fold | 类型 |
|---|---|
| fold-00 | business_name, color_type, email_folder, event_name, food_type, game_type |
| fold-01 | alarm_type, app_name, ingredient, list_name, person, transport_type |
| fold-02 | artist_name, currency_name, date, definition_word, player_setting, playlist_name |
| fold-03 | change_amount, coffee_type, music_genre, order_type, place_name, podcast_name |
| fold-04 | device_type, house_place, news_topic, song_name, sport_type, time_zone |
| fold-05 | general_frequency, joke_type, meal_type, radio_name, transport_agency |
| fold-06 | audiobook_name, movie_type, relation, time, transport_descriptor |
| fold-07 | audiobook_author, cooking_type, game_name, music_descriptor, timeofday |
| fold-08 | business_type, movie_name, personal_info, podcast_descriptor, weather_descriptor |
| fold-09 | drink_type, email_address, media_type, music_album, transport_name |

S1请求并行，全部Fold返回后对选中类型进行S2并行；这是每条样本的局部依赖，其他样本可继续推进。所有样本共享同一个SlotStages实例的请求信号量，默认64，不能为每条样本新建64额度。每条样本调用数为10+S1选中类型数，额外音频处理成本需要实际评测；尚不声称优于单次调用。无需修改推理引擎attention mask，独立请求天然隔离Fold上下文。

## 知识来源、分类与清理

构建仅读取clean-v7 TRAIN，不读取TEST。训练全域用于类型、值覆盖及混淆边统计；模型可见知识另排除已有坏源/开发重叠源组件，并在运行时排除当前训练query同源组件。来源排除同时作用于示例、词表和派生语义卡，不只删除来源ID。

30类整理为观察词表、21类为开放实体、4类为模式表达（date、time、change_amount、general_frequency）。这种分类是消费策略，不把TRAIN有限观察当成业务定义的封闭全集。没有任何类型被自动设置为封闭白名单。

观察词表得到1041个规范化条目、1042个表面形式。采用NFKC、大小写和空白规范化作为索引，保留原始表面形式及全部出处；不盲删标点、不模糊合并近似词、不把规范化索引强制作为模型答案。703个单来源条目、114个跨类型同表面条目保留并标注待审，不把稀有或多义等同标错。TRAIN中1条槽值等于类型名称的提及保留。

| 观察词表类型 | 有来源的规范化值数 |
|---|---:|
| alarm_type | 4 |
| business_type | 63 |
| coffee_type | 15 |
| color_type | 25 |
| cooking_type | 10 |
| currency_name | 52 |
| device_type | 75 |
| drink_type | 8 |
| email_folder | 5 |
| food_type | 154 |
| game_type | 1 |
| house_place | 38 |
| ingredient | 16 |
| joke_type | 16 |
| meal_type | 5 |
| media_type | 91 |
| movie_type | 8 |
| music_descriptor | 40 |
| music_genre | 55 |
| order_type | 17 |
| personal_info | 41 |
| player_setting | 47 |
| podcast_descriptor | 18 |
| relation | 81 |
| sport_type | 4 |
| time_zone | 19 |
| timeofday | 11 |
| transport_descriptor | 8 |
| transport_type | 10 |
| weather_descriptor | 104 |

数值/时间模式不建立穷举白名单，默认提供6个不同表达的训练示例及类型内部规则；开放实体默认3例；词表类默认2例加全部该类观察值。S1每类2个存在性示例。这些数量均为配置，尚非经实验优化的结论。few-shot按单条原始问句聚合该类型所有提及，避免只展示一个值而教模型漏掉第二个。

已通过配置补齐game_type和music_album的保守类型定义，保持真实TRAIN示例与来源，不生成额外示例、不使用测试集。game_type沿用既有“游戏种类”定义；music_album按锁定类型名称描述音乐专辑/集合，明确标为工程维护定义而非官方本体。两者各只有一个真实TRAIN来源；music_album孤例语义不典型，不将其泛化成所有主题曲均为专辑。其余53卡、10个Fold和全部来源映射不变。

## 工程入口与配置

源码：rails/slot_stage_knowledge.py（构建、图分组、词表分析），rails/slot_stages.py（四rails与异步调度），rails/slot_stages_native.py（openJiuwen与原journal适配）。配置：configs/slots/s-stages-v1/knowledge-policy.json及runtime.json。仅修改配置即可调整Fold约束、样例数、顺序、预算与请求并发。

准备脚本scripts/analysis/prepare_slot_stages.py只装配请求，不访问模型。NativeStageCaller复用原NativeRoleNode，JournalStageTransport复用现有observed_response、admission、journal及由owner提供的FIT/租约检查；不新增provider、常驻queue或自动重发逻辑。旧S的意图分支适配器不能直接执行这些新请求。新D接口为s1-s2-handoff-v1，旧按意图分支的D桥不应直接混用；本次不改变D决策实现。

模型配置保持Instruct、非thinking；S1输出预算256，S2为512，知识在前语音在后。预算与顺序是可运行配置建议，非测得最优。新S1真实语音FIT三路通过，最大保守上界35458，小于65536；已取得三路完整运行终态；全量语义评分与已测吞吐见下表，尚无持续AI Core满载证据。

## 验证与后续顺序

29项固定镜像运行/知识测试覆盖Fold覆盖/隔离、跨样本统一限流、真实openJiuwen hooks、同源知识排除、重复槽值与null保留、协议错误分类、原journal已知响应复用和未知请求不重发。模型调用0，NPU设备挂载0，网络禁用。

后续先审知识缺口与Fold分组，再在相同冻结数据上分别观测S1类型覆盖/混淆候选负担、S2给定正确类型的填值能力、S1→S2实际级联结果。D自行补充必须在固定P/S输出下做增益与新增错误对照。本轮先运行下述S1基线，S2/D另行安排。

## 复现命令

在studies/speech-aware-evidence-acquisition目录执行，输入文件由部署配置指定，原始训练数据和知识产物不进入Git：

```bash
PYTHONPATH=src python -m speech_aware_evidence_acquisition.rails.slot_stage_knowledge \
  --train "$CLEAN_TRAIN" --bundle "$LOCKED_SOURCE_BUNDLE" \
  --policy configs/slots/s-stages-v1/knowledge-policy.json --output "$KNOWLEDGE_OUTPUT"
PYTHONPATH=src python scripts/analysis/prepare_slot_stages.py \
  --knowledge "$KNOWLEDGE_OUTPUT/knowledge.private.json" \
  --config configs/slots/s-stages-v1/runtime.json \
  --sample-id "$SAMPLE_ID" --audio-part "$AUDIO_PART_JSON" --output "$PREPARED_OUTPUT"
PYTHONPATH=src python tests/unit/test_slot_stages.py -v
```

最后一条在普通环境跳过固定镜像依赖项；固定镜像内设置S_STAGES_NATIVE_TEST=1，并将S_STAGES_RUNNER_PATH指向现有run_s_native.py，可执行对应原生依赖测试；新增test_slot_stage_definitions.py验证配置卡来源。正式接续继续使用原owner租约、FIT与journal，不把离线准备脚本当作模型运行入口。

## S1三实例冻结验证

复用历史P@3命中后冻结的TRAIN cohort：6682条经过clean-v7移除1条变为6681条，不是原P输入池9934条，也未把后到的P成功样本追加进去。S1仅接语音与全局Fold知识，不接P输出或金标意图。当前分母包含4501条非空金标、2180条空金标，54个有支持类型、6444个样本—类型正例。来源组独立分片为2227×3，三个4卡Instruct实例，默认每实例实际请求并发64；每样本10个Fold，共66810请求。音频复用现有packet，不重编码或复制十份。

主要指标：类型集合完全一致率、micro precision/recall/F1、支持类别macro、非空样本全部金标类型覆盖率；同时报告类型数量一致、空样本正确率、平均额外候选和漏类型数、逐类型/Fold结果。空样本不进入非空覆盖率分母；失败/截断/未完成Fold保留全分母，不能伪装有效空答案或完整覆盖。参考未匹配候选不直接等同语义错误，需对原句做根因分析。

训练探索集game_type支持0、music_album支持1：本轮可检查两卡加载及误报，无法建立这两类的稳定召回结论。原独立dev1850条保留，本轮不新跑；TRAIN学习表现不能称为泛化能力。

运行状态：三路容器均正常退出且资源guard已释放，共完成6681条样本、66810请求。A路1个Fold输出不完整，保留原响应和该样本全分母，不以重发或删除掩盖；B/C协议完整。该基线任务已结束；后继提示词对照已启动，最新核验见下节。

| 实例 | 完成样本 | 完成请求 | 同主机墙钟 | 吞吐摊销秒/样本 |
|---|---:|---:|---:|---:|
| A，TP4 | 2227/2227 | 22270/22270 | 1521.28秒 | 0.6831 |
| B，TP4 | 2227/2227 | 22270/22270 | 1488.42秒 | 0.6684 |
| C，TP4 | 2227/2227 | 22270/22270 | 1454.81秒 | 0.6533 |

三个实例共12卡、每实例64请求并发；表中是并发吞吐摊销，非单样本延迟。原输入准备86–88秒，音频传输/重新编码0。终态记录本次成功attempt合计34696975 tokens、4.96057分配卡时；此前30次HTTP400另计约0.10585卡时、usage缺失，不能记token成本0。全量原始账本已复核，成功attempt每条响应均有usage；此前拒绝请求的usage缺失仍保留未知。不存在跨主机epoch相减的效率估计。

## 工程稳定性专项审查与归档修复

独立Codex CLI审查已完整交付，153份封存文件、30条原拒绝请求核验通过；结果见[运行基座稳定性审查](Runtime-Stability-Review.md)。下一release优先补真实服务能力契约、输入与部署READY闭环；计划不等于已实施。

本轮新增发现：归档遍历正在原子写入的临时文件，文件重命名后触发FileNotFoundError；主进程存活不足以证明各lane归档正常。远端推理未因此失败，三路终态均COMPLETE。只读归档链路已修复，固定镜像6项竞争/失败接续测试通过，三lane完整归档成功并由原终态评分进程完成评分；没有重发推理或修改远端执行树。

## 全量S1语义基线（6681条）

全量评分和主线程独立集合复算一致。推理任务已终态；评分器标PARTIAL是因为有1个Fold因length而不完整，不代表还有样本未运行。该条保留全分母，未重发、未删除。

| 指标 | 结果 |
|---|---:|
| 非空样本全部类型完整覆盖 | 4136/4501，91.89% |
| 类型micro recall | 6065/6444，94.12% |
| 类型micro precision | 6065/42760，14.18% |
| 类型micro F1 | 24.65% |
| 类型集合完全一致 | 206/6681，3.08% |
| 类型数量一致 | 232/6681，3.47% |
| 空标注样本正确返回空 | 94/2180，4.31% |
| 平均输出类型数 | 6.40 |
| 平均未匹配标注候选数 | 5.49 |
| 漏标注类型总数 | 379 |

可运行与较高类型覆盖尚未转化为有效候选控制。未匹配标注候选不直接等同语义错误，需结合知识定义、示例、原句及语音分别审查；不能凭模型分歧删除数据。6681仍是历史P命中后的TRAIN cohort，不能称全TRAIN泛化结果；独立dev1850尚未运行。

全量发现805次整Fold全选，涉及514条样本；其中789次所在Fold无金标，711次按卡片顺序输出。这种输出形状贡献4362/36695（11.89%）未匹配候选，不能解释其余88.11%。因果仍需干预验证，不能直接宣布正例污染或grammar偏置。

实际请求视图还发现49种类型共559次定义因同源排除降为类型名；分析保留这一真实视图，不拿完整卡片冒充模型实见内容。部分额外候选有目标文本与卡片支持，反映全局类型定义和原任务标注范围的张力；另有卡片已明确覆盖仍漏判的类型，需要继续区分声学与语义能力。没有听审证据的声学根因保留未决，未凭模型分歧删除样本。

后继已启动仅修改instruction的同cohort配对实验，明确最后语音为唯一判别对象，知识、示例、Fold、模型及预算保持不变；另一去示例配置只作后备机制诊断，未派发。候选不能仅凭召回提高晋级，需同时比较覆盖、precision、完全一致、空样本和逐源组退化。开发集1850用于后续验证；若据其继续调优，明确记录参与选择，不冒称无偏泛化。具体容差是分析建议，不是用户指定硬门槛。

## 利用率与剩余效率问题

一次现场快照显示项目卡AI Core约1%–17%，尚不满足80%目标。该值是瞬时采样，不能推断稳态均值或唯一瓶颈；较早服务快照running48/58/53，说明请求占用与算力利用率须分别报告。后继结合吞吐、CPU/传输/模型阶段耗时和core观测，再决定是否提高并发或优化供给。当前运行的固定基线不热改配置。


## 当前后继：目标语音证据提示词对照

实验已取得三路真实模型响应。复用相同6681条TRAIN、55类型/10 Fold、知识卡和示例、Instruct OFF、256输出上限，仅改变system instruction：明确参考卡不是目标query证据，允许无证据时返回空，保留有证据但值未知的类型。目标是降低额外候选负担，同时保持正样本类型覆盖。

首批实际核验如下（启动约42秒快照，非最新实时计数）：

| 实例 | 完整样本/本路总量 | 已响应Fold/总量 | 在途请求 | 服务运行请求 |
|---|---:|---:|---:|---:|
| A | 13/2227 | 138/22270 | 64 | 63 |
| B | 14/2227 | 144/22270 | 64 | 58 |
| C | 16/2227 | 171/22270 | 64 | 44 |

合计43/6681条、453/66810次响应；3个TP4实例共12卡，每路64并发。首响应均正常结束且有真实usage，尚无全量精度结论。启动阶段摊销约2.67–3.22秒/条，包含冷启动，不是稳态效率；此前基线稳态全程摊销0.65–0.68秒/条、本路约24–25分钟仅供参考，当前ETA未形成可靠估计。尚未测得本轮AI Core利用率，不能把在途64等同算力满载。

本次首次启动因正式容器缺模型配置挂载失败，已确认三路均0请求。原职责子进程修复后，使用正式driver完成三路CPU绑定验收，移除所需挂载的负例均按预期失败；固定镜像40项相关测试通过。运行复用旧FIT和音频，不重新编码。后台归档和终态评分已启动，正常约600秒或终态事件观察，配对分析另行CPU并行处理。

新增离线发现：559个定义退化为类型名的实际正例判断，命中473、漏86，占全部379漏项的22.69%；其余5885正例命中5592。两组样本/类型组成不同，不能据此声称补定义可修86项。后续优先核查独立知识来源能否补足定义，保留同源排除，不以query自身或测试答案补知识。

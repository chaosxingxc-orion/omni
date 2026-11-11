# S1/S2工程与知识基线

## 当前交付与运行状态

S拆成独立于意图的两个阶段：S1分Fold选择槽位类型，S2按选中类型填候选值。P不参与两阶段的候选筛选和知识供给。D保留跨类型消歧、意图与槽位关联和最终组装责任。

已完成可调用的工程模块、配置化知识构建、四个rails、真实openJiuwen适配与离线功能测试。固定镜像18项测试通过，65个实际知识视图（10个S1、55个S2）装配通过。测试使用模拟模型响应，不是模型精度实验。没有启动模型请求、教师请求、实验队列或后台开发worker。既有实验保持暂停。

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

发现game_type和music_album没有可复用的完整语义卡，当前仅保留原标签名称和训练证据，未用模型虚构定义。两者各只有一个有来源的观察示例；music_album现有来源尤其不足以建立稳定定义，应作为知识缺口审阅，而非据此扩写、删除或宣布类别已完整覆盖。

## 工程入口与配置

源码：rails/slot_stage_knowledge.py（构建、图分组、词表分析），rails/slot_stages.py（四rails与异步调度），rails/slot_stages_native.py（openJiuwen与原journal适配）。配置：configs/slots/s-stages-v1/knowledge-policy.json及runtime.json。仅修改配置即可调整Fold约束、样例数、顺序、预算与请求并发。

准备脚本scripts/analysis/prepare_slot_stages.py只装配请求，不访问模型。NativeStageCaller复用原NativeRoleNode，JournalStageTransport复用现有observed_response、admission、journal及由owner提供的FIT/租约检查；不新增provider、常驻queue或自动重发逻辑。旧S的意图分支适配器不能直接执行这些新请求。新D接口为s1-s2-handoff-v1，旧按意图分支的D桥不应直接混用；本次不改变D决策实现。

模型配置保持Instruct、非thinking；S1输出预算256，S2为512，知识在前语音在后。预算与顺序是可运行配置建议，非测得最优。实际语音长度FIT及真实端到端NPU指标仍待实验恢复时验证。

## 验证与后续顺序

18项固定镜像测试覆盖Fold覆盖/隔离、跨样本统一限流、真实openJiuwen hooks、同源知识排除、重复槽值与null保留、协议错误分类、原journal已知响应复用和未知请求不重发。模型调用0，NPU设备挂载0，网络禁用。

后续先审知识缺口与Fold分组，再在相同冻结数据上分别观测S1类型覆盖/混淆候选负担、S2给定正确类型的填值能力、S1→S2实际级联结果。D自行补充必须在固定P/S输出下做增益与新增错误对照。当前仅交付工程与知识准备，不恢复实验。

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

最后一条在普通环境跳过固定镜像依赖项；固定镜像内设置S_STAGES_NATIVE_TEST=1，并将S_STAGES_RUNNER_PATH指向现有run_s_native.py，可执行全部18项。正式接续继续使用原owner租约、FIT与journal，不把离线准备脚本当作模型运行入口。

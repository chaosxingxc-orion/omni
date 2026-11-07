# 原Fold A1080历史P顺序与供给复核

已完成原始1080/1080，不更换gold、不新增样本。top1/recall@1 **867/1080=80.2778%**；recall@2 **951/1080=88.0556%**；recall@3/候选集合召回 **977/1080=90.4630%**；MRR **0.849691358（2753/3240）**。最终1079有效、102有效miss、1 invalid；invalid以0计入全部主指标。当前只P，P多轮获益并冻结真实输出后才进入S、D。此交付不实施生产评分器或派发实验。

## 排序成立的证据与边界

历史source `2586419b119d7077a405a4fb951e00ee59c231eb` 的 `intent_slot_pipeline.py:141–171` 中，`candidates()`逐项读取原始JSON数组；首次intent追加到result，重复intent只追加sources。因此group_v2保留首次出现的相对顺序。没有按intent字典排序，也没有按D选择重新排序。实际归档的1104个P请求都绑定到包含“1 to 3 distinct candidates ordered by support”的actor提示；恢复请求也保留该原提示。最终1079有效响应均与stable first-occurrence顺序一致，1077个native P receipt的candidate_ids/grouping也全部一致；708个有效输出的实际顺序与字典排序不同。

本报告的rank是原模型被要求输出的support数组顺序，经稳定group_v2归并后的1-based位置；重复项不是新的rank。它支持“观测到的P support-order top1/MRR”，不证明模型每条都按真实置信度排序，也没有logprob、置信度校准或因果校准证据。字典顺序只用于聚合表展示，从未用于候选评分。`rank-proof.json`保存原函数和检查计数；私有order-witnesses保存可回溯反例。

1077条依native perception_source准确定位原contact；另外3条无最终P块，按实际compact reask或唯一原P请求恢复，其中2条P正确而下游终态失败、1条P自身invalid。逐条与既有pin的effective_request_id一致，不从D结果猜P top1。

## 主指标、候选数与分层

|指标|全1080分母|仅1079有效条件分数|
|---|---:|---:|
|top1 = recall@1|80.2778%|80.3522%|
|recall@2|88.0556%|88.1372%|
|recall@3|90.4630%|90.5468%|
|MRR|0.849691358|0.850478838|
|合法覆盖|1079/1080 = 99.9074%|条件分母不用于遮掉invalid|

候选数按合法归并后计：0候选（invalid）1条、1候选174条、2候选124条、3候选781条；合计2765，均值2.560185/全行、2.562558/有效行。gold rank1=867、rank2=84、rank3=26、未召回=103（102 miss+1 invalid）。因此已有110条候选集合正确但top1错误，是独立于扩K的可测排序改进空间。

按原始scenario_action定义，共60个实际意图；无笛卡尔积。每意图等权macro：top1/recall@1=80.7610%，recall@2=88.8154%，recall@3=90.3574%，MRR=0.853021978。每意图完整分母、有效条件分数、候选数与供给rank分布都在result.json。18个原始stratum各60条，故stratum等权macro恰等于整体micro，不等于意图macro。

|原stratum|n|top1|recall@2|recall@3|MRR|
|---|---:|---:|---:|---:|---:|
|alarm|60|56/60|58/60|58/60|0.950000|
|audio|60|48/60|53/60|55/60|0.852778|
|calendar|60|46/60|53/60|54/60|0.830556|
|cooking|60|50/60|55/60|57/60|0.886111|
|datetime|60|50/60|56/60|57/60|0.888889|
|email|60|51/60|54/60|54/60|0.875000|
|general|60|21/60|32/60|43/60|0.502778|
|iot|60|46/60|50/60|52/60|0.811111|
|lists|60|52/60|54/60|54/60|0.883333|
|music|60|43/60|52/60|53/60|0.797222|
|news|60|48/60|53/60|55/60|0.852778|
|play|60|51/60|53/60|53/60|0.866667|
|qa|60|53/60|54/60|56/60|0.902778|
|recommendation|60|58/60|60/60|60/60|0.983333|
|social|60|42/60|47/60|48/60|0.747222|
|takeaway|60|48/60|55/60|55/60|0.858333|
|transport|60|48/60|55/60|55/60|0.858333|
|weather|60|56/60|57/60|58/60|0.947222|

general的top1仅21/60、集合43/60；general_quirky按原gold为45条、top1为7/45、集合29/45，22条召回但排序靠后。该观察支持保留原定义并检查区分性与排序；不能据此删除quirky或修正标签。低样本意图如iot_wemo_on为2/4须保留分母，不能直接当稳定效应。

## 缺失、非法、截断与实际回退

缺失P contact为0；缺失最终native P块3条已按原contact恢复。有效原始JSON/schema之外还有引用合法性：最终raw-valid为1072，raw-invalid为8，其中7条因未供给的evidence引用被原生quarantine-unsupplied-v1隔离后有效；剩1条length截断且非完整JSON，仍invalid。没有因重复intent将合法group_v2输出判错，最终raw duplicates为221条（201条召回、20条有效miss）。

共1104个真实历史P contacts=1080初次+24实际compact reasks。初次1056有效、24 invalid，top1 851/1080、recall@2 932/1080、集合957/1080、MRR 0.833179012；恢复后23/24有效、20/24召回，最终top1增加16、集合增加20，仍有1条invalid。25个contact finish=length（24初次+1恢复），1079个finish=stop；按样本any-length为24。24条恢复供给与其初次相同。恢复是同一条路径的依赖处理，不能视作独立策略draw，也不能丢掉首轮成本。

## 历史供给绑定归因

“正确”在本节仅指P候选集合包含原gold，不使用D或系统正确性。供给rank是选定原P请求supplied_evidence数组中的位置，candidate rank是P响应经group_v2归并后的位置，二者分别记录；未将重算向量rank伪装成历史confidence。

|历史P结果|gold卡未供给|gold卡已供给|总数|
|---|---:|---:|---:|
|有效miss|50|52|102|
|invalid|0|1|1|
|集合召回|6|971|977|
|全部|56|1024|1080|

历史供给gold覆盖1024/1080=94.8148%；gold供给位置1..8分别787/105/57/19/16/16/13/11条，未供给56条。50个未供给miss可用于供给覆盖假设；52个已供给miss加1 invalid仍需区分音频理解、知识边界、提示理解和合法输出处理。6个未供给仍召回（其中4个top1、2个rank3）证明合法inventory仍可支持输出，不能把“未供给”直接等同于不可解或因果根因。

已供给971条召回中gold candidate rank1/2/3=863/84/24；另未供给的6条为4/0/2。完整供给位置×候选rank×结果joint分布及逐行source/hash在机器产物中；52个已供给miss中gold卡原供给位置1/2/3仍有4/14/14条，因此单纯将卡提到前3不等于解决P排序。

本轮逐条将实际1080P请求的卡ID、顺序、正文SHA与既有native供给分析比对：parent全部1080集合相同、顺序相同，60卡正文hash全部锁定；tau0.1也全部相同，属于供给no-op。breadth12/16只改变知识供给卡数，不改变输出候选K_max=3；它们均不匹配历史实际供给，不能把历史P输出挂到这些新供给上冒充效果。

离线反事实覆盖：12卡使原102个有效miss中的16个未供给gold变为供给（52→68）；16卡为29个（52→81）。原977个召回中均仅增加5个gold卡覆盖（971→976），invalid的gold始终已供给。覆盖总数分别1045、1058，不是P recall。只有真实新匹配P控制才能验证收益，不能声称16或29个miss会被修复。额外知识的完整请求token/延迟亦未在本任务测量。

原80历史TRAIN反馈单独保留：实际gold供给73/80，当前parent重算75/80，卡集合仅31/80相同、49/80不同；且原80 P thinking ON，与本1080 P OFF不同。这80条不能与当前重算供给直接配对推断提升，更不能组成当前策略TFGRPO组。没有新增数据名单或label修正。

## 下一轮P奖励与晋升建议

生产评分器未修改；recommendations.json给出实现建议与硬门槛。建议有效输出奖励 H + 0.20×RR − 0.02×(k−1)/2 − 0.02×T − 0.01×F；invalid/缺失为−1。H是原gold的候选集合命中，RR为原support顺序倒数，k为1..3个合法unique候选；T为全部P prompt+completion（含实际恢复）/20000后截到1，F为额外P请求/1后截到1。此处20000和单次恢复是待预注册预算建议，不能据此删结果或免计超额成本。任何有效命中奖励至少1.0167，miss至多0，确保排序或省token不覆盖集合召回。

先守完整合法覆盖及候选集合，再奖励rank和有界回退、约束候选数与成本。固定K_max=3，并同时报告recall@1/2/3、MRR、unique-k分布；只补满3候选且top1/MRR不升，不算晋升。配对比较至少不低于parent 1079有效/977集合召回；若集合相同，需预注册有意义的rank收益且合法率、恢复失败和成本不过门槛；集合提高而rank变差只能列权衡，不能自动晋升。每意图macro/原strata及配对gain/loss和不确定性同步给出，阈值在看后继结果前冻结；使用既有授权TRAIN分区，TEST不选优。权重是可执行提案，未宣称经过收益验证。

供给侧优先验证50个未供给miss对应的全量控制覆盖；知识内容/提示理解侧关注52个已供给miss及原成功反例，分别改变卡边界或P提示；排序侧关注110个rank2/3召回。每类均属可验证干预方向，现证据不能在知识内容、音频理解、提示理解之间作因果拆分。教师建议只是候选干预假设，不能把教师文本或历史相关性当成因果标签。

TFGRPO当前合格同策略多P draw组为0。组键需绑定同sample/audio、相同P policy与模板、相同卡正文及顺序、schema/parser、冻结模型及同一采样分布；独立draw seed/request另记。运行temperature是采样参数，不等于组内奖励标准化或advantage缩放温度。本父temp0/seed20260913/P OFF；确定性重复可能无奖励差异，zero-variance组应零advantage，不能制造反馈。若未来授权探索非零温度，应另注册运行配置，并在同评估解码下比较；不混温度/旧80 PON，不把compact reask当独立draw。参数保持冻结。

## 来源、复现与成本

方法使用原1080 roster和原train metadata逐行hash校验gold；原contacts逐个精确文件定位，核对request canonical hash、intent/response journal hash、原response与wire bytes hash、history hash、实际actor模板及源卡内容hash。未从头扫描仓库或重新采样数据。

既有聚合pin：on-refined-aggregate.json = 6bc5fa09fb90965bd1006c5b2d9c1244fddbd3443e5b3ed4960ca911a3b09195；on-refined-rows.private.json = 3f5d52443ed0da0ff755d8a08158768a8a090bc3a58e102787c7383ef4c5f2ab。原source-manifest、roster、gold、原records与4个供给arm逐一SHA/字节数见私有source-pins；公开result保留无路径的hash索引。

正式native解析使用精确image `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`、`/opt/saea-venv/bin/python -B /script/analyze.py`。network none、root/source/baseline/supply/gold只读、仅任务输出本地持久盘可写并fsync；CPU quota2，单实例，无NPU设备。正式CPU分析57.948191s（约0.053656s/条），独立精确分数复核0.742749s。包含一次验证失败的3个任务容器wall总计60.435999s；非端到端人工分析耗时。验证v1的预期分数字面量手误，修正为2753/3240后v2 PASS；原native结果不变，失败日志/receipt均保留。

历史P成本保留全部1104 contacts：prompt 5,880,781、completion 325,447；其中24恢复额外129,461/8,746 tokens。request latency累计45,940.977179s，均值41.613204s/contact；这不是墙钟吞吐或新P-only秒/条。当前分析新增模型/teacher请求0、tokens0、API费用0、NPU0卡/0秒，剩余0、ETA0；CPU基础设施货币费率未知。

独立固定镜像复核通过：1080唯一分母、原gold一致、真实选中response SHA、raw稳定去重顺序、invalid为0、精确MRR、意图/stratum全覆盖、1104 contacts统一P OFF/temp0/seed20260913/max4096、native reference config唯一。没有调用NPU/模型/teacher，也没有操作活动服务、恢复租约、生产代码、AGENTS、Git/main/Wiki或其他用户资源。

公开目录只含聚合/方法/建议与删去内部路径的来源hash，不含样本ID、原始轨迹、员工标识、内网地址或凭据。根report是本地交付索引；private逐行与源映射仅本地保留。

## 全60个原始意图明细

每行n保留原分母；recall@1等于top1，全部非法仍按0计入。

|原意图|n|有效n|top1/recall@1命中|recall@2命中|recall@3命中|MRR|
|---|---:|---:|---:|---:|---:|---:|
|alarm_query|19|19|16|18|18|0.894737|
|alarm_remove|8|8|8|8|8|1.000000|
|alarm_set|33|33|32|32|32|0.969697|
|audio_volume_down|8|8|8|8|8|1.000000|
|audio_volume_mute|24|24|19|21|21|0.833333|
|audio_volume_other|4|4|3|4|4|0.875000|
|audio_volume_up|24|24|18|20|22|0.819444|
|calendar_query|27|27|19|24|25|0.808642|
|calendar_remove|8|8|7|7|7|0.875000|
|calendar_set|25|25|20|22|22|0.840000|
|cooking_query|1|1|0|1|1|0.500000|
|cooking_recipe|59|59|50|54|56|0.892655|
|datetime_convert|10|10|9|9|9|0.900000|
|datetime_query|50|50|41|47|48|0.886667|
|email_addcontact|4|4|3|3|3|0.750000|
|email_query|31|31|28|30|30|0.935484|
|email_querycontact|6|6|4|5|5|0.750000|
|email_sendemail|19|19|16|16|16|0.842105|
|general_greet|5|5|5|5|5|1.000000|
|general_joke|10|10|9|9|9|0.900000|
|general_quirky|45|44|7|18|29|0.359259|
|iot_cleaning|6|6|4|4|4|0.666667|
|iot_coffee|9|9|8|8|8|0.888889|
|iot_hue_lightchange|8|8|6|6|8|0.833333|
|iot_hue_lightdim|8|8|6|7|7|0.812500|
|iot_hue_lightoff|13|13|10|12|12|0.846154|
|iot_hue_lighton|5|5|4|5|5|0.900000|
|iot_hue_lightup|6|6|5|5|5|0.833333|
|iot_wemo_off|1|1|1|1|1|1.000000|
|iot_wemo_on|4|4|2|2|2|0.500000|
|lists_createoradd|22|22|19|20|20|0.886364|
|lists_query|22|22|19|20|20|0.886364|
|lists_remove|16|16|14|14|14|0.875000|
|music_dislikeness|1|1|1|1|1|1.000000|
|music_likeness|18|18|12|15|15|0.750000|
|music_query|32|32|25|30|30|0.859375|
|music_settings|9|9|5|6|7|0.648148|
|news_query|60|60|48|53|55|0.852778|
|play_audiobook|7|7|5|5|5|0.714286|
|play_game|7|7|6|6|6|0.857143|
|play_music|19|19|16|16|16|0.842105|
|play_podcasts|11|11|10|11|11|0.954545|
|play_radio|16|16|14|15|15|0.906250|
|qa_currency|8|8|8|8|8|1.000000|
|qa_definition|11|11|11|11|11|1.000000|
|qa_factoid|27|27|21|22|24|0.820988|
|qa_maths|6|6|6|6|6|1.000000|
|qa_stock|8|8|7|7|7|0.875000|
|recommendation_events|20|20|19|20|20|0.975000|
|recommendation_locations|30|30|30|30|30|1.000000|
|recommendation_movies|10|10|9|10|10|0.950000|
|social_post|47|47|34|36|37|0.751773|
|social_query|13|13|8|11|11|0.730769|
|takeaway_order|29|29|21|25|25|0.793103|
|takeaway_query|31|31|27|30|30|0.919355|
|transport_query|17|17|12|16|16|0.823529|
|transport_taxi|13|13|12|12|12|0.923077|
|transport_ticket|17|17|12|14|14|0.764706|
|transport_traffic|13|13|12|13|13|0.961538|
|weather_query|60|60|56|57|58|0.947222|

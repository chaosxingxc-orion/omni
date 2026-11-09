# 全集合召回与P节点固定配置结果

Q-multi数据交付状态：COMPLETE_DATA_VERIFIED_LEGACY_FAILURE_UNRESOLVED；旧失败归因状态：PENDING_FAILURE_ATTRIBUTION。TEST数据核验已完成；原closeout/finalize失败保留，补件状态独立。

TEST各配置13078/13078，合39234逻辑/38982新请求/252精确成功复用/unknown0。固定镜像raw+复用核验PASS，本机116946份journal字节比对PASS；本补件消费原收据，没有重跑TEST或验收CPU大过程。

|集合|配置|完成/总量|Top1|Top3|Top3率|macro Top3|业务Top3|常设Top3|业务首选入常设|供给/缺供|供给条件Top3|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|R01C-fit624|Q15|624/624|415|545|87.3397%|0.872450|480/550|65/74|72|613/11|0.889070|
|R01C-fit624|Q10|624/624|425|542|86.8590%|0.873332|478/550|64/74|77|608/16|0.891447|
|R01C-fit624|Q5|624/624|431|544|87.1795%|0.881824|485/550|59/74|60|592/32|0.918919|
|FoldB1078|Q15|1078/1078|818|960|89.0538%|0.883840|888/991|72/87|81|1062/16|0.903955|
|FoldB1078|Q10|1078/1078|834|969|89.8887%|0.886243|894/991|75/87|80|1048/30|0.924618|
|FoldB1078|Q5|1078/1078|844|969|89.8887%|0.878982|895/991|74/87|57|1032/46|0.938953|
|FoldA1075|Q15|1075/1075|833|984|91.5349%|0.919453|921/1003|63/72|76|1075/0|0.915349|
|FoldA1075|Q10|1075/1075|842|986|91.7209%|0.925586|926/1003|60/72|82|1070/5|0.921495|
|FoldA1075|Q5|1075/1075|856|990|92.0930%|0.929028|935/1003|55/72|62|1047/28|0.945559|
|TEST13078|Q15|13078/13078|9992|11672|89.2491%|0.882690|10680/11939|992/1139|858|12781/297|0.913231|
|TEST13078|Q10|13078/13078|10082|11744|89.7997%|0.888785|10738/11939|1006/1139|899|12699/379|0.924797|
|TEST13078|Q5|13078/13078|10273|11840|90.5337%|0.905458|10880/11939|960/1139|631|12540/538|0.944179|

TRAIN只统计既有检索缓存，无P Top1/Top3。原10000/有效9985/既有选择7096；后续质量mask的7094/622/1075属于另一优化分母，本观察不改历史分母，TEST始终13078。

|TRAIN集合|配置|正确供给/总量|缺供|macro召回|业务供给/业务总量|来源阻断业务|平均候选|
|---|---|---:|---:|---:|---:|---:|---:|
|TRAIN7096|Q15|6914/7096|182|0.975319|6271/6453|0|14.619504|
|TRAIN7096|Q10|6862/7096|234|0.969807|6219/6453|0|11.476325|
|TRAIN7096|Q5|6778/7096|318|0.960786|6135/6453|0|8.732666|
|TRAIN9985|Q15|9346/9985|639|0.879634|8516/9155|422|14.647772|
|TRAIN9985|Q10|9279/9985|706|0.874645|8449/9155|422|11.491037|
|TRAIN9985|Q5|9169/9985|816|0.866379|8339/9155|422|8.727692|

以下配对均为同集合固定配置的描述性观察；TEST不用于选K、prompt或知识调优。TEST历史已用于检索研究，不称全新holdout。Q15均common-only，不混s2。

|集合|配对|修复|回退|净变化|
|---|---|---:|---:|---:|
|TRAIN7096|Q15->Q10|0|52|-52|
|TRAIN7096|Q15->Q5|0|136|-136|
|TRAIN7096|Q10->Q5|0|84|-84|
|TRAIN9985|Q15->Q10|0|67|-67|
|TRAIN9985|Q15->Q5|0|177|-177|
|TRAIN9985|Q10->Q5|0|110|-110|
|R01C-fit624|Q15->Q10|12|15|-3|
|R01C-fit624|Q15->Q5|24|25|-1|
|R01C-fit624|Q10->Q5|20|18|+2|
|FoldB1078|Q15->Q10|31|22|+9|
|FoldB1078|Q15->Q5|33|24|+9|
|FoldB1078|Q10->Q5|22|22|+0|
|FoldA1075|Q15->Q10|23|21|+2|
|FoldA1075|Q15->Q5|34|28|+6|
|FoldA1075|Q10->Q5|28|24|+4|
|TEST13078|Q15->Q10|276|204|+72|
|TEST13078|Q15->Q5|419|251|+168|
|TEST13078|Q10->Q5|326|230|+96|

逐类净回退（类名为原始intent定义，不含sample ID/text）；完整逐类、macro、全部修退计数和供给四格见METRICS.aggregate.json：

TRAIN7096 Q15→Q10：alarm_remove 35→34 (n=38)；calendar_query 415→405 (n=423)；calendar_remove 227→226 (n=242)；calendar_set 573→564 (n=595)；datetime_convert 18→17 (n=18)；datetime_query 203→202 (n=203)；email_query 297→296 (n=299)；email_querycontact 84→81 (n=91)；email_sendemail 214→213 (n=222)；iot_hue_lightoff 112→110 (n=119)；lists_createoradd 98→97 (n=101)；lists_query 124→123 (n=127)；lists_remove 100→99 (n=106)；news_query 268→266 (n=277)；play_music 484→476 (n=491)；qa_definition 153→151 (n=169)；qa_stock 96→94 (n=99)；recommendation_events 100→99 (n=103)；recommendation_locations 71→70 (n=73)；social_post 120→119 (n=120)；transport_query 111→109 (n=112)。

TRAIN7096 Q15→Q5：alarm_remove 35→33 (n=38)；alarm_set 109→108 (n=110)；audio_volume_mute 41→40 (n=42)；audio_volume_up 41→40 (n=42)；calendar_query 415→387 (n=423)；calendar_remove 227→224 (n=242)；calendar_set 573→556 (n=595)；datetime_convert 18→17 (n=18)；datetime_query 203→200 (n=203)；email_query 297→295 (n=299)；email_querycontact 84→80 (n=91)；email_sendemail 214→210 (n=222)；iot_hue_lightchange 76→75 (n=79)；iot_hue_lightdim 53→52 (n=54)；iot_hue_lightoff 112→110 (n=119)；lists_createoradd 98→96 (n=101)；lists_query 124→117 (n=127)；lists_remove 100→99 (n=106)；music_query 66→64 (n=68)；news_query 268→259 (n=277)；play_audiobook 108→103 (n=119)；play_game 79→77 (n=84)；play_music 484→469 (n=491)；play_podcasts 160→158 (n=162)；qa_definition 153→149 (n=169)；qa_maths 53→52 (n=55)；qa_stock 96→93 (n=99)；recommendation_events 100→98 (n=103)；recommendation_locations 71→69 (n=73)；social_post 120→118 (n=120)；transport_query 111→109 (n=112)；transport_ticket 80→79 (n=85)；transport_traffic 74→73 (n=75)；weather_query 340→338 (n=343)。

TRAIN9985 Q15→Q10：alarm_query 117→116 (n=128)；alarm_remove 65→64 (n=77)；audio_volume_mute 100→99 (n=110)；calendar_query 482→471 (n=495)；calendar_remove 247→245 (n=270)；calendar_set 622→613 (n=652)；cooking_recipe 161→160 (n=171)；datetime_convert 44→43 (n=52)；datetime_query 335→334 (n=344)；email_query 355→354 (n=366)；email_querycontact 98→94 (n=113)；email_sendemail 256→255 (n=271)；iot_hue_lightdim 66→65 (n=75)；iot_hue_lightoff 138→136 (n=153)；lists_createoradd 148→147 (n=157)；lists_query 176→175 (n=186)；lists_remove 135→133 (n=152)；news_query 390→384 (n=405)；play_music 539→530 (n=553)；qa_definition 171→169 (n=194)；qa_stock 110→108 (n=118)；recommendation_events 160→159 (n=171)；recommendation_locations 119→117 (n=127)；social_post 209→208 (n=217)；takeaway_order 104→103 (n=111)；transport_query 166→164 (n=176)。

TRAIN9985 Q15→Q5：alarm_query 117→116 (n=128)；alarm_remove 65→62 (n=77)；alarm_set 173→172 (n=182)；audio_volume_down 44→43 (n=52)；audio_volume_mute 100→98 (n=110)；audio_volume_up 100→99 (n=110)；calendar_query 482→452 (n=495)；calendar_remove 247→242 (n=270)；calendar_set 622→603 (n=652)；cooking_recipe 161→156 (n=171)；datetime_convert 44→43 (n=52)；datetime_query 335→331 (n=344)；email_query 355→352 (n=366)；email_querycontact 98→93 (n=113)；email_sendemail 256→250 (n=271)；iot_hue_lightchange 103→102 (n=115)；iot_hue_lightdim 66→64 (n=75)；iot_hue_lightoff 138→136 (n=153)；lists_createoradd 148→144 (n=157)；lists_query 176→168 (n=186)；lists_remove 135→133 (n=152)；music_query 133→130 (n=143)；news_query 390→373 (n=405)；play_audiobook 130→125 (n=149)；play_game 93→91 (n=105)；play_music 539→522 (n=553)；play_podcasts 182→180 (n=192)；qa_definition 171→167 (n=194)；qa_maths 63→62 (n=73)；qa_stock 110→107 (n=118)；recommendation_events 160→158 (n=171)；recommendation_locations 119→116 (n=127)；social_post 209→207 (n=217)；social_query 93→92 (n=102)；takeaway_order 104→103 (n=111)；takeaway_query 104→103 (n=115)；transport_query 166→164 (n=176)；transport_ticket 114→113 (n=127)；transport_traffic 95→94 (n=104)；weather_query 476→473 (n=488)。

R01C-fit624 Q15→Q10：audio_volume_up 13→12 (n=15)；calendar_query 17→16 (n=22)；calendar_set 15→14 (n=17)；cooking_recipe 24→22 (n=31)；general_quirky 34→33 (n=41)；music_query 14→13 (n=18)；takeaway_order 23→22 (n=23)；takeaway_query 13→12 (n=14)；transport_query 21→20 (n=21)。

R01C-fit624 Q15→Q5：audio_volume_down 5→4 (n=5)；audio_volume_up 13→12 (n=15)；calendar_set 15→14 (n=17)；cooking_recipe 24→23 (n=31)；email_sendemail 20→18 (n=20)；general_quirky 34→30 (n=41)；music_likeness 6→5 (n=7)；music_query 14→12 (n=18)；news_query 31→30 (n=36)；qa_factoid 31→29 (n=33)；takeaway_order 23→22 (n=23)；transport_query 21→20 (n=21)。

FoldB1078 Q15→Q10：alarm_query 26→25 (n=26)；audio_volume_up 19→18 (n=21)；calendar_query 23→22 (n=28)；email_querycontact 5→4 (n=6)；iot_cleaning 6→5 (n=7)；music_query 24→23 (n=25)；play_podcasts 7→6 (n=7)；qa_definition 9→8 (n=10)；recommendation_events 33→32 (n=36)；weather_query 59→58 (n=60)。

FoldB1078 Q15→Q5：audio_volume_up 19→18 (n=21)；calendar_query 23→22 (n=28)；calendar_remove 8→7 (n=11)；email_querycontact 5→4 (n=6)；iot_hue_lightchange 14→12 (n=15)；iot_hue_lightup 3→2 (n=3)；iot_wemo_on 4→3 (n=5)；music_query 24→23 (n=25)；play_music 27→26 (n=30)；qa_definition 9→8 (n=10)；transport_query 32→30 (n=33)。

FoldA1075 Q15→Q10：audio_volume_up 23→20 (n=24)；calendar_query 26→24 (n=27)；general_quirky 38→36 (n=45)；iot_hue_lightchange 8→7 (n=8)；iot_hue_lightdim 8→7 (n=8)；lists_createoradd 21→20 (n=22)；qa_factoid 25→24 (n=27)；weather_query 56→55 (n=60)。

FoldA1075 Q15→Q5：alarm_remove 8→6 (n=8)；email_query 31→29 (n=31)；general_quirky 38→31 (n=45)；iot_hue_lightchange 8→7 (n=8)；lists_createoradd 21→19 (n=22)；news_query 55→53 (n=58)；qa_factoid 25→24 (n=27)；qa_stock 8→7 (n=8)；recommendation_locations 29→28 (n=30)；transport_taxi 13→12 (n=13)。

TEST13078 Q15→Q10：audio_volume_mute 229→228 (n=266)；calendar_set 706→703 (n=824)；cooking_recipe 253→250 (n=289)；datetime_convert 50→49 (n=55)；email_querycontact 72→71 (n=101)；email_sendemail 367→361 (n=410)；general_joke 120→118 (n=121)；iot_hue_lightup 99→98 (n=108)；iot_wemo_off 51→49 (n=67)；iot_wemo_on 28→26 (n=37)；lists_remove 205→204 (n=230)；music_query 128→127 (n=136)；play_audiobook 261→260 (n=307)；play_game 123→121 (n=138)；qa_definition 175→172 (n=210)；recommendation_locations 98→95 (n=102)；social_post 223→220 (n=260)；transport_query 152→149 (n=167)；transport_taxi 79→78 (n=85)；transport_ticket 154→152 (n=155)。

TEST13078 Q15→Q5：datetime_convert 50→49 (n=55)；email_sendemail 367→363 (n=410)；general_joke 120→118 (n=121)；general_quirky 498→468 (n=619)；iot_cleaning 164→163 (n=203)；play_game 123→122 (n=138)；qa_factoid 494→492 (n=520)；social_post 223→220 (n=260)；transport_ticket 154→153 (n=155)。

|集合|配置|新P/复用|生成wall秒|秒/逻辑条|卡/实例|新tokens|分配卡时代理|HTTP均值/p95秒|剩余/ETA秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|R01C-fit624|Q15|0/624|22.778|0.036504|8/2|0|0.046861|3.980/5.581|0/0|
|R01C-fit624|Q10|0/624|97.041|0.155515|8/2|0|0.206874|17.038/21.254|0/0|
|R01C-fit624|Q5|0/624|80.625|0.129207|8/2|0|0.172316|14.295/17.623|0/0|
|FoldB1078|Q15|0/1078|206.443|0.191505|8/2|0|0.447935|22.453/26.099|0/0|
|FoldB1078|Q10|1078/0|174.060|0.161466|8/2|4834008|0.378629|18.975/21.782|0/0|
|FoldB1078|Q5|1078/0|128.276|0.118994|8/2|4292687|0.281479|14.148/15.926|0/0|
|FoldA1075|Q15|1075/0|206.760|0.192334|8/2|5412658|0.454146|22.831/25.624|0/0|
|FoldA1075|Q10|1075/0|165.346|0.153810|8/2|4806915|0.354529|17.870/20.971|0/0|
|FoldA1075|Q5|1075/0|127.768|0.118854|8/2|4278676|0.283098|14.294/16.203|0/0|
|TEST13078|Q15|13078/0|2482.639|0.189833|8/2|65615507|5.405106|23.668/26.444|0/0|
|TEST13078|Q10|13031/47|2064.651|0.157872|8/2|58143687|4.568848|20.077/22.338|0/0|
|TEST13078|Q5|12873/205|1665.572|0.127357|8/2|51198368|3.700879|16.458/18.280|0/0|

每臂原配置为两TP4实例、每实例cap64。fit及FoldB Q15的wall/token/卡时是历史来源成本，当前新增为0；历史tokens在COST.aggregate.json另列，不重复计费。各集合新P共44363（FoldB2156/FoldA3225/TEST38982），不能把这个请求数称为独立样本量。TRAIN两个口径重叠且没有新模型成本。

FoldA18条同wire未提前复用，实际3225新P、唯一3207，原成本不改。TEST252精确复用在launch前冻结；原成功/unknown请求不重发。CPU准备、历史失败attempt、来源编码成本仍保留在各原cost目录，本补件CPU见TERMINAL.json。钱、能耗、未返回cache、agent和归档CPU未知。卡时是分配代理，离散AIcore采样不代表持续有效利用率。

原收尾在数据核验之后重新要求两guard空闲/服务running-waiting为0，与正常P后继占用冲突。后继运行跨越原失败时刻，但原远端stderr未落盘，具体errno未知。交接时guard/lease/session/boot和7个原Q容器终态证据状态：PENDING_FAILURE_ATTRIBUTION；DATA-COMPLETE只描述数据与证据完整性，不把旧失败改判PASS。原错误尚缺：original runtime failure attribution；不得重跑探针或实验来倒推历史异常。

保留TRAIN/report.md、FoldA/report.md及全部历史失败/raw；效率局限见TEST/EFFICIENCY-DIAGNOSTIC.md和MONITOR-DEADLINE-CORRECTION证据（原监测空档不补造）。本目录报告与聚合JSON可供主审后脱敏发布；私有证据、INPUTS、manifest及原始sample text/IDs不公开。本职责未修改公开仓。

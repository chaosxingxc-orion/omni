# 四路TP4滚动效率审计：离线交付

证据支持客户端并发受限、请求decode时间主导，以及历史空白失效造成大量输出浪费；尚不足以判定引擎参数应如何调优。四实例共16卡。此次仅本地只读分析，先写ACK；无SSH/API/NPU、无子智能体、无运行源/配置/数据修改、无git提交。READY表示审计完成，非新任务准入或收益已验证。

## 证据范围与分母

- 资源UTC时间：2026-11-09T16:35:36.798948+00:00 至 2026-11-09T21:25:16.849777+00:00；31快照、62主机采样、124 lane点、496卡次，跨度17380.051秒。30相邻间隔中位603.153秒，最短154.563秒；不假设严格600秒。
- 前6快照没有token/latency counters；120个lane相邻段中24缺counter、96可算delta；80同任务饱和端点、5同任务非饱和/开端/尾部、11跨任务或终态空窗。未发现负delta/已知service变化，未发现重置不等于证明未重启。
- 完整归档7份，7×545样本×3分支=11445完成分支，10105 protocol valid；不同策略/seed，不能合并当独立样本或语义准确率。归档起止 2026-11-09T16:00:18.257358+00:00 至 2026-11-09T20:47:50.313228+00:00。TRAIN96另为失败终态1246/3240，非1080完整训练。
- 主输入为指定rolling-state/resources、lanes、tasks、TRAIN96-REAL-OPERATION及两个ENGINE分析文件；rolling-code按实际目录解析为delivery/s-single-intent-execution-v1/rolling-code。补充归档只顺序读取LAUNCH/TERMINAL/rows/resource-preparation，逐条聚合，不解包，不打印raw。INPUT-MANIFEST.json记录实际读取输入SHA和大小。
- 快照没有service_id/process_start_time。脚本用launch的boot_id/service_id/service_pid/session_id关联任务，再验counter单调性；“稳定”准确含义为同任务、同服务回执、无可见重置、两端饱和。不能保证中间没有短暂空闲，也不能排除重启后counter已追平的隐蔽重置。主机采样仍有时间偏差。

## 相邻delta与累计分开

实际计算：Δcounter=counter(t1)-counter(t0)；token/s=Δtokens/Δt，sec/branch=Δt/Δdecode_count；queue/prefill/decode均值=各自Δsum/Δcount；prefix=Δhits/Δqueries（token口径，非请求命中率）；KV是gauge，只报端点值/范围，preemption报Δ。缺counter、负delta、service变化不做跨段差分；各窗口按count/时长加权，绝不平均ratio。完成计数取latency histogram count，并核对queue/prefill/decode三类一致。
Histogram完成请求可能在窗口之前入场；均值不是所有在途请求延迟，token流量/完成数也不等于精确完成请求长度。以下80段不是连续时间积分；每行单个TP4实例，非单卡吞吐，多lane时长不能当共同walltime。

| Lane | 段/秒 | 完成branch | sec/branch | prompt token/s | gen token/s | queue/prefill/decode秒 | prefix | KV端点范围 | Δ抢占 |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| A | 21/12217.5 | 2852 | 4.284 | 1032.4 | 518.0 | 0.145/0.966/271.28 | 4.15% | 17.36%–24.57% | 0 |
| B | 21/12216.5 | 3331 | 3.668 | 1241.2 | 525.2 | 0.106/0.964/231.66 | 4.06% | 19.33%–24.51% | 0 |
| C | 20/11613.0 | 2759 | 4.209 | 1046.7 | 553.2 | 0.307/0.927/269.94 | 5.05% | 16.97%–24.85% | 0 |
| D | 18/10406.4 | 2634 | 3.951 | 1165.2 | 569.7 | 0.727/1.060/298.60 | 3.80% | 18.14%–37.21% | 0 |

decode约占三段均值总和99.4%至99.6%，但请求时间处于decode不等于算术核饱和。可见KV最高37.21%、Δ抢占0；暂无提高gpu-memory-utilization=0.92的证据。prefix约3.8%至5.1%，缺逐请求audio/前缀/排序证据，不承诺改变prefix会有收益。

最后资源点的引擎生命周期累计如下，包含先前任务，不能当本任务总量或与delta直接混比：

| Lane | lifetime完成数 | 累计queue/prefill/decode秒 | 累计prefix |
|---|---:|---|---:|
| A | 3907 | 0.661/0.947/275.50 | 4.23% |
| B | 4778 | 0.524/0.935/225.38 | 4.12% |
| C | 4192 | 0.616/0.920/280.83 | 4.90% |
| D | 4472 | 0.889/1.017/274.38 | 3.92% |

原ENGINE-CUMULATIVE源快照17:31:31 UTC是更早lifetime累计。原ENGINE-INTERVAL的603.047793秒段四lane×7项（完成数、prompt/gen、prefix、三段均值）复算全部一致。详见INTERVALS.json/.csv的120段及排除理由。

## AI Core、开端、尾部与空窗

| Lane | 卡次 | AI Core均值/最高 | >=80%卡次 | 饱和端点卡次/均值 | running=waiting=0点数 |
|---|---:|---|---:|---|---:|
| A | 124 | 33.58%/63% | 0 | 120/33.77% | 0 |
| B | 124 | 33.23%/66% | 0 | 120/34.33% | 1 |
| C | 124 | 32.44%/63% | 0 | 116/33.48% | 1 |
| D | 124 | 31.02%/78% | 0 | 108/32.41% | 1 |

124个lane点waiting全部0：引擎没有等待调度队列，并不等于客户端没请求；64/96在途只证明客户端上限用满。AI Core并非CUDA GPU utilization；gpu-memory-utilization只是引擎参数名，不是AI Core百分比；HBM容量占用也不是带宽。496卡次0次>=80%，不能外推未采样时刻。
尾部：D 17:51:37 UTC有44 running，C 18:21:45有35，A 18:51:53有5。完全空队列：B 18:11:43、D 19:22:03、C 20:22:23均running=waiting=0，对应交接/失败。D 19:32:07有1 running属新任务开端，与96稳态分开。终态→后任务launch间隔如下，不含初始化/首条串行，也不是精确空闲积分：

| Lane | 前任务→后任务 | 间隔秒 | 前终态/unknown |
|---|---|---:|---|
| A | timezone-single-v1 → grpo-fewshot-seed2026110901 | 24.865 | COMPLETE/0 |
| A | grpo-fewshot-seed2026110901 → grpo-fewshot-seed2026110905 | 22.327 | COMPLETE/0 |
| B | fewshot-complement-v1 → grpo-fewshot-seed2026110902 | 23.890 | COMPLETE/0 |
| B | grpo-fewshot-seed2026110902 → grpo-fewshot-seed2026110906 | 24.394 | COMPLETE/0 |
| C | historical-C-545 → compact-json-v1 | 22.122 | COMPLETE/0 |
| C | compact-json-v1 → grpo-fewshot-seed2026110907 | 212.174 | COMPLETE/0 |
| D | historical-D-545 → train-s_fit-single-fewshot-v1-retry2 | 172.199 | COMPLETE/0 |
| D | train-s_fit-single-fewshot-v1-retry2 → grpo-fewshot-seed2026110904 | 887.877 | FAILED/1 |

通常22至25秒；C一次212秒、D首次172秒，D失败后888秒，不能把四路整段称为持续满负载。故障交接须核验旧进程退出、boot、请求终态和来源；不能为利用率删guard或盲重发。

## D真实96并发与64并发545：描述性比较

| 口径 | UTC时间范围 | 段/完成数 | sec/branch | prompt/gen token/s | queue/prefill/decode秒 | prefix/KV | AI Core卡次/点均值 |
|---|---|---:|---:|---|---|---|---|
| D64原545 | 2026-11-09T17:31:31.517386+00:00 至 2026-11-09T17:41:34.565179+00:00 | 1/166 | 3.633 | 1350.1/572.2 | 0.026/1.058/233.17 | 4.01%/24.18%–24.18% | 8/37.50% |
| D96 TRAIN1080 | 2026-11-09T18:01:40.775583+00:00 至 2026-11-09T19:12:00.755323+00:00 | 7/1118 | 3.775 | 1251.0/661.6 | 1.683/1.099/351.71 | 3.43%/32.52%–37.21% | 32/33.62% |
| D64全部545窗口 | 2026-11-09T17:31:31.517386+00:00 至 2026-11-09T21:15:13.649679+00:00 | 11/1516 | 4.081 | 1106.8/507.1 | 0.022/1.032/259.43 | 4.08%/18.14%–24.21% | 52/32.15% |

D96对原D64生成token/s高15.62%，sec/branch却高3.90%（变慢）；若换为全部D64窗口，生成高30.47%，sec/branch低7.50%。换分母即改变解释，不能归因96。token流/完成数约2497 vs2079，TRAIN输出更长的流量迹象，但这不是同ID长度配对。
原D64完整终态1635/1635，3.767 sec/branch、588.1 gen token/s。D96终态在 2026-11-09T19:15:13.788032+00:00：FAILED，1246/3240完成、1245 valid、1 unknown、1993 absent、1247已接触，总4796.314秒即3.849秒/已完成branch。不能称完整1080训练或成功吞吐。TRAIN96-REAL-OPERATION仅保存早期1/3240、189.544 sec/branch，不可覆盖后续失败；首response在launch后152.762秒。
引擎argv审计为max-num-seqs=128、max-num-batched-tokens=2048、TP4。64/96客户端至多用序列容量50%/75%；这是实际并发供给上限，不是测得每步decode batch，也不是2048 token预算只用一半。D96 KV33%至37%、无抢占，可提出后续并发假设，但任务/长度/prompt/seed/完成选择均不同，且配置600秒deadline的timeout未解决，不能直接推广128或线性声称+33%吞吐。

## 输出失效与空白浪费

valid是协议通过，非语义正确。空白以wire_content字符聚合，不冒称token；reasoning来自usage。7完整归档没有整段content为空的响应；entity type and filler must be nonempty text表示entity字段空，不能报成返回空字符串。

| Lane/任务 | 完成/valid | length/字段空 | 失效completion/总completion | 占比 | 长尾空白失效条数/字符 | sec/valid |
|---|---|---|---:|---:|---|---:|
| D/historical-D-545 | 1635/1620 | 8/7 | 39476/3622396 | 1.09% | 8/16940 | 3.802 |
| B/fewshot-complement-v1 | 1635/1621 | 7/7 | 33693/3562195 | 0.95% | 7/21389 | 3.716 |
| C/historical-C-545 | 1635/1016 | 592/27 | 2453268/4430381 | 55.37% | 592/1130918 | 8.439 |
| A/timezone-single-v1 | 1635/1006 | 598/31 | 2481444/4431992 | 55.99% | 598/1095753 | 8.444 |
| C/compact-json-v1 | 1635/1620 | 3/12 | 41986/4066224 | 1.03% | 0/0 | 4.344 |
| A/grpo-fewshot-seed2026110901 | 1635/1608 | 13/14 | 68496/3461967 | 1.98% | 13/53067 | 4.285 |
| B/grpo-fewshot-seed2026110902 | 1635/1614 | 7/14 | 41955/2854170 | 1.47% | 7/32509 | 3.501 |

历史C592条length都有至少64字符尾空白，合1,130,918字符；A timezone598条，合1,095,753字符。失效completion占55.37%/55.99%；提高原始token/s可能只是更快输出错误。该占比只是全部失效输出成本均消失的token节约乐观上界，非可实现时间收益，不能全部当空白token。
compact-json同C服务真实完成1635分支，0长尾空白、3length、15invalid；其他fewshot完整任务失效completion约0.95%至1.98%。新任务不能套早期55%余量。compact更多reasoning、4.305 sec/branch，不能断言压短JSON总会加速；策略/时间均有混杂。不要无证据全局加thinking、截断4096或改引擎空白参数；逐ID保留原始输出与语义回归。

## CPU供给、重复校验与初始化

- RESERVE at 2026-11-09T21:17:19.142713+00:00：ready=16任务/40299分支，cpu_preparing=0、四路next_staged=true。这里0表示库存已备，不是生产饥饿；12h coverage仍NOT_YET_PROVEN，库存时间估计不是持续供给保证。
- 10个已有stage回执的已launch任务都提前stage，领先52至7679秒；TRAIN领先287.607秒。C长交接时seed07也提前2329秒stage；没准备好数据不足以解释空窗。现有CPU prepared、固定镜像预算和stage应复用，不重复抽样/物化。
- 11445分支CPU prepare均值12.76至13.82毫秒，p95约15至16毫秒；线程池排队p95约0.174至0.322毫秒，HTTP均值212.68至326.47秒；D96终态executor累计排队0.209秒/1246。没有持续CPU饥饿证据；缺host CPU/RSS/磁盘及NPU HCCL/HBM带宽时间序列，不能因此完全排除CPU/通信瓶颈。
- 重复校验有源码证据：train_runtime.py:15遍历源proof与所有payload，验SHA、P3、audio、hold与wire；bind_train.py:10执行preflight，run_s_native.py:267又执行。run preparation_seconds在272行结束，后面的receipt/session/guard校验、import及首请求准备不在该数内；bind单独耗时未知。
- 完整545 run准备11.718至12.305秒，仅wall的0.137%至0.218%；TRAIN为25.712秒，占失败wall0.536%。可另做manifest/packet/source/code SHA锁定的不可变CPU校验receipt缓存，运行时仍验boot/service/guard/unknown。完全移走已计run准备也只是该小上界；不能臆测bind耗时再加倍。
- execute_samples在148行先await第1条响应，再开并发worker。7完整545首HTTP55.09至78.18秒，launch至首row90.74至114.51秒；D96首response记录延迟152.762秒。按64理想槽归一，首条串行造成约54.22至76.96 lane-equivalent秒欠供/545任务（0.63%至1.26%wall）；这是容量估计，不是实测wall收益。若另设首波并发，先查明协议闸门目的，不取消fail-fast或unknown保护。
- host_driver.py:87每runner给8CPU，评分另3CPU；没有加CPU的证据。scheduler已在CPU预stage、后继启动后异步score。应围绕明确terminal/blocked准入日志缩短可优化交接，不把评分、完整校验、未知请求核验等一概删除。

## 下一项单机制建议

优先建议供执行主线程评审：下一个已授权且未接触的独立TRAIN任务，仅将客户端timeout_seconds 600→900作为候选；有效并发固定96、TP4、引擎128序列/2048 token调度、4096输出/3072thinking保持原值。未部署。原native配置仍标64，实际host_driver有TRAIN96 override，必须记录effective concurrency。900未证明最优。
依据：TRAIN配置timeout_seconds=600且出现TimeoutError，已触发stop admission，FAILED1246/3240且1unknown，至后继独立任务launch888秒。先减少明确中断，再判断并发上调。若慢请求能在600至900秒完成，可避免这类提前停供；新增完成数、速度、AI Core与金额未知，1993absent不是可保证追回收益。未取得该unknown请求的精确HTTP耗时，不能把配置600秒称为实测600秒延迟。风险：每慢请求按deadline上限额外占槽最多300秒，可能更迟发现卡死；不修底层问题、不启自动重试、不重发原unknown、不移除fsync/锁/guard。
同ID锁定：预先冻结真实TRAIN source ID/P3 candidate_index、P_response/packet/wire/snapshot/manifest SHA、seed/temperature和模型镜像/服务身份。只在另行已授权的同ID科研配对中，用独立attempt和唯一guard顺序比600/900，唯一变化是deadline。当前1247已接触集合不用于续跑；known response只离线复核，unknown不得盲重发。若没有获准的新同ID模型对照，只有观察性结论；不为填算力重复跑基线。
验收固定分母的完成/有效/语义正确branch、600至900秒新增完成、timeout/unknown/absent、HTTP分布、总wall/card-hours及相邻600秒delta；排除首条/尾部/service变动。保留全部历史成本；语义退步、unknown增加或超时不改善则不推广。READY只是本次分析交付，不是解除失败任务限制。
后续性能假设：可靠性通过后才考虑客户端TRAIN96→128单机制同ID验证；序列容量至多+33.3%，实际吞吐收益未知，可为0或负。现有16%/30%token差异不作为预期收益。不同时改引擎batch、KV、CPU或空白规则；没有每步batch/HCCL/带宽证据就不先调引擎。空白约束是独立协议实验，只对仍有证据的失败ID评估，不能混入并发试验。

## 当前lane回执与成本

| Lane | UTC观测 | 整样本/545 | branch/1635 | 累计sec/branch | 最近区间sec/branch | 最近completion token/s |
|---|---|---:|---:|---:|---:|---:|
| A | 2026-11-09T21:18:19.783069+00:00 | 160/545 | 498/1635 | 3.631 | 3.276 | 506.6 |
| B | 2026-11-09T21:26:16.439036+00:00 | 501/545 | 1512/1635 | 3.963 | 4.602 | 464.4 |
| C | 2026-11-09T21:14:33.052321+00:00 | 255/545 | 783/1635 | 3.845 | 4.295 | 472.3 |
| D | 2026-11-09T21:24:17.869952+00:00 | 525/545 | 1595/1635 | 4.299 | 4.628 | 498.3 |

lane回执时间不同，部分晚于最后资源点，不伪装同一时刻；每路4卡/1实例。累计秒/branch含开端等成本，最近区间是host-driver保存的同run观测，非engine delta。现金单价/金额、未来ETA及12h覆盖未知。TRAIN失败wall4796.314秒=5.329卡小时，1246已知响应tokens8,884,560；unknown真实模型耗时/token若未落账仍未知，不能抹掉失败成本。

## 复算与限制

analyze.py离线stdlib分析；build_report.py生成本报告、建议和READY。RESULT.json存聚合与unknown；INTERVALS.json/.csv存120相邻段，RESOURCE-POINTS.json存496卡次，INPUT-MANIFEST.json存输入SHA。归档原始输出不进入本交付目录。
复算命令（仓库根目录，本地CPU）：
    PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/analyze.py --self-test
    PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/analyze.py
    PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/build_report.py

本地算术/缺失/重置/AI Core解析自检通过，原603秒4lane×7数值一致、7归档token与终态一致、496卡次完整、D96失败保留。不是固定saea镜像正式NPU/模型测试。未来复算读取届时可变回执，as-of范围可能变化；本次SHA对应本次实际读取版本。因果并发收益、连续AI Core积分、每步batch/通信/带宽、TRAIN全量结果、bind独立耗时、精确空白token数和全部任务语义正确吞吐均未知。

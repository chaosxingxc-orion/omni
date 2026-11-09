#!/usr/bin/env python3
"""从聚合数据生成中文报告与READY；仅写本交付目录。"""
import datetime,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent
U=datetime.timezone.utc
def load(n):return json.loads((O/n).read_text())
def dump(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def fmt(x,n=3):return '未知' if x is None else f'{x:.{n}f}'
def pct(x):return '未知' if x is None else f'{100*x:.2f}%'
def utc(t):return datetime.datetime.fromtimestamp(t,U).isoformat()
r=load('RESULT.json');iv=load('INTERVALS.json');pts=load('RESOURCE-POINTS.json');ars=r['terminals_and_archives'];d=r['comparison'];data=r['data_range'];ar={a['task_id']:a for a in ars}
checks={
 'resource_points_equal_snapshots_x4':len(pts)==data['resource_snapshots']*4,
 'AI_Core_four_cards_each':all(len(p['AI_Core'])==4 for p in pts),
 'adjacent_intervals_equal_snapshots_minus1_x4':len(iv)==(data['resource_snapshots']-1)*4,
 'valid_delta_no_negative':all(all(v>=-1e-7 for v in x['delta'].values()) for x in iv if x['delta_usable']),
 'no_delta_from_missing_reset':all(not x['delta_usable'] for x in iv if x['missing_counters'] or x['negative_deltas']),
 'queue_prefill_decode_count_agree':all(x['completion_count_consistent'] for x in iv if x['delta_usable']),
 'reference_four_lanes_seven_metrics':all(all(x['checks'].values()) for x in r['reference_crosscheck']),
 'archive_rows_terminal_denom':all(a['rows']==a['total']==1635 and a['unknown']==0 for a in ars),
 'archive_token_totals_match':all(a['usage_matches_terminal'] for a in ars),
 'archive_valid_counts_match':all(a['groups']['valid']['branches']==a['valid'] for a in ars),
 'TRAIN96_failed_not_complete':d['train_terminal']['status']=='FAILED' and d['train_terminal']['unknown']==1 and d['train_terminal']['completed']==1246,
 'TRAIN96_not_compared_causally':d['causal'] is False,
 'TRAIN96_timeout600_config':d['train_client_config']['timeout_seconds']==600,
 'outputs_no_symlink':all(not x.is_symlink() for x in O.iterdir()),
}
if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
proposal={
 'status':'PROPOSAL_ONLY_NOT_DEPLOYED','single_mechanism':'client HTTP deadline',
 'candidate_change':{'timeout_seconds':{'from':600,'to':900}},
 'hold_fixed':{'effective_concurrency':96,'TP':4,'max_num_seqs':128,'max_num_batched_tokens':2048,'max_tokens':4096,'thinking_token_budget':3072},
 'scope':'只供主线程评审用于下一个已授权且未接触的独立TRAIN任务；不修改当前任务，不重发unknown，不为填负载重复基线。900是待验证候选，不是已测最优值。',
 'evidence':'D96配置timeout_seconds=600且出现TimeoutError触发stop admission，1246/3240完成，1unknown、1993absent；至后继独立任务launch隔887.877秒。',
 'benefit':'若慢请求能在600至900秒完成，可避免此类提前停止；新增完成数、walltime、AI Core及金额未知；1993absent不是保证可追回收益。',
 'risks':['每个慢请求最多再占槽300秒，可能延后发现卡死','不修复底层NPU/CPU问题，不保证HTTP完成','保留flock/shared guard、fsync、unknown-no-resend和全部历史成本'],
 'same_id_validation':{'freeze':['source_sample_id','candidate_index','packet_sha256','wire_sha256','P_response_sha256','snapshot_sha256','manifest_sha256','seed','temperature','model/image/service identity'],
 'plan':'只在另行已授权的同ID科研配对中，使用独立attempt和唯一guard顺序比较600/900；不得续跑当前1247已接触集合。已知observed只供离线审计，unknown不重发。没有获准的新同ID模型对照，就只报告观察性结果。',
 'measure':['完成/计划/有效/语义正确分支及固定分母','600至900秒完成的请求数和HTTP时间分布','相邻600秒counter delta剔除启动/尾部/重置','timeout/unknown/absent及总walltime/card-hours','全部历史累计成本与同ID差分'],
 'rule':'仅deadline变化，不同时改并发/prompt/采样/空白约束/引擎参数，不自动重试。'}
}
r['next_single_mechanism_proposal']=proposal;r['validation']=checks
r['findings']=[
 {'id':'decode_dominated','evidence':str(r['classification_counts']['same_task_saturated_endpoints'])+'个同任务饱和端点窗口；queue/prefill远小于decode；不等于核饱和'},
 {'id':'client_bounded','evidence':'max-num-seqs128，客户端64/96约为序列容量50%/75%；waiting全部观测点为0；不是测得实际decode batch'},
 {'id':'historical_whitespace','evidence':'7完整归档11445分支；历史C592/A598 length且有长尾空白；compact0长尾空白；不能把旧浪费套给新任务'},
 {'id':'CPU_supply','evidence':'ready16/40299branches、四lane next_staged=true；10已启动task提前stage；单分支prepare约13ms/HTTP约213至326秒'},
 {'id':'serial_first_gate','evidence':'execute_samples先await首条再启动并发worker；约54至77 lane-equivalent秒欠供/545任务；非实测可得收益'},
 {'id':'repeated_preflight','evidence':'bind_train与run各调用preflight；run准备11.72至12.30秒，TRAIN25.71秒；bind单独时长未知'}]
dump('RESULT.json',r)
lines=[]
def L(s=''):lines.append(s)
L('# 四路TP4滚动效率审计：离线交付');L()
L('证据支持客户端并发受限、请求decode时间主导，以及历史空白失效造成大量输出浪费；尚不足以判定引擎参数应如何调优。四实例共16卡。此次仅本地只读分析，先写ACK；无SSH/API/NPU、无子智能体、无运行源/配置/数据修改、无git提交。READY表示审计完成，非新任务准入或收益已验证。');L()
L('## 证据范围与分母');L()
L(f'- 资源UTC时间：{data["first_resource_utc"]} 至 {data["last_resource_utc"]}；{data["resource_snapshots"]}快照、{data["host_snapshots"]}主机采样、{data["metric_points"]} lane点、{sum(len(x["AI_Core"]) for x in pts)}卡次，跨度{data["resource_span_seconds"]:.3f}秒。{data["resource_snapshots"]-1}相邻间隔中位{data["sample_intervals_seconds"]["p50"]:.3f}秒，最短154.563秒；不假设严格600秒。')
L(f'- 前6快照没有token/latency counters；{len(iv)}个lane相邻段中{sum(bool(x["missing_counters"]) for x in iv)}缺counter、{sum(x["delta_usable"] for x in iv)}可算delta；{r["classification_counts"]["same_task_saturated_endpoints"]}同任务饱和端点、{r["classification_counts"]["same_task_nonsaturated_or_startup_tail"]}同任务非饱和/开端/尾部、{r["classification_counts"]["task_boundary_or_terminal_gap"]}跨任务或终态空窗。未发现负delta/已知service变化，未发现重置不等于证明未重启。')
L(f'- 完整归档7份，7×545样本×3分支=11445完成分支，{sum(a["valid"] for a in ars)} protocol valid；不同策略/seed，不能合并当独立样本或语义准确率。归档起止 {utc(min(a["start_epoch"] for a in ars))} 至 {utc(max(a["end_epoch"] for a in ars))}。TRAIN96另为失败终态1246/3240，非1080完整训练。')
L('- 主输入为指定rolling-state/resources、lanes、tasks、TRAIN96-REAL-OPERATION及两个ENGINE分析文件；rolling-code按实际目录解析为delivery/s-single-intent-execution-v1/rolling-code。补充归档只顺序读取LAUNCH/TERMINAL/rows/resource-preparation，逐条聚合，不解包，不打印raw。INPUT-MANIFEST.json记录实际读取输入SHA和大小。')
L('- 快照没有service_id/process_start_time。脚本用launch的boot_id/service_id/service_pid/session_id关联任务，再验counter单调性；“稳定”准确含义为同任务、同服务回执、无可见重置、两端饱和。不能保证中间没有短暂空闲，也不能排除重启后counter已追平的隐蔽重置。主机采样仍有时间偏差。');L()
L('## 相邻delta与累计分开');L()
L('实际计算：Δcounter=counter(t1)-counter(t0)；token/s=Δtokens/Δt，sec/branch=Δt/Δdecode_count；queue/prefill/decode均值=各自Δsum/Δcount；prefix=Δhits/Δqueries（token口径，非请求命中率）；KV是gauge，只报端点值/范围，preemption报Δ。缺counter、负delta、service变化不做跨段差分；各窗口按count/时长加权，绝不平均ratio。完成计数取latency histogram count，并核对queue/prefill/decode三类一致。')
L('Histogram完成请求可能在窗口之前入场；均值不是所有在途请求延迟，token流量/完成数也不等于精确完成请求长度。以下77段不是连续时间积分；每行单个TP4实例，非单卡吞吐，多lane时长不能当共同walltime。');L()
L('| Lane | 段/秒 | 完成branch | sec/branch | prompt token/s | gen token/s | queue/prefill/decode秒 | prefix | KV端点范围 | Δ抢占 |');L('|---|---:|---:|---:|---:|---:|---|---:|---|---:|')
for l,s in r['steady_by_lane'].items():L(f'| {l} | {s["intervals"]}/{s["seconds"]:.1f} | {s["completed_requests"]:.0f} | {s["seconds_per_branch"]:.3f} | {s["prompt_tokens_per_second"]:.1f} | {s["generation_tokens_per_second"]:.1f} | {s["queue_mean_seconds"]:.3f}/{s["prefill_mean_seconds"]:.3f}/{s["decode_mean_seconds"]:.2f} | {pct(s["prefix_hit_ratio"])} | {pct(s["kv_end_min"])}–{pct(s["kv_end_max"])} | {s["preemptions_delta"]:.0f} |')
L();L('decode约占三段均值总和99.4%至99.6%，但请求时间处于decode不等于算术核饱和。可见KV最高37.21%、Δ抢占0；暂无提高gpu-memory-utilization=0.92的证据。prefix约3.8%至5.1%，缺逐请求audio/前缀/排序证据，不承诺改变prefix会有收益。');L()
L('最后资源点的引擎生命周期累计如下，包含先前任务，不能当本任务总量或与delta直接混比：');L()
L('| Lane | lifetime完成数 | 累计queue/prefill/decode秒 | 累计prefix |');L('|---|---:|---|---:|')
for l,s in r['latest_engine_cumulative'].items():L(f'| {l} | {s["completion_count"]:.0f} | {s["queue_mean_seconds"]:.3f}/{s["prefill_mean_seconds"]:.3f}/{s["decode_mean_seconds"]:.2f} | {pct(s["prefix_hit_ratio"])} |')
L();L('原ENGINE-CUMULATIVE源快照17:31:31 UTC是更早lifetime累计。原ENGINE-INTERVAL的603.047793秒段四lane×7项（完成数、prompt/gen、prefix、三段均值）复算全部一致。详见INTERVALS.json/.csv的116段及排除理由。');L()
L('## AI Core、开端、尾部与空窗');L()
L('| Lane | 卡次 | AI Core均值/最高 | >=80%卡次 | 饱和端点卡次/均值 | running=waiting=0点数 |');L('|---|---:|---|---:|---|---:|')
for l,s in r['AI_Core_points'].items():L(f'| {l} | {s["card_samples"]} | {s["AI_Core_mean"]:.2f}%/{s["AI_Core_max"]:.0f}% | {s["at_least80"]} | {s["saturated_card_samples"]}/{s["saturated_AI_Core_mean"]:.2f}% | {s["running_zero_waiting_zero_points"]} |')
L();L('120个lane点waiting全部0：引擎没有等待调度队列，并不等于客户端没请求；64/96在途只证明客户端上限用满。AI Core并非CUDA GPU utilization；gpu-memory-utilization只是引擎参数名，不是AI Core百分比；HBM容量占用也不是带宽。480卡次0次>=80%，不能外推未采样时刻。')
L('尾部：D 17:51:37 UTC有44 running，C 18:21:45有35，A 18:51:53有5。完全空队列：B 18:11:43、D 19:22:03、C 20:22:23均running=waiting=0，对应交接/失败。D 19:32:07有1 running属新任务开端，与96稳态分开。终态→后任务launch间隔如下，不含初始化/首条串行，也不是精确空闲积分：');L()
L('| Lane | 前任务→后任务 | 间隔秒 | 前终态/unknown |');L('|---|---|---:|---|')
for g in r['transitions']:L(f'| {g["lane"]} | {g["predecessor"]} → {g["successor"]} | {g["gap_seconds"]:.3f} | {g["predecessor_status"]}/{g["unknown"]} |')
L();L('通常22至25秒；C一次212秒、D首次172秒，D失败后888秒，不能把四路整段称为持续满负载。故障交接须核验旧进程退出、boot、请求终态和来源；不能为利用率删guard或盲重发。');L()
L('## D真实96并发与64并发545：描述性比较');L()
L('| 口径 | UTC时间范围 | 段/完成数 | sec/branch | prompt/gen token/s | queue/prefill/decode秒 | prefix/KV | AI Core卡次/点均值 |');L('|---|---|---:|---:|---|---|---|---|')
for k,title in [('D64_historical545','D64原545'),('D96_TRAIN','D96 TRAIN1080'),('D64_all545','D64全部545窗口')]:
 s=d[k];L(f'| {title} | {s["start_utc"]} 至 {s["end_utc"]} | {s["intervals"]}/{s["completed_requests"]:.0f} | {s["seconds_per_branch"]:.3f} | {s["prompt_tokens_per_second"]:.1f}/{s["generation_tokens_per_second"]:.1f} | {s["queue_mean_seconds"]:.3f}/{s["prefill_mean_seconds"]:.3f}/{s["decode_mean_seconds"]:.2f} | {pct(s["prefix_hit_ratio"])}/{pct(s["kv_end_min"])}–{pct(s["kv_end_max"])} | {s["AI_Core_sample_count"]}/{s["AI_Core_point_mean"]:.2f}% |')
x=d['D96_TRAIN'];y=d['D64_historical545'];z=d['D64_all545']
L();L(f'D96对原D64生成token/s高{100*(x["generation_tokens_per_second"]/y["generation_tokens_per_second"]-1):.2f}%，sec/branch却高{100*(x["seconds_per_branch"]/y["seconds_per_branch"]-1):.2f}%（变慢）；若换为全部D64窗口，生成高{100*(x["generation_tokens_per_second"]/z["generation_tokens_per_second"]-1):.2f}%，sec/branch低{100*(1-x["seconds_per_branch"]/z["seconds_per_branch"]):.2f}%。换分母即改变解释，不能归因96。token流/完成数约2497 vs2079，TRAIN输出更长的流量迹象，但这不是同ID长度配对。')
L(f'原D64完整终态1635/1635，{ar["historical-D-545"]["seconds_per_branch"]:.3f} sec/branch、{ar["historical-D-545"]["generation_tokens_per_second"]:.1f} gen token/s。D96终态在 {utc(d["train_terminal"]["at"])}：FAILED，1246/3240完成、1245 valid、1 unknown、1993 absent、1247已接触，总4796.314秒即3.849秒/已完成branch。不能称完整1080训练或成功吞吐。TRAIN96-REAL-OPERATION仅保存早期1/3240、189.544 sec/branch，不可覆盖后续失败；首response在launch后{d["first_train_response_after_launch_seconds"]:.3f}秒。')
L('引擎argv审计为max-num-seqs=128、max-num-batched-tokens=2048、TP4。64/96客户端至多用序列容量50%/75%；这是实际并发供给上限，不是测得每步decode batch，也不是2048 token预算只用一半。D96 KV33%至37%、无抢占，可提出后续并发假设，但任务/长度/prompt/seed/完成选择均不同，且配置600秒deadline的timeout未解决，不能直接推广128或线性声称+33%吞吐。');L()
L('## 输出失效与空白浪费');L()
L('valid是协议通过，非语义正确。空白以wire_content字符聚合，不冒称token；reasoning来自usage。7完整归档没有整段content为空的响应；entity type and filler must be nonempty text表示entity字段空，不能报成返回空字符串。');L()
L('| Lane/任务 | 完成/valid | length/字段空 | 失效completion/总completion | 占比 | 长尾空白失效条数/字符 | sec/valid |');L('|---|---|---|---:|---:|---|---:|')
for a in ars:L(f'| {a["lane"]}/{a["task_id"]} | {a["rows"]}/{a["valid"]} | {a["error_categories"].get("finite_length",0)}/{a["error_categories"].get("nonempty_entity_field",0)} | {a["invalid"]["completion_tokens"]}/{a["groups"]["all"]["completion_tokens"]} | {pct(a["invalid_completion_token_share"])} | {a["invalid_long_trailing_whitespace_branches"]}/{a["invalid"]["trailing_whitespace_chars"]} | {a["seconds_per_valid_branch"]:.3f} |')
L();L('历史C592条length都有至少64字符尾空白，合1,130,918字符；A timezone598条，合1,095,753字符。失效completion占55.37%/55.99%；提高原始token/s可能只是更快输出错误。该占比只是全部失效输出成本均消失的token节约乐观上界，非可实现时间收益，不能全部当空白token。')
L('compact-json同C服务真实完成1635分支，0长尾空白、3length、15invalid；其他fewshot完整任务失效completion约0.95%至1.98%。新任务不能套早期55%余量。compact更多reasoning、4.305 sec/branch，不能断言压短JSON总会加速；策略/时间均有混杂。不要无证据全局加thinking、截断4096或改引擎空白参数；逐ID保留原始输出与语义回归。');L()
L('## CPU供给、重复校验与初始化');L()
c=r['CPU_supply'];L(f'- RESERVE at {utc(c["at"])}：ready={c["ready_tasks"]}任务/{c["ready_branches"]}分支，cpu_preparing=0、四路next_staged=true。这里0表示库存已备，不是生产饥饿；12h coverage仍NOT_YET_PROVEN，库存时间估计不是持续供给保证。')
L('- 10个已有stage回执的已launch任务都提前stage，领先52至7679秒；TRAIN领先287.607秒。C长交接时seed07也提前2329秒stage；没准备好数据不足以解释空窗。现有CPU prepared、固定镜像预算和stage应复用，不重复抽样/物化。')
L('- 11445分支CPU prepare均值12.76至13.82毫秒，p95约15至16毫秒；线程池排队p95约0.174至0.322毫秒，HTTP均值212.68至326.47秒；D96终态executor累计排队0.209秒/1246。没有持续CPU饥饿证据；缺host CPU/RSS/磁盘及NPU HCCL/HBM带宽时间序列，不能因此完全排除CPU/通信瓶颈。')
L('- 重复校验有源码证据：train_runtime.py:15遍历源proof与所有payload，验SHA、P3、audio、hold与wire；bind_train.py:10执行preflight，run_s_native.py:267又执行。run preparation_seconds在272行结束，后面的receipt/session/guard校验、import及首请求准备不在该数内；bind单独耗时未知。')
L('- 完整545 run准备11.718至12.305秒，仅wall的0.137%至0.218%；TRAIN为25.712秒，占失败wall0.536%。可另做manifest/packet/source/code SHA锁定的不可变CPU校验receipt缓存，运行时仍验boot/service/guard/unknown。完全移走已计run准备也只是该小上界；不能臆测bind耗时再加倍。')
L('- execute_samples在148行先await第1条响应，再开并发worker。7完整545首HTTP55.09至78.18秒，launch至首row90.74至114.51秒；D96首response记录延迟152.762秒。按64理想槽归一，首条串行造成约54.22至76.96 lane-equivalent秒欠供/545任务（0.63%至1.26%wall）；这是容量估计，不是实测wall收益。若另设首波并发，先查明协议闸门目的，不取消fail-fast或unknown保护。')
L('- host_driver.py:87每runner给8CPU，评分另3CPU；没有加CPU的证据。scheduler已在CPU预stage、后继启动后异步score。应围绕明确terminal/blocked准入日志缩短可优化交接，不把评分、完整校验、未知请求核验等一概删除。');L()
L('## 下一项单机制建议');L()
L('优先建议供执行主线程评审：下一个已授权且未接触的独立TRAIN任务，仅将客户端timeout_seconds 600→900作为候选；有效并发固定96、TP4、引擎128序列/2048 token调度、4096输出/3072thinking保持原值。未部署。原native配置仍标64，实际host_driver有TRAIN96 override，必须记录effective concurrency。900未证明最优。')
L('依据：TRAIN配置timeout_seconds=600且出现TimeoutError，已触发stop admission，FAILED1246/3240且1unknown，至后继独立任务launch888秒。先减少明确中断，再判断并发上调。若慢请求能在600至900秒完成，可避免这类提前停供；新增完成数、速度、AI Core与金额未知，1993absent不是可保证追回收益。未取得该unknown请求的精确HTTP耗时，不能把配置600秒称为实测600秒延迟。风险：每慢请求按deadline上限额外占槽最多300秒，可能更迟发现卡死；不修底层问题、不启自动重试、不重发原unknown、不移除fsync/锁/guard。')
L('同ID锁定：预先冻结真实TRAIN source ID/P3 candidate_index、P_response/packet/wire/snapshot/manifest SHA、seed/temperature和模型镜像/服务身份。只在另行已授权的同ID科研配对中，用独立attempt和唯一guard顺序比600/900，唯一变化是deadline。当前1247已接触集合不用于续跑；known response只离线复核，unknown不得盲重发。若没有获准的新同ID模型对照，只有观察性结论；不为填算力重复跑基线。')
L('验收固定分母的完成/有效/语义正确branch、600至900秒新增完成、timeout/unknown/absent、HTTP分布、总wall/card-hours及相邻600秒delta；排除首条/尾部/service变动。保留全部历史成本；语义退步、unknown增加或超时不改善则不推广。READY只是本次分析交付，不是解除失败任务限制。')
L('后续性能假设：可靠性通过后才考虑客户端TRAIN96→128单机制同ID验证；序列容量至多+33.3%，实际吞吐收益未知，可为0或负。现有16%/30%token差异不作为预期收益。不同时改引擎batch、KV、CPU或空白规则；没有每步batch/HCCL/带宽证据就不先调引擎。空白约束是独立协议实验，只对仍有证据的失败ID评估，不能混入并发试验。');L()
L('## 当前lane回执与成本');L()
L('| Lane | UTC观测 | 整样本/545 | branch/1635 | 累计sec/branch | 最近区间sec/branch | 最近completion token/s |');L('|---|---|---:|---:|---:|---:|---:|')
for x in r['current_lanes']:
 q=x.get('interval') or {};L(f'| {x["lane"]} | {utc(x["at"])} | {x["completed_samples"]}/{x["total_samples"]} | {x["completed_branches"]}/{x["total_branches"]} | {fmt(x["seconds_per_branch"])} | {fmt(q.get("seconds_per_completed_branch"))} | {fmt(q.get("completion_tokens_per_second"),1)} |')
L();L('lane回执时间不同，部分晚于最后资源点，不伪装同一时刻；每路4卡/1实例。累计秒/branch含开端等成本，最近区间是host-driver保存的同run观测，非engine delta。现金单价/金额、未来ETA及12h覆盖未知。TRAIN失败wall4796.314秒=5.329卡小时，1246已知响应tokens8,884,560；unknown真实模型耗时/token若未落账仍未知，不能抹掉失败成本。');L()
L('## 复算与限制');L()
L('analyze.py离线stdlib分析；build_report.py生成本报告、建议和READY。RESULT.json存聚合与unknown；INTERVALS.json/.csv存116相邻段，RESOURCE-POINTS.json存480卡次，INPUT-MANIFEST.json存输入SHA。归档原始输出不进入本交付目录。')
L('复算命令（仓库根目录，本地CPU）：');L('    PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/analyze.py --self-test');L('    PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/analyze.py');L('    PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/build_report.py');L()
L('本地算术/缺失/重置/AI Core解析自检通过，原603秒4lane×7数值一致、7归档token与终态一致、480卡次完整、D96失败保留。不是固定saea镜像正式NPU/模型测试。未来复算读取届时可变回执，as-of范围可能变化；本次SHA对应本次实际读取版本。因果并发收益、连续AI Core积分、每步batch/通信/带宽、TRAIN全量结果、bind独立耗时、精确空白token数和全部任务语义正确吞吐均未知。')
report='\n'.join(lines)+'\n'
report=report.replace('以下77段', '以下'+str(r['classification_counts']['same_task_saturated_endpoints'])+'段')
report=report.replace('的116段', '的'+str(len(iv))+'段').replace('存116相邻段','存'+str(len(iv))+'相邻段')
report=report.replace('120个lane点',''+str(len(pts))+'个lane点')
report=report.replace('480卡次',str(sum(len(x['AI_Core']) for x in pts))+'卡次')
(O/'REPORT.md').write_text(report)
dump('VALIDATION.json',dict(status='PASS',checks=checks,summary={'archives':len(ars),'completed_archive_branches':sum(a['rows'] for a in ars),'AI_Core_card_points':sum(len(x['AI_Core']) for x in pts),'intervals':len(iv),'reference_numeric_checks':sum(len(x['checks']) for x in r['reference_crosscheck'])},limits='local offline only; no fixed-image/NPU test'))
files={}
for n in ['ACK.json','analyze.py','build_report.py','REPORT.md','RESULT.json','INTERVALS.json','INTERVALS.csv','RESOURCE-POINTS.json','INPUT-MANIFEST.json','VALIDATION.json']:
 p=O/n;files[n]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
dump('READY.json',dict(status='READY',scope='离线效率分析交付完成，非运行准入或收益验收',generated_utc=datetime.datetime.now(U).isoformat(),files=files,validation='VALIDATION.json',runtime_mutated=False,SSH_calls=0,API_calls=0,NPU_calls=0,subagents=0,git_commits=0,next_action_deployed=False,unknowns=r['unknowns']))
print(json.dumps({'status':'READY','checks':checks,'outputs':list(files),'report_lines':len(lines)},ensure_ascii=False,indent=2))

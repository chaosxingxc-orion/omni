# RSI 可观测工程：正常与 debug 模式

## 目标和配置边界

可观测性需要支持定位问题、比较优化前后及选择下一步实验。不能以“现有记录无法拆分”结束分析，也不能将推测写成已测阶段。可观测配置与模型、解码、数据和奖励配置分离；每次运行记录有效配置及哈希，运行中不切换。新配置不回写旧实验。

| 层面 | normal | debug |
| --- | --- | --- |
| 进度与成本 | 样本完成/总量、token、请求、资源、秒/条、ETA及依据 | 同 normal |
| 科学恢复证据 | 完整intent/response、checkpoint、来源与历史成本 | 完全相同，不因调试关闭 |
| 首次预检 | 保留原完整权重身份校验 | 完整哈希耗时/字节数、内存或磁盘缓存命中、强制校验及失效原因 |
| 客户端 | 已有关键阶段和锁/容量等待 | 细化音频读取/编码、JSON编码/解析、HTTP到响应头及读取正文，记录请求关联和安全元数据 |
| 服务端 | 实例指标窗口的sum/count与关键counter | 增加直方图bucket，供分布和尾延迟分析 |
| 观察频率 | 600秒或终态事件 | 默认仍600秒；细粒度优先由请求事件产生，避免主线程高频轮询 |

配置文件为 `configs/observability/normal-v1.json` 和 `debug-v1.json`。通过 `SAEA_OBSERVABILITY_MODE`、`SAEA_OBSERVABILITY_CONFIG` 选择；`SAEA_STAGE_TIMING_PATH` 指定每次进程独立的新日志路径。debug不修改采样参数或请求正文，不输出原始prompt、音频、答案、凭据或内部地址。

## 已确认的观测缺口

当前两实例 `/metrics` 已提供 queue、prefill、decode、inference、E2E、TTFT、inter-token latency直方图，以及prefix/mm cache命中和抢占计数。旧观察器只采集运行/等待请求与token总数，遗漏了已有指标。新增 `scripts/analysis/server_observability.py` 实测每实例183个series，支持起止快照差分、计数回退/series变化拒绝及缺失项显式标记。

客户端非流式HTTP到响应头不能称为TTFT。服务端直方图是实例内完成请求的汇总：不同指标可能重叠、分母不同，不能相加为总墙钟；有其他负载时不直接归因于本实验。必须同时匹配窗口边界、引擎身份、journal完成数和外部负载。逐请求server trace、音频encoder细分仍需单独采集，不能从聚合数据反推。

## 使用与验收

```bash
python scripts/analysis/server_observability.py snapshot \
  --endpoint instance0=http://127.0.0.1:8200/metrics \
  --window-id RUN-ATTEMPT --mode debug --out /owned/before.json
python scripts/analysis/server_observability.py diff \
  --before /owned/before.json --after /owned/after.json --out /owned/delta.json
```

端点由部署配置传入，示例不是固定实验身份。配置、运行/attempt、样本/请求、实例与时间窗口应能关联。RSI读取结构化事件与分项统计，先区分CPU准备、客户端容量等待、服务端排队、prefill/decode与持久化，再提出单组件优化；效果报告同时给出精度、有效吞吐和观测开销。

模式一致性测试需证明相同输入得到相同请求字节与输出；debug只增加观测。测试覆盖指标重启、缺失和空窗口，不把缺失置零。真实部署需先做有界debug窗口，再按同一名单比较normal/debug开销；尚未完成的逐请求服务端trace和实测开销不宣称已具备。


## 首次校验的实际CPU证据

最终合并修正的生成进程在首次预检期间累计读取约65.45 GB，启动4分39秒时CPU占用约92%，随后进入learner与请求socket。该证据来自进程I/O计数与文件句柄，尚不是精确分阶段计时；不能将全部时间都归到模型哈希。但它确认启动路径有显著CPU/读取成本，现已独立记录首次完整权重校验、进程内cache命中与失效原因。debug事件区分 `runtime_identity_full_hash_process_floor`、`forced`、`model_identity_changed`、`cache_file_changed`、`cache_missing` 和 `disk_identity_mismatch`；读取字节与文件数只在真实测量成功后记录，校验失败记录error，不写入成功缓存。不得为提速跳过权重身份验证；优先复用经过验证的长期worker或设计有变更失效证明的跨进程校验缓存。

后台连续采集可使用 `server_observability.py watch --config configs/observability/debug-v1.json --mode debug --window-id RUN-ATTEMPT --endpoint instance0=http://127.0.0.1:8200/metrics --stop-file /owned/terminal.json --out /owned/server-metrics`。默认最多145份快照，首次/定时/终态各保存完整快照，相邻窗口输出差分。单独运行collector，不占用主线程反复轮询；终态文件只在本地检查，不触发模型调用。


首次debug服务端窗口已产出实测：一请求的queue约0.0000345秒、prefill 0.7069秒、decode142.0853秒、TTFT0.7347秒。第二实例完成数0，各延迟保持未知。55项模式/传输/汇总检查通过。该单请求实测验证采集通路，不作为并发吞吐基线或normal/debug开销结论。

## 启用方式与证据口径

在新任务启动前选择配置，日志写入任务自有的本地持久目录：

```bash
export SAEA_OBSERVABILITY_MODE=debug
export SAEA_OBSERVABILITY_CONFIG="$PWD/configs/observability/debug-v1.json"
export SAEA_OBSERVABILITY_RUN_ID=RUN_ID
export SAEA_OBSERVABILITY_ATTEMPT_ID=ATTEMPT_ID
export SAEA_STAGE_TIMING_PATH=/owned/local/ATTEMPT_ID-timing.jsonl
```

正常运行改为 `normal` 和 `normal-v1.json`。日志路径必须全新且绝对；不覆盖旧attempt。每份JSONL旁生成 `.config.json`，固定有效配置、SHA、能力边界与持久化说明。计时事件使用内核写入；科学checkpoint仍独立fsync。换模式须新进程/新日志，不能修改正在运行的实验。

debug span包含阶段、父子span、单调时钟/墙钟、run/attempt及当前上下文中的sample/request身份、状态和允许的测量字段。请求层身份具备关联时才作逐请求分析；启动校验没有sample身份，应独立报告为启动成本。无原始prompt、音频内容、答案或错误文本。服务端快照仍是实例窗口，客户端事件不能补造逐请求服务端阶段。

集成验证：学习恢复、容量策略、预检观测、模式、传输和服务指标共119项通过（10.51秒）；不是normal/debug生产开销结论。待完成真实高并发窗口的开销对照与逐请求服务端/音频encoder跟踪。

## RSI结构化汇总与调度边界

`scripts/analysis/summarize_stage_timing.py` 保留原统计接口，并输出 `observability`（有效配置、校验后的SHA、mode/run/attempt）及各阶段 `measurements`（观测数、缺失数、均值/极值/总量）。旧日志缺少配置时标未知；配置被改写或身份混合时拒绝归因。使用完整样本过滤后，未绑定sample的启动成本仍保留在独立 `process_scope`，不能与未过滤汇总重复累加。缓存覆盖字节不算磁盘I/O；只有 `runtime_identity_full_hash_*` 记录实际哈希字节，失败读取的未知字节不补零。

```bash
python scripts/analysis/summarize_stage_timing.py /owned/local/ATTEMPT_ID-timing.jsonl \
  --output /owned/local/ATTEMPT_ID-summary.json
```

普通及prefix调度路径新增debug阶段：`dispatch_contact_lock_wait`（联系锁获取）、`dispatch_trajectory_slot_wait`/`dispatch_prefix_slot_wait`（容量）、`dispatch_executor_wait`（提交至worker启动，含事件循环调度）、`dispatch_blocking_execute`（worker执行）。sample身份随异步上下文进入worker；worker内部已有request计时接续，不能将整个worker区间称为纯推理。node-batch/node-pipeline内部队列、local-tool队列和准备线程池尚未获得这套完整分段，分析必须声明执行路径。该改造不改变科学trace或调度容量。

真实新debug启动已记录26文件、63,447,489,451字节完整哈希84.552秒，整体验证84.571秒；同进程后续缓存验证0.011–0.013秒。该证据对应一次有界综合启动，不代替高并发与模式开销评估。

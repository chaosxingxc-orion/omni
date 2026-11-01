# RSI 可观测工程：正常与 debug 模式

## 目标和配置边界

可观测性需要支持定位问题、比较优化前后及选择下一步实验。不能以“现有记录无法拆分”结束分析，也不能将推测写成已测阶段。可观测配置与模型、解码、数据和奖励配置分离；每次运行记录有效配置及哈希，运行中不切换。新配置不回写旧实验。

| 层面 | normal | debug |
| --- | --- | --- |
| 进度与成本 | 样本完成/总量、token、请求、资源、秒/条、ETA及依据 | 同 normal |
| 科学恢复证据 | 完整intent/response、checkpoint、来源与历史成本 | 完全相同，不因调试关闭 |
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

最终合并修正的生成进程在首次预检期间累计读取约65.45 GB，启动4分39秒时CPU占用约92%，随后进入learner与请求socket。该证据来自进程I/O计数与文件句柄，尚不是精确分阶段计时；不能将全部时间都归到模型哈希。但它确认启动路径有显著CPU/读取成本，需独立记录首次完整权重校验、进程内cache命中与失效原因。不得为提速跳过权重身份验证；优先复用经过验证的长期worker或设计有变更失效证明的跨进程校验缓存。

后台连续采集可使用 `server_observability.py watch --config configs/observability/debug-v1.json --mode debug --window-id RUN-ATTEMPT --endpoint instance0=http://127.0.0.1:8200/metrics --stop-file /owned/terminal.json --out /owned/server-metrics`。默认最多145份快照，首次/定时/终态各保存完整快照，相邻窗口输出差分。单独运行collector，不占用主线程反复轮询；终态文件只在本地检查，不触发模型调用。


首次debug服务端窗口已产出实测：一请求的queue约0.0000345秒、prefill 0.7069秒、decode142.0853秒、TTFT0.7347秒。第二实例完成数0，各延迟保持未知。55项模式/传输/汇总检查通过。该单请求实测验证采集通路，不作为并发吞吐基线或normal/debug开销结论。

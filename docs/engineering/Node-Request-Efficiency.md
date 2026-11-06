# 节点请求效率与工程优化

## 当前结论

优先完善固定工程基座，暂缓新增科学实验。原运行任务已按切换脚本自然排空并保留账本；新基座已完成无请求部署验收，**新版在线吞吐尚未验证，不能宣称效率已拉满**。

2026-11-05/06 UTC 的配置化切换使用 release 镜像 `sha256:3aa751e2864ca1206bee89e1a517c2c533a3d7ee50b42e1b78b42bc63bc1f17f`，源码提交 `cf273e52`。两个四卡实例 `saea-efficiency-a/b-cf273e52` 已运行，分别使用 0–3/4–7 卡，端口 8200/8202，重启次数为 0；两个 `/health` 和 `/v1/models` 均通过。切换阶段没有发送科学请求，不能把启动成功当作吞吐收益。

只读 runtime receipt 已生成并通过 `gate_runtime_vllm`：`docs/receipts/runtime-vllm-efficiency-dual-tp4-ctx64k-v1.json`，文件 SHA256 为 `96c2a11f…8144ac66`，协议 SHA256 为 `4da7dc0d…362ac98a`。receipt 固化了 26 个模型文件、运行时包/源码身份、两个实际 PID、端口、卡映射和 HCCL 环境；生成过程零模型请求。

本轮基座改造加入外部 `saea-runtime-efficiency-v1` 配置：在 `preprocess`
阶段并行预热冻结 roster 的原始音频，并复用既有身份绑定的有界缓存；同时
保留准备、模型接触和响应校验三段流水线。预热默认关闭，缓存条数和字节数
双上限，文件缺失或读期间变化会在首个模型请求前失败。它只改变输入准备
时机，不改变请求字节、样本顺序或科学参数。

原生 P→Slot→Decision 路径现在也支持独立的
`saea-native-efficiency-v1` 外部 overlay（示例：
`configs/runtime/native-slotfill-efficiency-v1.example.json`）。它把
`recall_prefetch`、`role_input_offload`、`ready_request`、准备/校验池和
`admitted_trajectories`/`model_inflight` 放在运行配置中；配置在工厂和实际
FlightRuntime 两处使用同一校验器，并写入有效 config hash。只有 native role
episode 才能启用，且不能和 scientific `dispatch`、`request_pipeline`、
node batch/pipeline 或 prefix-slot 调度混用。该改造只改变调度，不改变模型
可见请求、知识、名单或算法参数；真实收益仍需在后续受控 600 秒窗口测量。

最新终态证据来自 2026-11-05 20:08/20:10 UTC 的后台观测记录，本轮另核对实际容器退出状态和 A 日志。两个 TP4 服务仍监听，8 张卡的本轮瞬时 AICore 均为 0%；不是持续利用率统计。本轮没有停止或重启实验。

|同一 Fold A 上的实验臂|已记录终态/总样本|退出原因|运行期资源|累计秒/终态|后续|
|---|---:|---|---|---:|---|
|A：P off → Slot off → Decision on|919/1080|17:52 UTC，预期 AdmissionDrained，exit 1|4 卡/1 实例|42.45|161 条尚未记录终态；基座验收后核对恢复集合|
|B：P off → Slot thinking → Decision on|1080/1080|20:04 UTC 正常完成，exit 0|4 卡/1 实例|41.60|独立核验、评分，无须重复生成|

终态包含协议拒绝，不能当作正确答案或完整有效轨迹。上述速度包含启动、排队和排空，不是单样本推理延迟。A 已暂停，ETA 暂不估计；B 生成已结束。观测器报告 A/B 生成调用为 2231/2526、在途或未知为 0；FIT custody 数分别相同，但不等于已完成跨账本恢复核验。历史费用全部保留。

## 已确认的效率瓶颈

旧版本 09:00:26–09:07:48 UTC 引擎窗口：

|指标|A|B|
|---|---:|---:|
|平均服务请求耗时|14.07 秒|39.62 秒|
|平均 prefill|1.03 秒|1.20 秒|
|平均 decode|12.98 秒|38.34 秒|
|prefix token 命中率|40.57%|36.93%|

另一个 10 分钟日志窗口中，两实例平均 running 仅 0.80/2.54，最大 3/6，waiting 最大为 0。引擎上限 128 未被应用持续供满。模型 P、Slot、Decision 在 NPU 推理；数据读取、知识检索、上下文构造、身份检查、响应校验和评分主要在 CPU。

应用一次 render 均值约 38.81/37.25 秒，session 检查约 14.81/13.10 秒；响应返回后到外层请求结束仍有明显等待。重复知识快照校验、预算锁内慢身份检查和 CPU 调度是已识别问题。父子计时重叠，不能把这些均值直接相加。

音频读取均值仅约 0.35/0.28 秒，base64 约 0.0002 秒，因此音频缓存属于辅助优化，不是已证实的主要瓶颈。

## 固定基座已完成的改进

- **输入复用**：同一 flight 复用 adapter、split roster、raw packet binding；媒体校验缓存按身份失效。知识投影采用私有不可变上下文，避免逐条重复校验整份知识。
- **并行流水线**：native 请求准备和 FIT 先于模型发送许可；准备、模型在途、响应校验分别限流。响应计数独立锁，慢 session 检查移出预算临界区。
- **配置驱动**：固定镜像或版本固定的只读挂载代码，外部配置声明并发、线程数和 normal/debug；不为每个实验生成新业务实现。
- **音频缓存**：`SAEA_AUDIO_CACHE_ENABLED=1` 显式启用，默认关闭；条目数和编码字节双限，绑定路径/device/inode/size/mtime/ctime，读取变化拒绝，并发 miss 不重复计费。新的外部 overlay 可在正式推理前并行预热同一缓存；仅用于冻结输入，不替代内容身份门禁。
- **可观测性**：debug 区分 capacity_wait、executor_wait、execute；normal 保留关键统计。常规观测每 600 秒或终态事件触发。
- **有界校验背压**：prepared pipeline 对已返回但尚未校验的响应设置与
  `queue_depth` 相同的上限；达到上限时仅等待最早完成的校验，避免 CPU
  校验落后时无界保留响应和轨迹。dispatch receipt 额外记录
  `model_queue_wait_seconds`、`validation_backpressure_events`、
  `validation_backpressure_seconds` 和 `max_validation_pending`，用于区分
  NPU 槽位不足与 CPU 校验不足。
- **校验失败快速收敛**：每轮完成事件中先消费已完成的校验 future；发现
  响应不合格后取消尚未开始的准备和模型 future，保留已实际发送的响应与
  trace，避免错误结果继续消耗 NPU。

单请求 direct 使用 `configs/runtime/request-pipeline.example.json`；native 多节点使用 `configs/runtime/native-slotfill-efficiency-v1.example.json` 的 `dispatch` 配置，包括 `prepare_workers`、`validation_workers`、`admitted_trajectories`、`model_inflight`、`ready_request`。两者不可混用。`model_inflight` 是 contact 许可，包含最终检查和响应处理，并非纯 HTTP 在途。

原生节点的效率开关可通过同一外部 overlay 提供，避免为每次调度探测修改
科学 baseline；准入前仍严格校验角色图、传输引擎和并发窗口。

## 验证范围

|验证|结果与边界|
|---|---|
|固定历史 P/Slot/Decision 请求投影|body/receipt 相同；无 profiler 的 Slot 5.5630→1.0429 秒，Decision 5.9386→1.1750 秒，P 基本不变；每条件单轮，不推算整体倍数|
|native 请求、持久 FIT 恢复及 fallback|既有固定镜像 91 项通过，另有 2 项取消边界测试|
|runtime/factory/debug 组合|既有固定镜像 219 项通过；发生于最新音频缓存硬化之前|
|音频缓存硬化|固定基座候选 27 项通过；最终 release 构建保留该实现，真实请求收益待测|
|prepared pipeline 背压|固定镜像受影响回归通过；本机带源码路径 56 项通过，2 项为既有 `openjiuwen`/环境变量边界，不归因本轮改动|
|跨版本恢复与外置准入|既有隔离三节点恢复正例、挂载和故障恢复测试通过；真实终态结算与正式接续仍待|

最新候选已从 `4997a43b` 构建：镜像 ID
`sha256:879127561b4426b6a76730ed48ed4266f5e3e9c3776b3887b4bce09b201b2bf0`，
源码清单 SHA256 `58eccc2fb597e55a9affc39dae22b365ac8c50dd2d7b1cdd2d1a6834b256be71`。
固定 parent、无网络 detached 构建退出 0；清单 7497 个文件验证通过。
最终镜像内挂载只读测试文件、使用镜像内业务源码执行受影响回归，
**111 passed，5 条依赖弃用 warning，26.48 秒**。测试包括外部配置真实
native 调度接线、请求一致性、取消、恢复及有界背压；没有真实模型请求。
临时测试仓用例仅在测试进程清除 `SAEA_RELEASE_MANIFEST`，生产门禁保持启用。

CPU harness 的 venv 导入通过；系统 Python 的 ACL/NPU/vLLM 模块可发现。
本轮未执行新镜像的设备挂载推理或动态库加载 smoke，因此不宣称新镜像已完成
真实 NPU 推理验收。旧双 TP4 服务仍使用 `cf273e52` 镜像；本轮没有停止或
替换服务，也没有扩大科学实验。候选完成 CPU 验收，在线吞吐仍待验证。

## 剩余工作与验收顺序

1. 候选镜像已构建并完成 CPU 验收。先核对真实 workload 的 native overlay 与外置准入配置、设备挂载 smoke、resident/session 身份和恢复证据，再决定首实例交接；已有 cf273e52 runtime receipt 不可冒用为新镜像的证明。
2. 从 A 真实终态采集 checkpoint、生成/FIT 账本和共享 guard 证据；用原生 reader 确认可保留集合，再生成后继配置。919 条终态不自动等于 919 条可保留；未知请求禁止盲重发。
3. 首实例用已核验可保留的原 Fold A 后继数据做 600 秒真实效率窗口，同时记录完成数、有效完整轨迹、秒/条、各阶段等待、实际引擎在途、CPU 队列和 NPU 利用率；未知请求不得重发。验证合格后再扩大实验。
4. 根据实际瓶颈调整 32/64/96/128 并发及准备/校验池，直至增加并发不再增加有效吞吐或出现资源压力。不同数据窗口不能声称严格倍数收益；不把逐档重复整份实验设为必需。
5. 只有供给充足后，才考虑 batch token 2048→4096/8192。prefix 已开启；修改 prompt 顺序须另做精度回归。全量科学实验和 TFRL 保持暂缓。

当前无新增用户决策项。GitHub main 与 Wiki 本轮推送均超时，最新本地提交尚未确认远端同步。

验收结果文件：[Native-Runtime-Efficiency-Validation.json](../results/Native-Runtime-Efficiency-Validation.json)。示例并发 64、准入轨迹 128 是配置起点，并非已测最优值；调档以真实有效吞吐和稳定性为准。

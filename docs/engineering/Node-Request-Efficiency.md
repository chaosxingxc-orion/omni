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

## 当前优先级与断点核验

按最新指令，新增科学实验暂缓，先完成工程基座效率验收；仍在运行的实验不主动停止，基座通过后才排空、切换和断点接续。本轮没有停止或重启服务，没有发送新模型请求。2026-11-06 01:08 UTC 核对，两实例 running/waiting 均为 0；服务驻留不等于实验在运行或 NPU 满载。

最新候选镜像中的只读 CPU 审计已退出 0：

- A 的 919/1080 条终态分类为完整 601、有限完成 19、拒绝 299；分类不是准确率，也不是最终可恢复数量。
- checkpoint SHA256 为 `91e3fe35…4ce19158`，哈希链通过、截断尾部 0 字节。
- 2231 条生成请求都有已记录响应，生成账本与 attempt 成本一致；累计音频 6160.057875 秒是模型接触计费量，不是实验墙钟耗时。
- 随后完整原生 terminal custody 容器退出 0：919 条均通过可保留行核验，incomplete 为 0；2231 条生成调用与2231 条 FIT endpoint 记录逐项匹配，缺失 custody 为 0。拒绝终态仍保留，不回收其费用。后继编译、请求等价及现场 guard/服务检查尚未完成，不能直接启动剩余 161 条。

工程验收按依赖推进：**输入预热与有界缓存 → 准备/模型/校验并行供给 → 原断点恢复核验 → 真实吞吐窗口**。输入、并发供给和校验优化的代码及相关回归已经具备；新恢复配置已绑定候选镜像，终态包已生成；后继编译已退出0（1080条原名单、新模型/FIT调用均0），当前服务身份仍需接通。候选版本固定，不因文档更新重复构建。

效率验收使用原始冻结名单和已核验断点，科学请求保持一致。每个窗口记录有效完成样本/秒、各阶段排队/执行耗时、模型实际在途、校验积压、内存和逐卡利用率；并发上探以边际吞吐收益为准。“效率拉满”需实测，无请求测试只能证明功能边界。新增全量实验与 TFRL 在验收后恢复；无需用户追加决策。

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

最新候选从 `4ad95774` 构建：镜像 ID
`sha256:2e8d3e239d9a052b7b40a58c97c5b30a81795bb8782ee8ef8b798e62930be126`，
源码清单 SHA256 `bbb13509b18e5afdaa540c643eab6cb53bcf2e6a462ffbe946f41f778399403d`。
固定 parent、无网络 detached 构建退出 0；清单 7497 个文件验证通过。固定镜像内
受影响回归 **111 passed，5 warnings，25.73 秒**；与在线服务相同的 Ascend 设备挂载
下 `acl/torch_npu/vllm/vllm_ascend` 导入 smoke 通过，未发送模型请求。
测试包括外部配置真实 native 调度接线、请求一致性、取消、恢复及有界背压；
没有真实模型请求。临时测试仓用例仅在测试进程清除 `SAEA_RELEASE_MANIFEST`，
生产门禁保持启用。

CPU harness 的 venv 导入和设备挂载动态库 smoke 均通过，但这仍不等于真实 NPU
推理吞吐验收。旧双 TP4 服务仍使用 `cf273e52` 镜像；本轮没有停止或替换服务，
也没有扩大科学实验。新候选已完成 CPU/设备层验收，在线吞吐和断点接续窗口仍待验证。

## 固定依赖镜像与只读业务基座

新增 `mounted-release-v1` 启动模式，把依赖镜像与业务源码分别固定。完整源码目录
只读挂载到 `/opt/saea-release`，使用现有 source manifest 校验完整文件集和内容；
启动器在执行 verifier 前先核对其字节哈希。禁止可写源码、局部子目录覆盖和依赖
目录覆盖。`image-release-v1` 保持原有约束，不放宽。

`configured_job_launch.py` 的 `source_identity.mode` 选择此模式；`image` 固定
依赖镜像 ID，`manifest_sha256` 和 `expected_commit` 固定业务源码。用 `user`
明确运行 UID/GID，输出目录预先按同一属主创建。不得为了读取旧私有账本修改其权限。

引擎使用 `scripts/npu/npu_deploy.sh` 的 `SAEA_DEPLOY_MODE=mounted-release`，另传
`SAEA_RELEASE_SOURCE_DIR`、`SAEA_RELEASE_IMAGE_ID`、
`SAEA_RELEASE_SOURCE_MANIFEST_SHA256` 和 `SAEA_DEPLOY_MODEL_FRAGMENT`。
源码仅挂载到配置解析 helper；服务容器沿用固定基础镜像的工作目录和导入路径，
业务代码升级无需改写引擎环境。`plan` 已在实际 TP4/64K 配置执行，通过源码验证并
正确报告旧服务占卡；没有绕过占卡检查或启动第二个占用相同卡的实例。

181验证结果：固定依赖镜像内启动器 **46 passed / 0.29s**；只读源码与旧引擎运行时
文件核验通过；原生后继配置和 **1080 条真实媒体哈希验证通过 / 7.22s**。三项均
零模型请求。该路径尚未完成真实服务切换及在线吞吐验收，不能据此报告效率提升。

## 剩余工作与验收顺序

1. 候选镜像已构建并完成 CPU 验收。先核对真实 workload 的 native overlay 与外置准入配置、设备挂载 smoke、resident/session 身份和恢复证据，再决定首实例交接；已有 cf273e52 runtime receipt 不可冒用为新镜像的证明。
2. A 的完整终态核验已通过：919 条可保留，生成/FIT 账本各2231条匹配。继续用固定脚本编译后继配置，保留旧成本；启动前再核验现场 guard/部署，不重发未知请求。
3. 首实例用已核验可保留的原 Fold A 后继数据做 600 秒真实效率窗口，同时记录完成数、有效完整轨迹、秒/条、各阶段等待、实际引擎在途、CPU 队列和 NPU 利用率；未知请求不得重发。验证合格后再扩大实验。
4. 根据实际瓶颈调整 32/64/96/128 并发及准备/校验池，直至增加并发不再增加有效吞吐或出现资源压力。不同数据窗口不能声称严格倍数收益；不把逐档重复整份实验设为必需。
5. 只有供给充足后，才考虑 batch token 2048→4096/8192。prefix 已开启；修改 prompt 顺序须另做精度回归。全量科学实验和 TFRL 保持暂缓。

恢复配置审查复用28项已验证源码差异，对新增或变化的7项补审。挂载方案已分离本地热状态和只读科学输入；终态核验和后继编译成功不代表后继服务绑定已成功。旧/当前引擎的模型、解释器、包、源码树与运行库receipt内容相同，但工作目录、环境及导入路径文件变化，导致协议身份不同；需要配置化等价启动或明确的兼容证明，不能直接复用旧session。

当前无新增用户决策项。GitHub SSH 22 和 443 连接均超时，最新本地提交尚未确认远端同步。

验收结果文件：[Native-Runtime-Efficiency-Validation.json](../results/Native-Runtime-Efficiency-Validation.json)。示例并发 64、准入轨迹 128 是配置起点，并非已测最优值；调档以真实有效吞吐和稳定性为准。

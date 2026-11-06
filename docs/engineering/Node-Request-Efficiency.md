# 节点请求效率与工程优化

## 当前结论

按最新指令，先完善固定工程基座，暂缓新增科学实验。正在运行的实验不主动停止；候选通过功能验收后，才排空、保留断点、切换，并用真实输入验证效率。

截至 2026-11-06 03:24 UTC，A/B 两个四卡推理服务保持原状。A 的候选业务源码为完整只读 `4fa68e30`，依赖镜像固定 `7e60208f`。**接续worker在03:19:13 UTC自行退出，exit 1；原因是框架默认向只读源码目录写日志。没有新增模型调用，也没有新增终态样本。** 新科学实验继续暂缓，不能报告满载或新版吞吐。

本次真实preflight通过：历史919条保留，剩161条。输入预热实际加载1080/1080条音频、失败0，缓存77,788,012字节，耗时1.123秒、8线程。它证明缓存接线有效，不代表端到端效率验收完成。

|任务|已记录终态/总量|历史生成调用|原运行资源|历史累计秒/终态|当前安排|
|---|---:|---:|---|---:|---|
|A：P off → Slot off → Decision on|919/1080|2231|4卡/1实例|42.45|worker已失败；保留161条待后继验收|
|B：P off → Slot thinking → Decision on|1080/1080|2526|4卡/1实例|41.60|生成已完成，无须重复生成|

终态包含拒绝，不等于正确答案或完整有效轨迹。速度包含启动、排队和排空，不是单样本延迟。A 恢复前 ETA 未知；B 无剩余生成时间。A 完整 custody 核验确认919条可保留，incomplete 0；2231条生成调用与FIT逐项匹配，缺失0。完整601、有限完成19、拒绝299；保留所有历史成本。

## 本轮定位的配置问题与修复

真实准入在任何模型请求、workspace发布和新attempt之前拒绝：候选配置身份不一致。已用相同固定源码、相同输入做零请求对照：

- 编译/原预检没有数据根环境变量，配置哈希为 `07994ff8…e9be`。
- 准入有 `SPEECHRL_DATA_DIR`，同一配置哈希变为 `ba7b8521…b300`。
- 仅去掉此环境变量，预检哈希恢复为 `07994ff8…e9be`。

原因是配置哈希会根据环境变量把绝对数据路径归一化。编译与运行环境不一致，导致编译出的派生身份无法准入；这是工程问题。正式启动器要求显式数据根，不能靠删变量修复。

通用编译器已改为在配置声明的数据根下计算身份，退出时恢复调用方环境；环境预检新增可选 `expected_config_hash`，身份不匹配时在读取全量媒体前拒绝并记录预期/实际值。固定依赖镜像内针对性回归 **34 passed、5 subtests passed / 3.61秒**。生产候选源码保持不变，先用原封存编译器加一致的外部数据根配置重新生成派生资产，旧资产和账本完整保留。

准入第二次已通过配置身份检查，但发现发布源码不含运行账本、实验资产又漏打包原账本。编译器已新增提前缺项拒绝；原账本作为独立、带哈希的只读实验输入保留，未改写任何历史行。修复属于输入闭包完整性，不改变模型或算法。

知识源文件由其他UID持有且不可被受限worker直接读取。已复用 `configured_asset_stage.py` 将约45MiB、23个文件按逐文件哈希复制到本地私有输入区，复制后复核哈希，零模型调用；原文件权限不改。新的准入使用只读本地知识挂载。失败的部分workspace保留，后继使用新workspace，未自动重放实验。

## 最新验收结果

2026-11-06 02:20 UTC，恢复准入v6 **exit 0**：已生成真实completion、决定记录和workspace清单，零模型调用。全部12份来源凭证逐项核对，遗漏的仓库凭证和原实验说明补入带哈希的外部资产，不修改原证据。生成worker的layout/plan也通过，共享guard为clear；尚未launch。

候选源码 `de1429f5` 已在181以完整只读目录发布，7498文件清单验证通过；manifest为 `7d254509…c463ae`。仍使用固定parent镜像，不修改引擎。候选1080条真实媒体哈希与预期配置身份预检通过，4.32秒、零模型调用。它只证明输入/环境可用，不是生成吞吐或实际预热运行结果。

|验证范围|固定镜像结果|
|---|---|
|恢复编译器输入闭包、环境预检|36 passed、5 subtests / 3.70秒|
|worker内预热、缓存与真实transport读取|42 passed / 5.34秒|
|观测器、挂载源码启动器|61 passed / 4.66秒|

预热新增显式环境配置 `SAEA_AUDIO_PRELOAD_WORKERS`（1–64），配合现有缓存开关和容量，在worker自身preprocess阶段填充后续transport使用的缓存。运行中campaign沿用最多8个预处理线程。默认不启用；与runtime audio_cache配置冲突或缓存关闭时拒绝，媒体缺失在推理前失败。测试确认预热后transport不重读媒体、配置哈希不变。示例：`configs/runtime/audio-preload-environment.example.json`。

## 实际接线与验收缺口

新候选 `de1429f5` 的独立编译与准入均 **exit 0**，已生成新completion和workspace清单 `d3acaa33…98c0`，没有复用旧候选准入。其跨版本checkpoint桥接预检也 **exit 0**：核验919行checkpoint、35项源码审查、21项数据pin和10项reference，全部零模型/FIT请求。

这些结果证明恢复输入和源码兼容预检通过，**不等于效率验收通过**。新候选离线重渲染历史P、Slot、Decision各一条真实请求，三个请求体与投影凭证哈希均与旧值完全一致，零模型/FIT请求；覆盖代表请求的投影层，不覆盖所有动态请求和HTTP传输层。

新worker的layout/plan通过，运行guard空闲且原inode保留，8个新输出路径无冲突。02:53 UTC通过通用启动器启动 `generation-preload-v1`，接续原名单剩余161条作为工程效率验收，不扩展科学实验。运行随后在runtime preflight退出：缺少源码路径下的 `docs/receipts/e0-closure.json`。原凭证实际存在于只读历史执行树，但门禁仍硬编码源码目录，未通过外部资产解析器读取。这是基座配置接线bug，不是模型错误。当前生成/FIT attempt账本和请求journal均未创建，零新模型/FIT调用；部分run目录和失败输出保留，不能原地重提。

修复 `97b12bfd` 让E0、runtime和D1补充凭证统一通过已有哈希绑定的 `resolve_asset` 读取，源码/评分冻结核验继续保留。固定parent测试 **45 passed / 2.90秒**，6条pytest marker注册警告（测试挂载未带项目pytest配置）；测试包括凭证篡改和评分源码篡改必须拒绝。修复已包含在新源码 `4fa68e30` 中，独立完整只读发布（7498文件，manifest `f6f3aa6e…40ed1f`），未改写旧源码。

通用编译器新增E0四项artifact/runtime/D1闭包检查，缺项或内容变化在编译前拒绝；发布验证脚本增加 `gate` 模式，使用正式配置/计划调用生产 `FrozenCoreGate`。固定镜像回归 **37 passed / 4.46秒**。新候选编译及独立workspace准入均exit0；completion清单 `a3692537…45463` 已绑定到新worker。

真实 `gate` 模式 **exit0**：26个模型文件全哈希，共63,447,489,451字节，vLLM运行依赖、E0及评分冻结校验通过；零模型/FIT/教师请求。它验证模型调用门禁，不等于完整flight preflight或实际吞吐验收。

新准入配置的恢复bridge也exit0（919行、35源码审查、21数据pin、10引用）；三个节点的代表请求重渲染再次全部匹配历史body/receipt哈希。新worker layout/plan通过、guard clear、独立停止控制目录，保留旧失败输出。03:14:52 UTC启动工程效率验收；真实preflight/preload通过，随后框架初始化失败。独立新进度observer与逐卡observer均active、间隔600秒；启动前两端点metrics已留存，待按新窗口求差值。
恢复候选内置dispatch：准入32、模型许可32、准备线程4、校验线程4；它是恢复起点，不是最优并发。缓存4096条/4GiB，预热候选建议8线程。不叠加第二套调度配置。normal/debug与服务指标600秒采集保持配置驱动。

观测器修复 `c93b696d` 增加checkpoint恢复识别：核验前驱文件哈希、run/attempt/split身份与当前preflight凭证，拒绝重复前驱、路径越界和计数冲突。固定依赖镜像回归 **24 passed / 4.78秒**。该外部工具修复不修改已准入业务源码。

进一步核查确认恢复attempt的checkpoint只含新跑样本。观测修复 `25568b15` 使用历史/新增样本集合并集汇报总进度，吞吐仅按新增终态样本计算；拒绝重叠、越界或计数冲突。固定镜像 **28 passed / 4.69秒**。前置凭证尚未发布或无新增终态时，速率/ETA留空。速度包含当前attempt启动/预处理，不等于纯推理延迟。

独立systemd观测任务已启动，600秒轮询、SSH断开继续运行、无自动重提。逐节点计时、真实HTTP在途、各卡利用率及稳定并发上限仍待运行验收，不能宣称效率已拉满。

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
推理吞吐验收。A 已在无在途请求、共享 guard 保护下切回固定依赖镜像；B 保持原服务。
当前业务源码已推进至 `4fa68e30`；上述 `4ad95774` 是历史构建记录。在线吞吐窗口仍待验证。

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
零模型请求。A 服务切换及实际进程注册现已通过；在线吞吐尚未验收，不能据此报告效率提升。

## 框架启动故障与修复

真实worker进入工作流时，openJiuwen默认把日志写到当前工作目录下；当前目录是只读源码树，因此初始化抛出只读文件系统错误。这是部署接线bug，不能归因模型或推理引擎。失败attempt明确ABORTED、实际calls/audio_seconds均0；未出现新contact journal，保留其全部状态与共享lease metadata。

通用启动器新增 `framework_runtime.log_directory`，通过框架公共配置API将日志写入外部本地可写目录，并在执行科学入口前导入真实工作流。源码和依赖镜像不改，禁止局部可写覆盖。新增 `runtime-check` 复用同一初始化路径，隔离网络、只做框架初始化，不执行实验入口。

- 固定依赖镜像回归：**51 passed / 0.33秒**，包括外置目录约束、禁止源码覆盖、检查不执行业务入口和不自动重提。
- 真实只读 `4fa68e30` 源码＋固定依赖镜像：检查 **exit 0**，实际工作流导入和外置日志写入通过，零模型请求。尚未用该修复启动后继生成worker。
- 启动计时：准备资产校验70.178秒；26个模型文件首次全哈希合计58.953秒。两者属于启动成本，须与稳态请求延迟分开；嵌套区间不相加。已存在进程内memo，不能误称每个请求都全量重校验。

用法见 [配置化启动说明](../../studies/speech-aware-evidence-acquisition/configs/launcher/progress-and-handoff.md)。

## 下一阶段与验收标准

1. 固化通用启动器与外置配置，保留已失败attempt、旧源码、历史919条响应及全部成本；审计并结算失败lease，unknown不盲重发，guard inode不替换。
2. 在不修改已准入源码的前提下，创建独立后继配置和attempt；完整输入、框架初始化、恢复兼容及请求一致性检查通过后，才接续原161条作工程验收，不扩展科学实验。
3. 测量新增终态、有效完整轨迹、墙钟秒/条、各节点等待/执行、真实HTTP在途、CPU队列、内存和逐卡利用率。启动开销、稳态吞吐和单条延迟分别报告；后台600秒或终态触发。
4. 供给稳定后配置化上探32/64/96/128并发与准备/校验线程；按有效吞吐边际收益及资源压力停止上探。不同样本窗口不宣称严格倍数收益。
5. 应用供给充分后再评估batch token 2048→4096/8192。prefix缓存已有开启证据；保持科学输入不变。当前32并发只作起点，尚未证明最优。
6. 工程验收通过后锁定镜像、源码和运行配置，再恢复全量科学实验与TFRL；有在运行的实验时，先安全排空、保留断点再切换。当前无需新增用户决策。

GitHub同步状态以实际push返回为准；SSH连接超时，不能把本地提交称为远端更新。结构化证据见 [Native-Runtime-Efficiency-Validation.json](../results/Native-Runtime-Efficiency-Validation.json)。

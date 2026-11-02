# 实验总账

## 2026-11-02：TFRL-v2 admission audit

181 固定 `saea-base:20261030-naturalend-v1` 镜像已执行公开的 CPU-only admission audit（main `3899eda32`）。真实 readiness 为 `ready=false`，`model_calls=0`、`test_labels_read=false`。阻断项包括：root admission 仍为 `DRAFT-NOT-MINTED`、`PORT_DEPLOYMENT_PROFILE` 缺失、bank release 仍含 pending/zero sentinel，以及 9 个 construction/Fold-B smoke/iteration FIT 与 generation plan 仍为 `DRAFT-HELD-REGISTERED_BEFORE_CONTACT`。该结果不启动模型请求；TFRL smoke 继续 held。

## 2026-11-02：三队列 live runner 接入桥

公开 `main` 提交 `3fdb1f0c0` 新增默认关闭的 `QueuePipelineBridge`，后续 `8aa21a55e`、`0d0992f81` 修正并验证跨轮次单调 stage 身份。它把现有 `NodePipelineCoordinator` 的 prepare、model contact 和 terminal callback 映射到 durable 三队列 checkpoint，并强制同一 member 的阶段依赖只能在前一阶段 observed 后解锁；未知、重复或乱序事件沿用 fail-closed 语义。未注入 bridge 时旧 runner 路径保持不变，因此没有改变既有实验配置或请求顺序。新增 bridge 单测和纯 Python smoke 通过；尚未在 live NPU runner 启用，也没有新增模型调用。

## 2026-11-02：CPU-only evidence-card 消费端

公开 `main` 提交 `f3d9042f5` 新增隔离的 TFRL evidence-card consumer。它只接受与原始数据实际 intent-slot relation 完全匹配、带来源和证据的卡片，并按 source/intent/slot/relation 稳定排序；无匹配、证据缺失、答案缺失或同键冲突时 fail-closed 回退 lock。组件不构造意图×槽位笛卡尔积、不读取 test 标签、不调用模型，也未接入 live runner 或登记 NPU 预算。纯 Python smoke 与语法检查通过，开发机未安装 pytest，未宣称完整 pytest 回归。

更新时间：2026-11-01。当前状态原位维护；历史失败、成本与变更保留在结果文档和Git历史。

| 实验 | 完成量 | 结果 | 资源与效率 |
| --- | --- | --- | --- |
| Direct lock / aa / ab | 各1080/1080 | SLU-F1 45.1178% / 45.6633% / 45.2940% | 官方完整评分 |
| A0 Fold B | 1080/1080 | SLU-F1 73.2225%，首轮收益锁定 | 多attempt合并，保留全部成本 |
| A0 Fold A | 1080/1080 | SLU-F1 73.3139% | 8卡/2实例，2.568秒/条 |
| REAL500 TP4-128 | 500/500 | SLU-F1 72.6877%，工程基线保留 | 8卡/2实例，2.766秒/条 |
| TF-GRPO 63经验 H2 | 312/312 | learned−lock SLU −0.4813pp，未晋升 | 936请求，8卡/2实例，5.277秒/条 |
| 短ID三臂 H2 | 312/312 | 短ID−lock +0.6731pp，95%区间跨0，未晋升 | 1248请求，8卡/2实例，7.1675秒/条 |
| 全错组部分奖励学习 | 51/51组中间阶段完成 | 最终合并及首次修正均超库容量，未封存 | 累计308调用，6,026,841输入/757,815输出tokens |
| 有界综合策略 | FINALIZED，10条/3442字符 | 容量修复成功，无精度结论 | 新1调用，累计309调用，历史成本保留 |
| 有界综合策略 H2 | 312/312，936/936响应 | learned 66.7400%，lock 66.0671%，SLU +0.6729pp；Intent −1.6026pp，未晋升 | 8卡/2实例，5.238秒/条；成本与归档已核验 |

H2 终态证据明确 `continuous_full_utilization_verified=false`；资源分配不等于持续 8 卡满载。最终 13,078 条 test 仍未启动。
| 最终三系统 test | 0/13078 | 方法冻结后完整评估 | 未运行 |

当前已完成的H2实验剩余0、ETA0；下一轮实验须等待工程修复审查。结构化经验规则校验已合入公开main（fa0a14dd），固定镜像回归79项通过；VerifiedImport 同进程内容复用已合入公开main（36ae9f33，基础实现3c00f3e8），181固定镜像相关回归59项通过，并在每次缓存命中重新校验SHA，文件替换/删除仍fail-closed。该修复尚未启动新的NPU实验，8卡持续有效满载仍未被证明。8卡分配不等于持续有效满载；单次最终合并是依赖瓶颈，不声称两实例满负载。CPU验证/分析与NPU任务异步；600秒或终态事件监测。

来源条件化候选已完成CPU-only构建并公开（main `9994e218`）：固定H1的51组/102个原操作，保留5条与来源语义直接一致、含scope、trigger、action、exceptions、来源SHA和冲突状态的规则；结构化校验、TF-GRPO与队列原型回归47项通过，新增模型调用和评分均为0，未读取H2/test，也未做意图×槽位笛卡尔积。生成器自动拒绝H1审计标记的F1/F2/F3/F4/F5/F8高风险来源，并强制意图、槽位、组件及意图—槽位关系均来自固定H1 roster。候选仍标记为`eligible=false`，需完成固定消费端反例审查和准入演练后才能登记NPU预算。
固定消费端 dry-run 已通过并公开（main `36e3c0a1`）：5条规则、2047字符投影、5个来源payload全部按SHA复核；运行配置和Solver prompt均未改变，新增模型/评分调用为0，H2/test未读取。该结果只证明解析和来源绑定兼容，不代表精度收益，候选仍未准入NPU。
dry-run 的 fail-closed 单测已合入 main `a31c9831`，181固定镜像综合回归49项通过；若候选携带 H2/test 读取证据、来源字节变化或固定关系不一致，CPU阶段直接拒绝。

三队列解耦仍未准入：现有 node-pipeline 只有 CPU prepare worker 与模型 contact barrier。一次 ready FIFO 改造在181固定镜像原生流水线测试中约3分钟未终态，已撤销且未影响运行树；阻塞记录已发布（main `00b7c1c2`）。后续必须先设计显式状态机、取消排空、prefix/实例亲和和 checkpoint 恢复，再重新验证，不能把简单 pending 扫描替换称为三队列完成。
隔离的三队列状态机原型已完成并公开（main `b3c774ad`）：明确 prepare/ready/generation、P→S 依赖、FIFO、实例/prefix 亲和、draining 取消语义和 checkpoint/restore；181 固定镜像纯 CPU 测试3项通过。原型尚未接入 live runner，仍需 fake callback 和小批量 smoke 验证后才能替换现有 barrier。
三队列接入前的 fake-callback 合同测试已补齐并公开（main `adbf7263`）：覆盖 FIFO、P→S 依赖、实例/prefix 亲和、取消后的 draining、checkpoint/restore 及重复 settle 拒绝；181 固定镜像测试5项通过。该测试仍是隔离 CPU 验证，不代表 live runner 或 NPU 吞吐已经改变。
随后修复了原型的依赖完整性（main `cd6fcdb0`）：每个 S job 现在必须绑定已提交的更早阶段 P job，P 只有进入 `observed` 后才允许 S 完成 prepare；checkpoint/restore 同时校验依赖存在与阶段顺序。固定镜像队列测试仍为5项通过，live runner 尚未替换。

监控已补充逐加速卡利用率证据并公开（main `8807d8e8`）：每轮只执行一次 `nvidia-smi`/`npu-smi`，保留设备 ID、利用率、显存和功耗，再从同一轮数据派生兼容的聚合字段；设备读取失败会明确记录为空，不把“已分配”写成“满载”。固定 181 镜像相关回归 46 项通过。真实 8 卡持续有效满载仍未有运行证据，下一次受控 smoke 必须报告逐卡样本、有效占用时间和空闲尾部。

运行态核验（2026-11-01 17:23 UTC）：181 上两个 TP4 vLLM 服务仍驻留，最近逐卡 AICore 利用率约 82.9%–93.1%，HBM 约 61.4/65.5 GiB；15 秒窗口内两个实例的请求计数和生成 token 均未增长，因此这段时间只能记为服务驻留，不能记为推理满载。历史指标累计约 2,177/2,231 个成功请求，不能据此推断当前有新实验。

`tfgrpo-corrected-learning-v1` 的权威失败状态已核验：146/146 个已写入 contact 均有 response，observer 记录 146/187 contacts、3/32 groups，最终 `candidate_present=false`，失败原因为 `LearningError: one nonempty semantic advantage block required`，并明确 `automatic_retry=false`、保留 receipts/pending intents/FIT/generation costs。当前没有学习进程；在拿到原始优势 response 并完成 CPU-only 解析复现前，不恢复或重发该任务。

当前尚未实施的效率优化按优先级收敛为：先完成真实 runner 的 prepare/ready/generation 解耦与多任务资源归属；再做同进程已验证对象和数据准备的长期 worker 复用；随后用实测生成长度决定分阶段预算；最后评估音频特征缓存与 prefix 亲和调度。全量音频预加载暂不是最高收益项：现有客户端读取/base64约0.018秒/条，必须先证明 encoder/H2D 重复搬运才值得投入。以上项目均保持候选 held，未登记 NPU 预算。

## 可观测系统

normal/debug使用独立配置；normal保留关键耗时、进度和完整恢复证据，debug记录音频读取/编码、JSON、HTTP分段、首次完整权重校验和缓存命中/失效原因，服务端增加直方图bucket。119项集成测试通过。新增调度/汇总联合45项通过：普通及prefix路径拆分锁/容量/worker调度等待与执行，RSI汇总校验配置哈希并保留进程启动成本；node-pipeline/batch内部和local工具池仍未完整细分。新debug执行树154项通过，准备/注册/准入完成；新合并已FINALIZED：10条/3442字符，新1请求/1响应；累计309调用、6,032,239输入/761,689输出tokens。首次真实哈希26文件/63.45GB耗时84.552秒，同进程后续缓存校验0.011–0.013秒。后继完整H2通过145项隔离测试，来源核验、预算恢复、注册准入均完成；312条/936请求已终态，完整精度、成本和归档均已核验。

已有单请求服务端窗口：queue 0.0000345秒、prefill 0.7069秒、decode 142.0853秒、TTFT 0.7347秒。它只验证采集通路，不代表高并发瓶颈；客户端HTTP响应头等待不是TTFT，模式生产开销与逐请求server trace尚未完成验收。

[完整现状与计划](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/Research-Status-and-Plan.md) · [可观测模式](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/decisions/2026-11-01-observability-modes-zh.md) · [详细实验结果](Research-Status)

## 2026-11-02：TFRL-v2 准入复核

本次复核确认 TFRL-v2 bank readiness 仍为 `ready=false`。H1、H2 和 union 三份 release 均缺少真实 bank 文本、非零内容哈希、`vllm-tokenize` token 计量、token count 和 `bank-receipt.json`。181 的 scratch、run1、run2 只有注册计划、roster 与配置，没有可直接复用的真实 bank distillation 输入或 rollout/score 封存产物。

current-seed H1/H2 结果不能替代 bank 构建输入：H1 有 supply-variant custody failure，H2 虽有完整推理结果但不是对应的 bank 构建产物。未伪造 hash、token 数或 receipt，未启动 TFRL-v2 NPU 实验。下一步必须取得真实构建轨迹和评分产物，在固定 vLLM 服务上逐项测量 token 数，原子生成 receipt 后重新运行 readiness gate。

[有界综合完整结果与耗时](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/2026-11-01-dense-synthesis-result.json)。


[有界综合策略 H2 完整结果](https://github.com/chaosxingxc-orion/omni/blob/main/docs/results/2026-11-01-dense-h2-full-result-zh.md) · [终态成本](https://github.com/chaosxingxc-orion/omni/blob/main/docs/results/2026-11-01-dense-h2-terminal-cost-timing.json) · [终态归档核验](https://github.com/chaosxingxc-orion/omni/blob/main/docs/results/2026-11-01-dense-h2-terminal-archive-verification.json)

## Dense H2 终态更正（2026-11-02）

上一版 Dense H2 行沿用了生成阶段旧计时，现以终态 observer、paired-read 和 cost-timing 凭证为准：真实开发集 **312/312**，936/936 原生请求，8 卡、2 个 TP4 实例，生成阶段 **1891.170 秒，6.061 秒/条**。learned SLU-F1 **66.7400%**，lock **66.0671%**，差 **+0.6729pp**；Intent 差 **−1.6026pp**，`candidate_accepted=false`，因此不晋升。持续八卡有效满载仍未证实。

效率拆解显示 HTTP 首部等待均值 313.866 秒/请求，轨迹槽位等待均值 129.206 秒/样本（P95 804.397），FIT 容量等待均值 3.762 秒/请求；contact lock 等待总计约 0.003 秒。下一轮并发探索必须新建 deployment profile，将 slot、queue、`max-num-seqs`、实例/卡数和预算绑定后再做真实数据匹配实验。完整 13,078 条 test 仍冻结，不用于调优。

# TFRL 当前任务状态与下一阶段计划

## 结论

TFRL 的 CPU 工程基座已经具备继续准备的条件，但真实 NPU 轨迹采集还没有形成可运行的 successor 队列。现在不应重复加固已经可用的服务和队列框架；应把已有 K001/K002 的 held 资产通过正式配置入口登记为新的候选，完成准入后接入空闲的 B 实例。当前 8200/A 有负载，8202/B 服务健康但空闲，八卡尚未持续有效利用。

## 已完成进度

- 固定 Qwen3-Omni-30B-A3B Thinking BF16 推理服务已运行：两个 TP4 实例、上下文 65536、`max_num_seqs=128`、`max_num_batched_tokens=2048`、显存利用率 0.92、SHM 多模态缓存、prefix cache/chunked prefill 开启。
- 完整 ON（P OFF → Slot ON → Decision ON）Fold A 1080 条已完成并通过正式配对核验：Intent 84.0741%、官方 SLU-F1 70.9703%、精确槽位 F1 61.4446%、Strict 587/1080；合同合法 1077/1080，仍有 3 条空答。
- ON 已作为当前精度工作基线，OFF 保留为效率和回退参照。测试集 13078 条仍只用于最终完整汇报，不参与调优。
- TFRL 的 GEPA、ACE、Reflexion、TF-GRPO 配置和反馈字段已分别整理；真实数据集、H1/H2 划分和来源哈希已固定，禁止重新抽样或用合成样本估算成本。
- K001/K002 各 625 条已经完成 CPU `stage → prepare → register`，但均为 `held`、`generation_authorized=false`、模型调用为 0；这代表可审的输入资产，不代表已采集轨迹。

## 当前 NPU 实验

| 实验 | 资源与设置 | 进度 | 当前判断 |
|---|---|---:|---|
| A：紧凑 P 提示，Slot ON，Decision ON | 8200，TP4，4 卡，客户端并发 64 | 1080/1080 P 已完成；Slot 817/1080 | 正在运行，评分异步进行 |
| B：P 操作辨别提示，Slot ON，Decision ON | 8202，TP4，4 卡，客户端并发 64 | P 583/1080 后容器退出 | 已知失败，不能自动重发；需按终态接续规则处理 |
| ON K2 K001/K002 | 8202 候选资源，2 个 625 条成员 | 0/1250 模型调用 | 尚未准入，不能占用 B |

A 端服务当前约 60 个请求运行、等待为 0，KV 使用约 45%；B 端服务当前运行和等待均为 0。当前没有证据证明八张卡都持续达到 AI Core 80%，因此不能把“两实例在线”表述为“八卡满载”。

## TFRL 基座现状与真实阻塞

已经具备的部分包括：固定镜像、外部配置启动、真实数据来源绑定、原生 FIT、断点文件、历史响应和成本留存、600 秒轮巡、CPU 评分与 NPU 生成解耦，以及 GEPA/ACE/Reflexion/TF-GRPO 的独立输入字段。

真实阻塞只有准入闭环：旧 ON 队列已经 `failed`，不能复用或盲目重发；共享 `flight.lock` 仍绑定旧 TF-GRPO 会话；K001/K002 的编译和阶段文件由 root 持有，尚未进入当前 release 下可读取的 candidate/readiness/experiment 树。8202 的 resident preflight 已通过，问题不在服务本身，而在候选、来源、策略、checkpoint、runtime receipt、session 和 PID 的一致绑定。

## 需要决策的内容

目前没有必须暂停工作的用户决策。建议沿已确定顺序执行：

1. 保持 ON 作为精度工作基线，继续保留 OFF 作为效率参照。
2. 不等待完整评分再派发 NPU；先完成 B 上 K001/K002 的 successor 准入，释放 8202 的空闲资源。
3. TFRL 第一轮先采集同策略 ON K2，再比较 GEPA、TF-GRPO、ACE 和 Reflexion；不能把旧 OFF K2 或单次 GEPA 诊断当作 ON 训练轨迹。
4. 并发先沿已配置的 C96 做单变量验证；只有真实终态显示吞吐提升、质量和合法覆盖不退化，才考虑 C128 或调整引擎参数。

## 下一阶段异步任务图

### CPU 任务

1. 在共享 guard 下审计旧 failed queue 和 lease 终态。
2. 将 K001/K002 的 held 资产通过正式注册入口写入 successor readiness、candidate 和 experiment 配置；不复制文件、不改内容哈希、不重新采样。
3. 绑定当前 B 的 profile、runtime receipt、session、PID/端口和服务启动时间，重新运行只读 preflight。
4. 编译 K001/K002 的 generation/FIT/launcher pins，设置 600 秒监控并保留断点接续键。
5. 独立完成 A 的 Slot 评分与 B 的失败原因归档；评分不阻塞 NPU 生成。
6. 统计每个节点的样本数、秒/条、模型调用、prompt/completion tokens、FIT 成本和失败/恢复成本。

### NPU 任务

1. 当前 A 继续运行到终态，不重启、不重复已知请求。
2. K001/K002 准入后立即进入 8202 队列，两个成员按依赖图排队，使用固定 ON 配置和断点续推。
3. 维持客户端供给高于实际在途请求，先验证 C96，再评估是否到 C128；不要同时改变提示、知识库、预算和引擎 geometry。
4. 每 600 秒或收到明确终态事件更新运行快照，报告“已处理/总量、秒/条、卡数/实例、running/waiting、KV、ETA 和成本”。

## TFRL 的科学顺序

第一轮先使用 completeON 的固定 625 条 construction roster，补采至少两个显式 seed 的 ON 轨迹；同分全错组标记为 `no-learning-signal`，不能强行制造优势。GEPA 先用于根因反馈和候选策略，TF-GRPO 需要真实同策略多 rollout 后再计算相对奖励，ACE 只能提交有来源的卡片差分，Reflexion 只能保存绑定真实失败和后继尝试的 episodic memory。任何候选都要在固定开发回归集上完成配对评分，再决定是否进入 Fold B 确认。

TFRL 不改变模型权重；后续 LoRA 属于独立阶段，仅更新主模型适配器并冻结音频塔、视觉塔及既有对齐投影。所有改动通过配置文件和脚本进入固定工程基座。

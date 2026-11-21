# P rollout 性能与 Ascend 升级判断

建议优先精简当前正式路径，同时准备独立镜像作版本对照。公开数据表明同款模型在昇腾有几十毫秒／输出 token 的实测；现网每 token 约 5.5–6.4 秒，串行轮次几乎没有引擎排队。升级值得测，但没有证据表明换一版算子包能消除这段差距。

| 对照 | 文本解码间隔 | 适用范围 |
|---|---:|---|
| [vLLM-Omni #10631](https://github.com/vllm-project/vllm-omni/pull/10631) | 平均 TPOT 67.47 → 55.84 ms | 同款 Instruct，昇腾 910B1、Thinker TP2，100 个合成多模态请求，1 req/s，固定 100 个输出 token，文本输出；变更是跳过 Talker payload |
| [vLLM-Omni #10037](https://github.com/vllm-project/vllm-omni/issues/10037) | A2 多模态文本输出 61.30 → 61.96 ms；A3 CI 文本输出 36.08 ms 基线，回退后约 58–65 ms | A2 对照为 v0.32 → v0.33，A3 为 `random-mm` 100/100、1 req/s。不同负载、设备和引擎路径，不能据此估计我们的升级收益 |
| 本次双 TP4 | 同组同实例，每路并发 1／2／4，后续 token 间隔分别约 5.53／6.37／6.26 s | 910B3，真实 P 语音，提示约 4.6k／5.4k token，输出约 10–12 token，实际第 3 步 LoRA。vLLM 的 ITL 按输出间隔计权，与公开 TPOT 只可作量级比较 |

# 公开基准的实际版本

55.84／67.47 ms 来自上游在昇腾 910B1 的稳态测量。实际环境是 vLLM **0.33.0**、vLLM-Ascend **`221fb096cd1cdf9f0b273da9b9624fbf4bb7f200`**、CANN **9.2.1**、torch **2.10.0** 和 torch-npu **2.10.0.post7**。vLLM-Omni 使用 **`e63fe779286d82c7a28856b873113958d3592281`**，两次测量只相差跳过 Talker payload 的原型。以 `vllm serve --omni --async-chunk` 启动，Thinker 使用 Model Runner V2、TP2、PIECEWISE 图和 64 个序列上限。其 NPU 配置关闭 prefix cache，第三张卡跑其余阶段。原型将平均 TPOT 降低 17.2%，P99 从 88.09 降至 78.89 ms。0.34 是合入目标，PR 只记录了 0.34 上的 CPU 测试。PR 未列出 A2 现场镜像 digest。[测量与命令](https://github.com/vllm-project/vllm-omni/pull/10631) · [Ascend 源码](https://github.com/vllm-project/vllm-ascend/commit/221fb096cd1cdf9f0b273da9b9624fbf4bb7f200) · [NPU 部署配置](https://github.com/vllm-project/vllm-omni/blob/e63fe779286d82c7a28856b873113958d3592281/vllm_omni/deploy/qwen3_omni_moe.yaml)

同项目 A3 CI 的 vLLM 0.33.0 使用 `quay.io/atlas-ci/vllm-ascend:v0.33.0-a3`，镜像摘要 `sha256:ec7cd7c98616d55500e21a17b63be3a99603d520087f41dc37b60ae6c75d0730`。0.33 上从 Runner V1 切换到 V2 后，纯文本长输出的并发 16 TPOT 从 117.63 ms 回到 48.76–51.55 ms。同期多模态文本输出均值仍在约 60–65 ms。[A3 CI 版本与运行日志](https://github.com/vllm-project/vllm-omni/issues/10037#issuecomment-6968740745)

Qwen 模型团队的技术报告给出 234 ms 理论首包时延，描述使用 vLLM 和 CUDA Graph，并对 MTP、codec decoder 使用 torch.compile。报告没有披露用于该表的 vLLM 版本或 commit。首个音频包与 P 阶段的后续文本 token 间隔采用不同计时边界。[Qwen3-Omni 技术报告表 2](https://arxiv.org/html/2509.17765v1)

当前 P 服务从原生 vLLM 的 HTTP 入口启动单个 Thinker，没有启动 `--omni` 多阶段管线。版本对照应以 0.33.0 + Ascend 对应 commit + Runner V2 为独立参考，检查 Thinker、TP4 LoRA、缓存和 raw logprob，再考虑迁移。上游跳过 Talker 的改动对只跑 Thinker 的路径没有同样的触发条件。

# Decode 同机对照

四套 TP4 分别做两组同机配对。第一组只去掉重复执行的原算子。第二组在 MoE 中跳过没有 token 的四专家权重块，四个 rank 在跳过通信前核对活动专家一致。两组使用相同发布版第 3 步 LoRA、两条真实语音、相同逐样本供给和采样参数，KV 和 prefix cache 均开启。串行轮每侧 2 个请求，并发 4 轮每侧 8 个请求。以下均为 vLLM 引擎记录的首 token 之后的平均 ITL。

| 候选 | 每实例并发 | 对照 → 候选 ITL | 对照 → 候选间隔数 | 变化 | 回执 SHA 前缀 |
|---|---:|---:|---:|---:|---|
| 去掉重复算子 | 1 | 5.487 → 5.561 s | 20 → 20 | 慢 1.4% | `8e1a4a36aed8` |
| 去掉重复算子 | 4 | 7.158 → 7.126 s | 81 → 81 | 快 0.45% | `b709b3d982bf` |
| 跳过空专家权重块 | 1 | 5.427 → 3.797 s | 20 → 20 | 快 30.0% | `e77c89fb8e35` |
| 跳过空专家权重块 | 4 | 5.989 → 5.685 s | 78 → 87 | 快 5.1% | `bf0479437dcc` |

空专家串行轮两侧各生成 22 token，序列及 logprob 逐值相同。请求排队均小于 0.1 ms，平均首 token 为 44.30 → 43.84 秒，批次墙钟为 197.18 → 163.65 秒。两侧均命中 2,176/9,990 prefix token。并发 4 时，两侧各成功 8 请求，生成 86 → 95 token，5 对序列相同且 54 个动作 logprob 逐值相同，另 3 对序列分叉，批次墙钟为 219.95 → 231.64 秒。去掉重复算子两轮的 10 对序列和 111 个动作 logprob 全部一致。四轮共 40/40 个请求正常终止并排空。

三个并发分叉发生在第 1、9、1 个输出位置。第 9 位置分叉前，8 个共享 token 的 logprob 逐值一致。并发轮的采样分叉先用同动作回放定位，后续更新态验收仍需覆盖缓存和 top-1。3.797 秒/间隔约为上游 55.84 ms/输出 token 的 68 倍。下一处优先核对每个 decode 步的专家权重搬运、投影 gather 与设备同步时间。

# 本地实测口径

同一推理主机、同一镜像的历史基础服务，两路各处理一次真实 P 语音，10 个输出 token，HTTP 耗时 5.23／5.56 秒；当时没有这套更新态 LoRA 和精度叠加，冷热、输入和批大小未配对。它比跨硬件公开数字更直接地说明旧镜像本身能运行得快。

本轮 6 次独立 cache salt 的短测共 48/48 个有效响应，已排空。每轮由两套实例处理两组，每组四个 seed；复用两条语音的同一份逐样本召回输入。吞吐不计正式训练。原始回执通过下表 SHA-256 前缀定位。

| 路由 | 每实例并发 | 8 请求耗时（秒） | 完成组／分 | prefix 命中 | 回执 |
|---|---:|---:|---:|---:|---|
| 同组同实例 | 1 | 304.91 | 0.394 | 29,568/39,960 | `4190e54ff5d1` |
| 同组同实例 | 2 | 219.93 | 0.546 | 29,568/39,960 | `505807186ac1` |
| 同组同实例 | 4 | 126.91 | 0.946 | 29,568/39,960 | `4764c57fc3af` |
| 同组拆分 | 1 | 349.81 | 0.343 | 24,064/39,960 | `057a243d1af7` |
| 同组拆分 | 2 | 242.33 | 0.495 | 24,064/39,960 | `9495ec3c96e3` |
| 同组拆分 | 4 | 154.74 | 0.776 | 24,064/39,960 | `4cdeb540f5a0` |

同组同实例并发 4 时，从输入就绪到发出 HTTP 平均 0.04 秒，实例排队 13.72 秒，prefill 28.65 秒，两路平均首 token 59.79 秒，平均完成 HTTP 121.99 秒。后续 token 间隔 6.26 秒；prefix 命中 74.0%，多模态处理缓存命中 6/8，没有请求抢占。各指标有嵌套且并行，不能相加。并发 1 时排队近零，首 token 22.83 秒。压测使用现成 wire，0.04 秒不代表正式数据的实际召回、序列化和供样开销。并发和路由对照各一轮，模型输出长度不同，因此吞吐差异含随机生成波动。

实际 `max_num_seqs=8`／实例，测试上限 4，正式采样草案也只在同组内并发。引擎日志报告约 123 万 KV token／实例、按 24,576 最大上下文可容纳约 50 个满长序列；这是 KV 容量估算。要测超过每组 4 个 seed 的持续利用率，需要同一实例同时供多个真实组，原租约下做有界补位。先单独升到每实例 8，确认持续吞吐后再测试 16／32，记录首 token、后续间隔、NPU/HCCL 等待、实际在途请求及请求队尾。知识与指令保持逐样本召回，KV/prefix cache 启用。

当前部署为 vLLM `0.30.0+empty`、vLLM-Ascend `0.30.0rc2`、CANN `9.2.0`、torch/torch-npu `2.10.0/2.10.0.post7`、主机驱动 `25.2.0`。现场读取实际文件，以封闭配置的 10 个摘要核对配套源码：

| vLLM / vLLM-Ascend | 源码匹配 | 判断 |
|---|---:|---|
| 0.30.0 / 0.30.0rc2 | 10/10 | 当前镜像在这些算子上与上游旧组合一致 |
| 0.30.0 / [0.30.0 正式版](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0) | 8/10 | norm 与 MoE router 变动；CANN 9.2.1，适合做小步兼容对照 |
| 0.32.0 / [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1) | 3/10 | attention、norm、MoE 相关路径及 LoRA row 有变动；CANN 9.2.1，许多发布说明中的加速针对其他模型。需要独立移植和验收 |

上游 0.33 的 A2 同机版本对照没有明显文本 TPOT 改善；其 text-only 修复主要作用于 Omni 多阶段 Talker handoff，我们的 P 路径需要先验证是否经过该环节。[vLLM-Omni 0.34.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.34.0rc1) 要求匹配的 vLLM 0.34 和 ModelRunner V2，并明确原生 0.34 Ascend 尚未验收。这里的候选指容器内运行时和算子组合，主机 OS 内核及 NPU 驱动不在本轮变更范围。

优先在现镜像做 A/B：保留已对齐的 BF16 LoRA、注意力和 MoE 输出，首先移除不被消费的旧算子对照，核实实际 cache 写入仍由原后端负责。当前活跃配置仍有 `preserve_original=true`；48 层路径执行额外原算子、按小块反复 TP all-gather 权重、同步读取设备上的 MoE 元数据，并将分页 KV 物化后按固定块处理注意力。没有开启图执行。以上由现网源码证实；各项占比需要独立 A/B 或短程剖析。若仍慢，依次验证空专家块、冻结权重复用、在设备上组织路由和批 attention；图执行应单独检验，否则现有 eager 包装器、Python 状态和同步点会影响图兼容性。

升级与算子精简均使用同权重、真实语音、冷缓存/跨录音部分命中、业务采样并发和热切换的配对回执。沿用已有一 token／整段 logprob 门槛各 0.01 和 top-1 回归门槛，并核实实际版本、缓存失效和每 rank 权重读回。正式 5,000 条训练进度仍为 0，Clock 3 来自有限验收。

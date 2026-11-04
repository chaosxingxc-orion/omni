# 外部 ACE 增量提案与消费

外部 ACE 候选具有独立 `external-ace-v1` 来源，表示离线 Reflector/Curator 增量提案适配，不是原生在线 ACE，也不伪造 TF-GRPO 的 native learning seal。

`ace_external.py collect --config PREPARED_DIR` 只收集已完成 wire；`compile --config SPEC --out PREVIEW` 核对原 prepared/collected、真实请求/响应、完整 source group、实际曝光和 actor 身份。SPEC 为 `saea-ace-external-compile-v1`，包含 prepared/collected 的绝对路径及 SHA、整数 candidate_index。所有入口不生成新 API 请求。

当前仅接受版本化空增量层，原知识冻结，只允许最多两个 ADD。每项保留 condition、content、exception，须提供真实 TRAIN 证据反思与反例。程序按这三个字段的精确哈希归类和去重，保留全部操作索引，不以模型筛选或相似度删证据。模板只供给规则，源样本、gold、反思与usage留在私有审计材料。

独立 review/release 使用 `saea-ace-external-review-v1`、`saea-ace-external-release-v1`，绑定候选、来源、固定H2消费者、TRAIN625音频视图、非空selection及逐项支持证据。共享校验保持 query、utterance、audio 三层学习/消费/selection隔离。`prepare --config CONFIG --out PARAMS` 接收与TF-GRPO发布入口相同的路径/模型/template/consumer参数，但调用ACE专用 replay、render 和审查 schema。

已有 `external_experience_workload.py` 接受此来源，经原消费模块与Flight执行同P learned/lock配对；实验 metadata 明确ACE算法身份。条件当前由模型阅读，不是程序性的运行态 applicability gate，`programmatic_applicability_gate=false`。真实候选还需要人工语义审查、tokenizer/FIT和NPU实验；通过编译不代表有效。

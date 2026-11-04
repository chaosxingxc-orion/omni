# 外部多阶段 TF-GRPO 配置执行

## 行为与算法范围

`tfgrpo_external_reference.py` 将固定真实轨迹送入既有 `ExperienceLearner`，保留 summary→advantage→group_update→batch_update 的依赖链和原始版本化阶段模板。实际教师由 `ExternalTeacher` 调用并核验 provider wire；actor 只负责原始轨迹与候选归属。输出仍是待语义审查的经验库，不签发 native seal 或部署批准。

当前明确标记 `tfgrpo-reference-multistage-external-dense-offline-v1`：阶段机制是已有参考实现，连续 intent/slot 奖励、离线历史轨迹和外部教师属于本地适配。它与一次生成多个提案的融合式路径分别报告。summary 接收完整 query 与无损去重的原始 trajectory；advantage 使用原 query、TRAIN reference 与带奖励的模型摘要；group update 使用父经验与语义优势；batch update 使用父经验与操作。后两阶段不会悄悄补完整轨迹，因此不是 grounded 融合扩展。

## 配置字段

job schema 为 `saea-tfgrpo-external-reference-job-v1`，必须显式给出：

- `source`、`source_batch`、`parent`：绝对 path/sha256；source 使用完整证据文档，batch 是其 `source_batch_sha256` 绑定的原始完整学习 batch。
- `query_ids`：固定且有序的 TRAIN 子集；`actor_model_id` 必须与 parent、原 batch 和所有 rollout 一致。
- `template`：`upstream_commit` 与 `source_sha256` 对应 `rails/tfgrpo_prompts.py` 常量。
- `teacher`：既有外部教师配置，含 model/endpoint/api_key_env、固定 `research_prompt` pin、stream/timeout、max_tokens 和 analysis_budget。密钥只从环境读取，不进入配置文件。研究策略不得覆盖每个阶段已有的输出协议。
- `stage_completion_caps`：summary、advantage、group_update、batch_update 各自显式预算；本版本均必须等于 teacher.max_tokens 且至少 131072。reasoning 的 49152 是提示目标，不冒充 provider 硬上限。
- `bounds`：max_calls、concurrency、max_operations、max_items、max_characters、num_experiences、max_completion_tokens_total，全为正整数。
- `objectives`：agent_objective、learning_objective；`output_root`：新建私有持久目录。

固定 16 组、每组 4 条、每组有奖励差异时，计划为 64+16+16+1=97 次调用；131072 每调用的最坏输出分配为 12713984 tokens。这是预算几何上限，不是预计消耗或服务容量证明。没有 provider tokenizer 时输入 tokens 与容量验证如实为未知；不要使用字符数冒充 token 数。样本相同且预测重复也保留每条轨迹权重。

## 命令与断点

在固定工程镜像中运行：

```bash
python scripts/analysis/tfgrpo_external_reference.py prepare /private/job.json
python scripts/analysis/tfgrpo_external_reference.py run /private/prepared
python scripts/analysis/tfgrpo_external_reference.py collect /private/prepared
```

prepare 不调用模型，验证源文件、原始 group hash、完整证据投影、每条 trajectory 的 actor/policy/parent、奖励公式与 scored response hash，再固化阶段计划和模板。对原始响应的哈希绑定证明其与该来源一致；它不替代原上游归档/seal 审计，也不声称已独立听音。

run 应由既有脱离 SSH 的配置化进程管理启动；再次运行同一目录只复用已知响应，未知请求由原持久化机制拒绝重发。collect 要求完成 candidate 已存在，不启动缺失阶段，并重验每条原始 provider wire 与 learner receipt。不要复制 candidate 单文件代替原 teacher/learner 目录；路径 pin 必须在运行容器中可解析。

CPU prepare、API 学习和 NPU 评估相互独立。该入口不会自动评分、发布经验或启动 NPU；后续需独立语义审查和显式外部来源 release 适配，不能借用原生 seal。未取得真实配对结果前不能宣称算法收益。

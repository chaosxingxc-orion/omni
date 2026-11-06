# E6：ACE 启发的单卡构建配方

固定教师模板为 `configs/prompts/learner/tfrl/ace-slot-construction-v1.md`，动态输入由 `scripts/analysis/ace_slot_recipe.py` 离线组装，例子为 `configs/learning/recipes/ace-slot-construction-v1.example.json`。原有 `ace.md` 的 ADD/MODIFY/DELETE 不能直接送入本编译器。这里仅借鉴先反思后提案；完整 ACE Curator、在线迭代、冲突合并均未实现。没有真实新轨迹和响应时，不生成候选、不声称有效。

## 私有输入与证据检查

配置中的所有输入均是 JSON 文件的绝对路径与文件字节 SHA-256，替换例子占位符后放私有目录。`authored` 是与待编辑原始 snapshot 对应的 `saea-authored-semantic-knowledge-source-v1` 完整原始 rows，不能合成或改写定义。`records` 是非空数组：每条至少有唯一 `source_id` 和非空 `source_sample_ids` 数组；每个完整 TRAIN 样本 ID 必须以 `source_id + ':'` 开头，包含该反馈的全部传递 TRAIN 祖先。其余字段保留真实 H1 轨迹、节点请求、实际供给的目标卡正文及标识、模型输出、评分/参考及来源；明确区分真实失败与真实反例、缺失信息和竞争解释。`pool` 是真实 H1 构建池及其身份/来源的 JSON；`forbidden_sample_ids` 是包含 H2、Fold A/B、TEST 和其他消费者的完整禁用 ID 数组。不得把这些禁用分区作为新教师反馈。配置的 slot 必须来自真实原卡且在相关 H1 失败请求中实际送达；勿因历史上称某卡有问题而直接选择它。新轨迹至少要有可追溯 TRAIN 身份与完整祖先、原始定义、实际送达证据、节点输入输出、反馈/失败及可用反例；缺失时停止或明确不足。

拼装器校验文件 pin、原始 authored schema、已有卡及允许字段、合同边界、祖先格式与禁用身份交集。它不核验记录确实来自 H1、供给和失败的真实性、祖先完整性或反例质量；这些必须由运行者先基于原始 custody 审查。编译器另核对原 snapshot 与全部 base rows、响应请求与 wire、禁止集和单卡边界；历史 snapshot 的禁用交集会报告，不自动清除，评估资格仍需审查。

## 命令和绑定

以下命令在固定镜像内从 study 根目录执行，`/PRIVATE` 是替换为实际私有挂载的占位符，源码只读、CPU、无网络完成准备及编译。确保 `PYTHONPATH=src`。目录必须为全新目录，输出含真实材料，禁止提交或公开。

```sh
PYTHONPATH=src python scripts/analysis/ace_slot_recipe.py --config /PRIVATE/recipe.json
```

输出 `base.json`、`contract.json`、`dynamic.json`、`messages.json` 及 `pins.json`。`messages.json` 是唯一 user 消息，其 content 是完整 dynamic JSON；合同中的 `base_rows_sha256` 和每个字段的 `previous_value_sha256` 是现有 `content_hash`，不是文件 SHA。原始 source 的文件 SHA、snapshot 文件 SHA 和 `pins.json` 的文件 SHA 均用 `sha256sum` 获取。固定模板文件也用同样方式取 SHA；运行配置内路径用绝对路径，因为编译器直接按路径校验 prompt pin。

新 H1 证据审查通过后，在私有目录按 `saea-external-learner-v1` 建真实教师配置：必需 `endpoint`, `model`, `api_key_env`, `messages`（输出 pins 里的 messages pin）, `max_tokens`, `timeout_seconds`, `output_root`, `research_prompt`（模板绝对路径和 SHA）；按现有 executor 要求 `max_tokens >= 131072`。服务地址和凭据仅放私有配置或环境。教师调用需另行安排；配置准备不代表调用或响应成功。调用命令：

```sh
python scripts/environment/configured_external_learner.py --config /PRIVATE/teacher.json
```

对真实成功响应，从 `candidates` 显式选择一个索引（空数组则停止），制作 `saea-external-slot-candidate-config-v1` JSON，**恰好**含 `schema`, `original`, `base`, `contract`, `dynamic`, `teacher`, `candidate_index`, `forbidden_sample_ids`, `output`。所有文件项为 `{"path":"绝对路径","sha256":"文件字节SHA256"}`：`original` 是原 snapshot 文件；`base`、`contract`、`dynamic` 取 `pins.json`；`teacher` pin 指向上述 teacher 配置文件；`forbidden_sample_ids` 与配方一致；`output` 为不存在的新目录。保留教师 output_root 内 intent、request、result、原始 wire（stream 时含 response.sse）及响应和成本。不得对未知状态盲重发。

```sh
PYTHONPATH=src python scripts/analysis/external_slot_candidate.py --config /PRIVATE/compiler.json
```

输出仅 candidate-only，未授权提升。后续要用 PSD adapter，复制 `configs/evaluation/psd-slot-knowledge-v1/publication.example.json`，填写真实 data_root、compiler 文件 pin 和新的 data_root 相对 output，再运行：

```sh
PYTHONPATH=src python scripts/analysis/psd_knowledge_adapter.py --config /PRIVATE/publication.json
```

adapter 重建并核验候选，发布 descriptor 不等于完成实验评估。其消费时还检查构建祖先与消费者隔离。独立审查 reflection、单卡替换、历史交集与未见反例，再按既有评估流程执行；无需把 reflection 插入严格的 proposal 内。

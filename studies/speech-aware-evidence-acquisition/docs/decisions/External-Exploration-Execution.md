# 配置驱动的外部TFRL候选生成

## 入口与职责

`scripts/analysis/external_exploration.py prepare JOB.json`离线准备输入，`run OUTPUT_ROOT`发送单次外部请求，`collect OUTPUT_ROOT`只读核验已有完整响应并保存提案。未知请求不重发；原始费用和响应由既有ExternalTeacher及执行器保留。prepare会冻结模板、执行器、实际消息和context，源数据变更需新任务目录，不能覆盖已发任务。

四个独立profile位于`configs/learners/tfrl/`，对应独立完整模板`configs/prompts/learner/tfrl/`。分别要求TF-GRPO相对反馈、ACE反思整理、Reflexion记忆重试、GEPA父提示变更。只有传输、来源和预算实现共享。

## 任务配置

配置schema为`saea-external-exploration-job-v1`，字段为：

- `profile`、`source`、`context`：各为`path/sha256`文件pin，相对路径按JOB目录解析。
- `query_ids`：固定、互异的TRAIN名单；source使用完整开发证据文档`document.groups`，每组包含query_id、group_sha256和evidence。
- `teacher`：既有外部执行器参数，含schema、endpoint、model、api_key_env、timeout_seconds、stream等；不含messages/output_root/research_prompt，由入口生成。凭据仅环境变量。
- `output_root`：新的私有持久目录。

prepare按每组字典恢复引用和canonical JSON字符串，要求实际request、supplied_evidence、task_instruction；不删除重复rollout及其权重。原始完整输入不进公共仓。profile固定至少131072输出预算，具体输入容量以provider计量为准。

## 产物与验收边界

`prepared.json`记录来源及冻结文件；`collected.json`保留经ExternalTeacher核验的响应身份、wire哈希、usage和提案。GEPA逐候选核验父哈希、contract和允许修改模块；其余算法核验各自提案字段。结构检查不是语义批准，更不是知识更新或部署成功。语义审查、各算法编译及真实NPU配对仍是独立阶段，收集产物明确标pending。

外部GEPA候选可通过 `saea-gepa-external-mutation-v1` 来源接入 `gepa_pair_material.py`。来源包含prepared与collected的path/sha256及candidate_index，必须与原始wire、可信seed的parent/contract一致。actor仍使用已有tokenizer/FIT与配对编译；同一教师请求的多个候选共享一次调用成本，按request哈希去重计费。保留原始绝对路径挂载，不静默改写已固定请求。

发送前完整轨迹进行可逆字典打包；排除名单正文留本地，教师context只带其数量/哈希，不改变本地合同校验。不声称四算法完整闭环已运行。正式测试使用固定镜像、无网络模拟服务；真实调用及收益分别记录。

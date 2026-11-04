# 故障恢复后的后继配置与准入缺口

## 已准备的真实配置

知识构建四臂已从原生产prepare配置派生：full-query、full-query-task、diagnostic-task与compact-control。每臂固定原51组、原source/roster/诊断材料、64K上下文、16384输出、3072静态余量、2条compact上限、相同字符与累计预算；统一采用显式关联隔离策略。仅新仓库/输出位置、name/run ID和policy改变，逐字段差分私有归档。compact与rich的runtime identity内容相同。

四臂正常计划408次生成，FIT另列；153/204工作组并不代表153/204个独立数据样本。旧失败响应和成本保留，旧18组compact不充完整对照。

消费侧新增evidence-use part-b的独立node-grounded-feedback-v2草稿，保留原两条反馈和source pins，改变name/run ID及提示策略。它不是重发原请求，也不替换三个已成功候选的paired16。新提示需重新测容量/FIT；若评估节点差分本身，应以显式ID策略作消融，而不是将v2-v0所有差异归因于节点差分。

## 状态与资产缺口

以上均为DRAFT，未prepare/register/admit/preflight/dispatch，没有新模型调用。现有固定learning_strategy_job.py与GEPA配置入口可消费，未编写新一次性启动器。

构建source-spec、原51组roster及封存source manifest映射到本地热盘。此前将缺少tokenizer.json判为缺失资产不准确：此模型使用vocab.json、merges.txt和tokenizer_config.json。应核验实际tokenizer身份及容量，不能要求不存在于原模型格式中的文件。消费还需要真实音频和知识/评分依赖。

Ceph读取已于11月4日恢复，原失败音频可读且A0新预检通过；历史blocklist不再列作当前阻塞。本地输入检查只是挂载根与pin可读性，不能冒充完整依赖快照或READY。

## 执行顺序

先核验授权恢复后的文件身份与原服务健康，再以新冻结树做prepare、注册、准入、环境绑定。A0用原恢复入口接续；两探索lane独立运行。消费原候选配对及CPU评分优先闭合，构建四臂按统一变体对照。后继仅在真实前置满足时进入可执行队列，不以草稿数量报告NPU负载。

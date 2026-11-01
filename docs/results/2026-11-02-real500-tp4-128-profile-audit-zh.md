# TP4-128 真实基线 Profile 审计

TP4-128 的真实 500 条结果可以作为候选工程配置：8 张卡、2 个 TP4 实例、`max_num_seqs=128`、1000 次生成、约 23.05 分钟、2.766 秒/条。该结果已经完成断点封存和官方评分，固定数据集无需重跑。

本审计明确把它标记为 `candidate-only`，不直接铸造 TFRL port deployment binding。当前缺少该次 TP4-128 运行的公开 runtime receipt/session identity 对、detached smoke checkpoint 身份和连续逐卡任务利用率证据；历史逐卡采样不能证明 8 卡持续有效满载。synthetic width binding 也不能替代真实 profile。

后续只有在精确 runtime/session receipt、启动链、checkpoint 和 owner admission 绑定完成后，才能将该候选用于 TFRL smoke。此过程不会重复 500 条推理，只补齐身份和运行证据；若证据不完整，准入审计继续 fail-closed。

# ChatGPT研发协作 Token 消耗与优化

**用户已明确：要节省的是与ChatGPT协作研发的交互tokens。** 优化对象包括主/子线程往返、重复上下文、工具回传和状态协调。业务系统推理与TFRL探索按效果需要配置预算；知识消费前归类去重属于独立功能，不能计作本项节省。

## 审计结论

既有快照截至2026-11-03 09:36:02 UTC：229个Omni会话、21442条唯一响应，输入2,808,195,151、输出9,028,698 tokens。主线程占总记录量78.28%；缓存占输入96.09%，非缓存输入109,806,817。平均输入130967/响应，P95为234215。以上是日志累计口径，不是账单或独立文字量，不能计算实付费用与总体浪费比例。

## 执行方案

- 当前记忆原位维护为不超过8KB的状态索引；完整历史保留，按需检索。
- 子智能体接收最小任务包，交付完整报告，聊天只回结论、阻塞、终态和证据路径。
- 固定脚本与配置负责启动、队列、清算和归档，减少逐操作协调与重复实现。
- 后台约600秒或终态事件监测；工具返回短摘要，原始日志落盘；已验证且未变化的检查不重复运行。
- 按工作窗口核对响应次数、上下文长度、非缓存输入、输出及每个有效交付的交互量，同时保留实验质量、覆盖和队列效率。

范围纠正、报告和交付契约已更新；实际节省仍需后续同口径统计验证，不预报百分比。历史业务推理统计保留在报告附录，不作为此次优化目标。

[完整报告](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/Token-Consumption-and-Optimization.md) · [用量快照](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/Orchestration-Token-Usage.json) · [子智能体交付契约](https://github.com/chaosxingxc-orion/omni/blob/main/docs/engineering/Subagent-Delivery-Contract.md)

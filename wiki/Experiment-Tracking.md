# 实验总账

更新时间：2026-11-03 02:30 UTC。当前状态原位维护，历史过程与成本保留在Git及原实验凭据中。

| 实验 | 完成量 | 结果 | 资源、效率与剩余 |
| --- | --- | --- | --- |
| A0 / Direct，Fold B | 各1080/1080 | SLU 73.2225% / 45.1178%，A0 +28.1046pp | 开发集收益，剩余0 |
| Direct完整test | 13078/13078 | SLU 46.5558%，Intent 77.8789%，FINALIZED | 8卡2×TP4，每实例C96；6415.778秒，0.4906秒/条，剩余0 |
| A0全测试查询编码 | 13078/13078 | SEALED；原始语音查询封存，非A0精度结果 | 4worker/64CPU核/0NPU；12529.962秒，0.9581秒/条，剩余0 |
| A0完整test推理 | 未启动 | alias/content各13078完成，agreement亦13078完成；无完整A0指标 | ETA未知 |
| Dense TFRL H2 | 312/312，936响应 | learned/lock SLU66.7400%/66.0671%，区间跨0，Intent−1.6026pp，不晋升 | 8卡2实例；生成1891.170秒，6.061秒/条；剩余0 |
| ACE v2 | 4/51组、9调用 | 输出截断，ABORTED，无候选 | 762.058秒；输入37652/输出32897 tokens，无完成ETA |
| Reflexion compact v2 | 16/51组、17调用 | 缺schema；存在错误槽位规则，无候选 | 1118.258秒；输入66306/输出53062 tokens，无完成ETA |
| Reflexion errors v3 | 7/51组、8调用 | 缺reflection外层，ABORTED，无候选 | 572.872秒；输入37207/输出25057 tokens；已结清归档，无完成ETA |
| GEPA | CPU组件已实现 | 真实Solver搜索及H2未接入 | 无模型实验结果 |

当前仅两推理服务驻留，无生成任务。八卡持续有效满载、稳定TFRL收益、跨轮RSI均未实现。完整测试未用于调优。

上述秒/条各有统计边界，不混用CPU编码、并发总吞吐、生成阶段和单请求延迟。ACE/Reflexion串行反思链不等于八卡满载；历史共享轨迹成本不重复累加。

[当前研究状态](Research-Status) · [完整计划与决策事项](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/Research-Status-and-Plan.md) · [真实终态费用](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/tfrl-mechanism-results.json)

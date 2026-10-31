# 实验总账

更新时间：2026-10-31 05:18 UTC

研究顺序：先完成 Direct/A0 收益、稳定性、成本及错误分析，再开展 training-free RL，保留固定 A0 控制组。方法冻结后，Direct、A0、最佳 TFRL 均在完整 13,078 条 SLURP test 上报告指标；此前 test 子样本报告未用于调优，不视为污染。

| 实验 | 数据范围 | 当前状态 | SLU-F1 | 证据边界 |
| --- | --- | --- | --- | --- |
| Direct lock | TRAIN Fold B 1,080 行 | FINALIZED/PASS | 45.1178% | 完整官方评分 |
| Direct aa | 同上 | FINALIZED/PASS | 45.6633% | 重复噪声估计 |
| Direct ab | 同上 | FINALIZED/PASS | 45.2940% | 重复噪声估计 |
| 初始 Agent A0 | 同上 | 正在推理，已核验 190 个持久化样本 | 待完成 | 局部 checkpoint 非全量官方评分 |
| Fold A 编码、检索准备 | TRAIN Fold A 1,080 行 | 已排队，等待 A0 和 paired read 成功 | 不适用 | 准备工作不是任务指标 |
| 当前 A0 方法的 Fold A 评估 | TRAIN Fold A 1,080 行 | 独立准入、完整推理及六列诊断已入队，待上游 | 待运行 | 真实预检与推理尚未执行 |
| TFRL | 首轮 625 条 construction rows | 等待 D0/A0 lock | 待运行 | 313/312 split，K=4，最多三轮 |
| 最终三系统比较 | 全部 13,078 条 test | 方法冻结后执行 | 待运行 | 10,068 子集仅作补充 |

旧 Agent 部分运行已封存，不与当前 A0 混合。中间准备失败保留为工程诊断，不纳入效果表。TFRL 相对固定 A0 至少提升 1 个百分点且通过完整评估后才接受。

[详细运行与工程记录](https://github.com/chaosxingxc-orion/omni/wiki/2026-10-31-A0-and-NPU-Scheduling)

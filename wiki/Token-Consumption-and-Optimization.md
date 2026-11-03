# Token消耗与优化

原专项子智能体未交付完整审计，主线程已补齐。截至2026-11-03 09:36:02 UTC，本地229个Omni会话、21442个唯一响应，日志输入28.08195亿（其中缓存26.98388亿）、输出902.87万；缓存占输入96.09%，非缓存输入1.09807亿，平均输入13.10万/响应。数据是provider日志口径，不是费用账单，且不是独立文字量。

主线程占总记录tokens78.28%，其他Omni线程21.72%。目前不能精确分配多少属于必要基建或重复管理，因此没有可信总体浪费比例。

业务历史625条共享P/K4实验为4375调用；337条未获相对基线收益、121条无意图正确轨迹、68条正向credit、99条基线不稳定。正向credit占已统计input+output约11.54%，不等于最终精度收益；其他88.46%也不能全称浪费。291/625条四候选归一化后一致，提示应增加可验证的路径变化而非机械重复采样。

优先措施：子智能体完整报告和证据验收；当前状态与历史记忆分离；统一脚本驱动生命周期；事件/600秒后台监测并输出摘要；开发态允许长反思但要求假设—干预—重跑闭环，运行态预算独立优化；稳定材料及表示复用。每项都以实测效益验收，不预报无依据的节省比例。

[完整审计报告](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/Token-Consumption-and-Optimization.md) · [用量快照](https://github.com/chaosxingxc-orion/omni/blob/main/studies/speech-aware-evidence-acquisition/docs/results/Orchestration-Token-Usage.json) · [子智能体交付契约](https://github.com/chaosxingxc-orion/omni/blob/main/docs/engineering/Subagent-Delivery-Contract.md)

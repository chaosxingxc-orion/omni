# PSD v2 Decision 策略配置

`intent_slot_workload.py` 现在支持 `decision_policy`，与 `perception_policy`、`slot_policy` 使用相同的 thinking 配置和带 SHA256 的模板引用。Decision 对应内部 `solver` 角色，仍只返回候选 ID，最终实体从所选 Slot 结果组装。此入口不改变知识库、输出协议或候选集合。

将 `configs/learning/examples/psd-v2-decision-policy-overlay.json` 的字段合入既有 Fold A/B workload 配置；它是覆盖片段，不是可独立启动的实验。须同时保留显式 v2 SDK pipeline。未配置该字段时行为不变。模板存于 `configs/probes/`，更换内容后更新 SHA256，再通过现有 stage→prepare→register→admit→run 流程生成独立实验身份；不得热改运行目录。

该入口允许 GEPA 等候选只改变 Decision 提示，保留 P、Slot 和知识供给，便于节点归因。thinking 也可配置，但首轮提示对比应沿用基线 thinking，不同时改变两个因素。教师模板及候选编译器仍需明确适配三节点任务；此入口不代表四种 TFRL 算法已完成适配。

验证覆盖：仅 Decision 配置改变、实际 phase instruction 渲染、模板 hash 不符拒绝、无 v2 pipeline 拒绝和既有 workload 配置回归。尚未进行候选真实模型实验，不宣称精度收益。

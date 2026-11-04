# Solver 配对输出协议实施

## 已实现与实验边界

独立 `protocol_variants` 轴复用共享 Flight 和多臂执行器：一份 P 输出供两个 S 使用，`constrained` 使用 TRAIN ontology 构造的 label-only JSON schema，`lock` 保持原请求格式。正常每条3次生成，固定80条240次生成与240次FIT，生成含单次 answer-only 接续上限480。该实验是输出协议工程消融，不冒充TFRL算法。

模块配置使用 `kind: protocol`、`protocol_frame`，不放宽 sampling-only 的 `decoding_variants`。原知识、P/S模板、音频、3072 thinking +1024 answer、采样、seed、normalizer和评分器保持一致。schema使用真实scenario/action配对和开放filler，不删除低频槽、不按意图闭集裁剪实体。

## 可信度与接续

schema由 Flight 的 governed TRAIN ontology loader 注入；声明、schema请求字节、protocol receipt、Flight封存及官方reader都绑定协议轴。reader比较两臂实际请求，除输出格式及显式contact标识外拒绝其他变化。若仅一臂触发fallback，校验原始两臂输入相等，再验证fallback仅改变thinking和answer reserve，并保留相同输出格式。未知调用保护和共享恢复机制继续生效。

固定镜像CPU测试实际执行OpenJiuwen图→请求→receipt→Flight输出封存；此外验证配置拒绝、采样轴隔离、answer-only真实adapter接续及reader。CPU假传输只验证实现，不宣称NPU吞吐或精度收益。仍需实际引擎检验grammar与reasoning parser边界、延迟和正确率。

## 配置化使用

`configs/probes/solver-protocol/development80.json` 复用之前Reflexion80原名单，未重抽数据，也不是未曝光验证集。对应material记录原roster与基线哈希及原模板208 tokens。部署别名和runtime receipt复用运行主机已有配置，部署实物必须按hash和实际实例核验，不从公开文档伪造。

固定镜像内使用：

```bash
python scripts/analysis/solver_protocol_workload.py stage --repo "$REPO" --data "$DATA" --config "$CONFIG" --out "$STAGE"
python scripts/analysis/solver_protocol_workload.py prepare --repo "$REPO" --data "$DATA" --config "$CONFIG" --stage-record "$STAGE/complete.json" --out "$PREPARED"
python scripts/analysis/solver_protocol_workload.py register --repo "$REPO" --data "$DATA" --config "$CONFIG" --stage-record "$STAGE/complete.json" --prepared "$PREPARED/complete.json"
```

之后沿用配置化admit/launch/queue和 `stage3_rollout_score.py --mode supply-variants --stage all`。启动权限与资源由主线程安排，本交付0新API/0NPU。报告须同时列协议合法率、官方SLU/intent/slot、strict修复/破坏、tokens、P/S耗时和完整样本进度；没有真实80条结果前不晋升。

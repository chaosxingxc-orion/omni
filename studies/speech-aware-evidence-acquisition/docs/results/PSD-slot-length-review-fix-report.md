# Slot length 审查修正

基于 `54f030390a594e01597f476752682de446fdfd11` 修复：复制原chat template mapping并只更改enable_thinking，保留其他配置，非法mapping明确拒绝；max_tokens与thinking_budget按原策略变化。P length纠错先耗尽共享extra=1后，Slot length无额外重问，真实SDK脚本路径保留3次请求、响应、COST和3个FIT桩；输出有限终止，canonical output/trace校验通过。没有伪造答案。

固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`；ssh npu2、CPU、`--network none --read-only`、源码只读、tmpfs，PYTHONPATH `/src/src:/src/scripts/analysis`。测试日志：

```text
/opt/saea-venv/bin/python -m pytest -q --tb=short --show-capture=no -p no:cacheprovider /src/tests/unit/test_intent_slot_runner.py -k 'slot_repair_preserves_non_thinking_template_options or perception_repair_exhausts_shared_budget_before_slot_length or slot_nonempty_length_repair' --basetemp=/tmp/pytest
3 passed, 73 deselected, 5 warnings in 11.90s
```

源码pin：`intent_slot_pipeline.py` `8facba18c1d4e32916279def8a0d520f6ab345340e20cfff27b398afb243a946`；`test_intent_slot_runner.py` `94198e20ad52c18415b8f4deedafb28026dd516bdc62665f238dd2bfa07a65f9`。远端测试副本逐项相同。真实SDK、模拟模型与tokenizer；无业务模型、教师、真实FIT或NPU调用，未做真实注册、部署或Fold覆盖率测量。此前136项回归未重复。

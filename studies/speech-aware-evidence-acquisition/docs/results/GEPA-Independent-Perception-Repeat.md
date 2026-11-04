# GEPA独立Perception重复控制

现有GEPA P四臂配置使用`template_frame.twin=true`，历史twin语义让AA的Solver复用lock的P状态；AA自身虽也调用P，其输出不进入AA的S。真实六条完成轨迹中五条AA P与lock P不同，原四独立臂审计正确拒绝。这是实验配置语义与声明不一致，不能当模型错误或修改历史轨迹消除。

新增配置在`saea-gepa-perception-pair-build-v1`中显式设置：

```json
{"repeat_scope": "independent-perception"}
```

该选项编译为`saea-gepa-perception-template-pair-v2`。只有v2在AA Solver渲染与知识供给时使用AA自己P的状态；P输入仍与lock相同，P/S共八次主调用，预算、旧身份、旧v1行为不变。遗漏此字段保持旧v1来源重放；严格四独立臂reader会在评分前明确拒绝v1，不能以宽松审计掩盖差异。

回归用不同的AA/lock P文本验证实际八contact图：旧模式AA S仍读lock，新模式AA S读AA且按AA内容重新获取供给；同时覆盖新声明的native custody与未知模式拒绝。正式启用需独立冻结后继配置并保留旧run全部成本，不能热改正在运行的配置。当前修复不证明精度提升。

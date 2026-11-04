# Solver输出协议可行性与实验设计

## 结论

当前Solver不传response_format是冻结配置显式选择，P节点有独立schema；传输层没有丢字段。ACE/Reflexion的双JSON和非法scenario/action属于模型输出协议偏离，不能单凭现象判程序bug。固定镜像vLLM 0.30.0、xgrammar 0.2.8具备编译完整合法意图联合schema的能力；本轮只有CPU验证，0模型调用、0卡。

新增注册格式`slurp-solver-paired-label-json-v1`从已有governed TRAIN ontology读取60个真实scenario/action配对和55个实体类型。它要求单对象与完整外层字段，按scenario分支限定合法action，不执行笛卡尔积；所有filler仍是任意字符串，不限制槽位共现、实体数量或重复出现，也不引入当前样本答案。原默认配置不变。

## 固定镜像证据

`solver_protocol_probe.py`用镜像内xgrammar编译完整oneOf schema。合法pair和未见filler可通过，非法pair、缺字段和双对象被拒绝。320份已有Solver原始响应里317份符合JSON schema，另3份不合；剥离外围空白后的grammar结果一致。原始响应没有被改写，strip/canonical仅用于解释grammar接受范围。

grammar默认`any_order=False`，对象键顺序也被固定。源代码vLLM后端没有传`any_order=True`。这是一项额外生成偏置，必须在实验解释中披露，不能把格式约束当作完全不影响模型决策。vLLM默认`enable_in_reasoning=False`，qwen3 parser负责定位推理结束后再施加grammar。真实forced-close、fallback、吞吐和有效输出率仍需NPU验证；未修改服务或镜像。

## 接线边界与下一实验

现有`decoding_variants`和`request_decoding_overrides`刻意拒绝response_format，避免采样轴混入输出协议。现有cited-answer组件会改答案封装和normalizer，不宜借用。本次只注册格式与交付草案，**不是可直接启动的成品workload**。实施严格逐臂协议仍需：独立协议轴声明、逐S臂格式绑定、receipt/custody和预算验证、真实请求差异审计、官方配对reader。

`configs/probes/solver-protocol/paired-development-plan.json`固定复用已有开发80：共享一份P与知识供给，两个S的音频、原文prompt、thinking3072+answer1024、采样、seed、normalizer和scorer保持相同，唯一配置因素是response_format。正常240生成及240FIT；fallback上限须按最终图编译，不能沿用未核验值。名单已曝光，只作开发验证。

报告strict JSON、合法pair、有效输出率、intent/slot/SLU、strict修复与破坏、raw→delivered变化、P/S耗时、thinking/content tokens与grammar等待开销。不会通过宽松解析把无效输出改成有效，也不会只汇报格式成功率代替系统精度。

## 复现

固定镜像内：

```bash
PYTHONPATH=src /opt/saea-venv/bin/python -m pytest -q tests/unit/test_slurp_solver_protocol.py tests/unit/test_slurp_closed_label_inventory.py
PYTHONPATH=src python3 scripts/analysis/solver_protocol_probe.py --inventory configs/probes/solver-protocol/slurp-label-inventory.json --output /tmp/probe.json
```

业务venv测试31 passed、2 skipped（原数据集可用性集成用例），新schema测试全部通过。schema探针用镜像系统Python的xgrammar，无网络、无模型权重加载。私有320条响应仅在交付目录供审计，不发布。

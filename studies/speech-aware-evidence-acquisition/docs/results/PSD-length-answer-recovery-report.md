# PSD Perception 非空 length 有界纠错

已实现可配置的 `compact-reask-v1`：Perception 得到已知、非空且 wire 终态为 length 的响应后，在原音频、原知识、原 schema、原采样设置上追加一条固定诊断指令，单次重新请求真实模型输出；成功后继续 Slot→Decision。初始失败文本不会被截取补全，也不填默认标签或伪造实体。当前未做真实Fold运行，不能声称100%有效回答。

入口为 NodeWorkload `pipeline.perception_length_recovery: "compact-reask-v1"`，必须同时配置现有 `coverage_recovery` 共享额外预算；示例 `configs/examples/psd-perception-length-recovery-v1.json` 同时启用引用隔离及已有 answer-only。纠错占用一个extra，Slot补问和thinking fallback使用同一计数，独立生成和FIT均受计划上限约束。有限次额外请求耗尽时保留失败，不循环。示例是配置片段，需由主线程按既有物料创建新的独立run和release，不热改生产。

两条历史样本是在distinction附近出现大段空白且未闭合JSON；schema存在distinction长度约束、原提示已有简短JSON要求。该证据不足以确定grammar、模型或其他组件导致空白退化；新指令是已登记的纠错策略，并非根因定论。指令要求依据原音频与已供给知识，输出完整紧凑JSON、短distinction和所有必需字段。原样本response与wire reason保留；追加contact有独立request、response、FIT、COST及parent trace。正式bundle重算核对父响应hash和length、两个请求的原始音频、供给、schema与解码一致，仅允许登记的诊断后缀，以及child身份与parent关系。UNKNOWN不触发自动重发；恢复响应再次length保持终态失败。跨attempt恢复本策略尚未支持，入口明确拒绝；已有Slot空字段/缺候选可用同attempt既有补问，预算不足或补问失败仍为失败。

已知范围：仅处理非空P length；P schema/必填字段错误、空非thinking length、Slot length、语义错误、漏候选、服务故障和UNKNOWN仍未解决。若有历史thinking fallback，既有机制和费用不变；不能由结构成功推断语义正确。100%须后续在真实完整Fold A1080验收全部legal字段、官方评分、失败类别和总成本，并与已封存历史P→Solver结果按口径比较，PSD仍为主流程。

固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`，ssh npu2、CPU、断网、容器和源码只读、tmpfs。命令为 `/opt/saea-venv/bin/python -m pytest -q --tb=short --show-capture=no -p no:cacheprovider /src/tests/unit/test_intent_slot_runner.py /src/tests/unit/test_intent_slot_pipeline.py /src/tests/unit/test_intent_slot_workload.py /src/tests/unit/test_native_episode_source.py --basetemp=/tmp/pytest`，`PYTHONPATH=/src/src:/src/scripts/analysis`。结果 **117 passed，5 warnings，35.20秒**；新增工作负载配置检查单独 **1 passed，2.44秒**。真实SDK和正式canonical output/trace verifier，音频、网络模型和tokenizer为测试桩；没有实际模型、教师、真实FIT或业务NPU调用。覆盖首P非空length→纠错→Slot补问→Decision、五次模型及五份FIT桩、原生bundle重算、父wire篡改、纠错再次length、UNKNOWN中止和旧thinking fallback。测试未覆盖真实网络/服务及跨attempt恢复。代码与测试文件pin见配套JSON。

诊断输入沿用固定快照：1174条终态中1121结构有效、53失败（20 refs、32 length、1实体非空）；结构有效不代表legal enum和语义正确，完整总分母为两臂各1080。业务P schema拒绝与系统来源拒绝不能统称模型语义错误。推理资源、成本及真实覆盖收益本次未知，未修改队列、lease或生产release。

## 模板和失败路径补充（本轮）

纠错指令现为 `configs/probes/slurp_perception_length_repair_v1.txt`，SHA256 `e9baa301ffa49bfc067d9571e517e7228d2ea48171a9c18448184c4f79a74507`。可用 `pipeline.perception_repair_template: {relpath, sha256}` 指定位于 `configs/probes/` 的其他固定模板；旧 `compact-reask-v1` 默认使用上述 pin。注册派生与运行前均核路径和内容摘要；独立子contact的parent记录保留模板pin和内容，bundle从receipt重算并核实原请求只增加该模板，native source核实receipt与composed配置一致。更改模板须新的pin及独立身份。

补充脚本化SDK覆盖子contact length、invalid JSON和UNKNOWN：前两项核对两个contact的原始request、response和cost，并通过canonical output和原生诊断；UNKNOWN检查持久journal有已观察父响应和未知子intent，且不重发。模板篡改/路径逃逸拒绝；现有shared budget、旧thinking fallback和Slot补问回归保留。测试为模拟传输与tokenizer；新固定镜像相关集 **139 passed、5 warnings、38.45秒**。UNKNOWN没有响应或最终费用时如实保持未知，不能声称已有子response/cost。

注册重要限制：本分支早期 `d2e5b28c` 改动 `native_contact_custody_enabled`，会误禁用首次捕获，独占启动者修复已在主线 `1c8d6c7c`；本次新提交没有更改该文件。主线程整合时必须保留启动者修复，不得把 `d2e5b28c` 的旧 `False` 覆盖回去；length策略的跨attempt许可及整合后的真实注册仍须主线程核验。本次未进行真实注册、业务模型、真实FIT或部署，不能称可部署或声称覆盖率收益。

# 输出协议失败复盘与版本化修复

## 结论

输出失败至少有三种不同原因，不能统一解释为“模型不会输出 JSON”：①内容超过应用层字符上限；②合法 JSON 遗漏必需的外层字段；③生成预算耗尽，JSON 被截断。另有“协议合法但规则错误”的独立问题，必须通过规则证据与独立开发评估解决。

本次新增 `saea-reflexion-envelope-v4`，只修复 Reflexion 提示中完整输出外壳不明确的问题；旧版本、旧失败响应、已知成本和断点逻辑保持不变。**尚未在真实模型上验证 v4 的成功率，尚未部署；协议修复不能声称提高系统精度。**

## 已复核的历史证据

从本地封存 `state.tar` 只读提取学习响应、失败账本与原生响应，无新增模型调用。

| 分支 | 完成训练组/51 | 已知调用 | 输入/输出 tokens | 失败类型 |
|---|---:|---:|---:|---|
| ACE v1 | 0/51 | 1 | 2,917 / 3,963 | 初始记忆为空，仍生成不属于当前库的 signal item_id |
| ACE contract-v2 | 4/51 | 9 | 37,652 / 32,897 | 第5组 reflection 输出截断；总 completion 4,096，reasoning 3,093 |
| Reflexion v1 | 0/51 | 1 | 2,940 / 3,308 | content 299 Unicode 字符，超过200上限 |
| Reflexion compact-v2 | 16/51 | 17 | 66,306 / 53,062 | 合法 JSON 缺 schema；content 102字符 |
| Reflexion errors-v3 | 7/51 | 8 | 37,207 / 25,057 | 合法 JSON 仅返回内层 content/refs，缺 reflection；content 82字符 |

这些分支复用相同的历史312次调用（输入6,128,213、输出774,237 tokens），不能将历史成本按分支重复累加。上述5个分支新增36次调用、输入147,022、输出118,287 tokens；所有失败账本 unknown_contacts=0。完成组数不能相加充当一份完整候选。

ACE contract-v2 的原生响应明确 `finish_reason=length`，单次耗时67.74秒。服务器上报 reasoning 占当次completion的75.5%，最终内容仅余约1,003 tokens。输出重复携带长 trajectory_id，且允许多个 findings/signals，当前预算和输出体积存在冲突。单纯重发、接受截断 JSON 或增加总输出预算，都不能替代重新设计输出预算及引用方式。

## 根因与修复范围

旧 compact-v2/errors-v3 的空结果示例给出了完整外壳，非空示例却只展示 content/refs 内层对象。历史失败与该歧义一致，但尚无对照试验证明它是唯一原因。

v4 采用两个完整且一致的 JSON 示例：`{"reflection":null}`，或 `{"reflection":{"content":"...","refs":["r0"]}}`。明确外层只能有 reflection，内层只能有 content/refs，禁止单独输出内层、增加 schema、Markdown围栏或双重 JSON 编码。版本写入不可变输入和请求身份，使新协议对应新实验。

v4 保留原四条轨迹、官方反馈、literal_slot_errors、200字符上限、唯一且组内的引用校验、空反思、追加/FIFO行为、来源与全部历史成本。它明确允许在只能记忆参考答案时返回 null，但这只是提示约束，不是语义正确性保证。`semantic_support_verified` 和 `deployment_accepted` 仍为 false。

未启用推理引擎 structured decoding：共享桥目前没有这一配置接线，不能把提示中的 JSON 示例称为“引擎强制输出”。后续如引入 response_format/JSON schema，应单独锁定服务兼容性、thinking兼容性、FIT计数及请求身份，先小规模验证。

## 验证与后续步骤

已修改公开仓 study 的4个文件：

- `rails/reflexion_offline.py`：新协议、完整示例和严格外壳校验；旧提示分支保持原行为。
- `rails/reflexion_runtime.py`：显式允许新协议注册。
- `tests/unit/test_reflexion_offline.py`：新旧身份分离、证据不变、完整例子、null/引用映射/回放、错误输出不重发、旧错误继续拒绝。
- `tests/unit/test_reflexion_runtime.py`：新版本纳入真实共享Flight流程的测试替身集成测试。

主线程已在固定镜像执行 Reflexion offline/runtime 的64项回归，全部通过；新全量注册fixture补齐真实依赖文档后，另54项回归全部通过。真实模型 v4 探针尚未执行。

下一步先跑固定小规模TRAIN协议探针，分别统计完成组数、协议拒绝数、null率、截断率、prompt/completion/reasoning tokens及秒/组。不要在全量测试中学习协议或规则。历史失败不能靠补外壳转为成功；新协议必须新身份运行，历史失败成本仍保留。小规模探针通过后再构建完整候选，随后用冻结的独立开发集决定是否保留规则。

ACE紧凑协议v3现已实现：使用CPU映射的短轨迹别名、限定反思条数与每字段长度、完整输出示例；将thinking预算与最终JSON所需预算分开验证。新增短引用由CPU映射回原轨迹，历史原始响应不改写；完整示例、最多2条及字段字符限制已覆盖回归。ACE与槽位准入合计60项固定镜像测试通过。字符限额不能保证token适配，真实探针仍待验证。不能简单借用 perception thinking-off 的收益推断反思阶段也应关闭thinking。


## TP2预算对照结果

固定同一51组TRAIN来源、ACE v2契约和thinking3072。每臂一个TP2实例；只改变completion上限4096/16384，未修改模型权重。4096臂完成1/51组，3次调用，输入10373/输出11162 tokens，容器288.31秒；第2组反思`finish_reason=length`，4096 completion包含3093 reasoning，JSON截断。16384臂完成0/51组，2次调用，输入6642/输出5834 tokens，209.21秒；第1组整理`finish_reason=stop`，只使用2166 completion（1991 reasoning），完整JSON的两条delta均缺少`reason`。

原提示明确要求delta包含reason，解析器也要求该字段，现有证据不支持“解析丢字段”或“提示/解析契约不一致”。这是模型未遵循结构契约的观测；扩大预算没有自动消除它。后续保持预算分组，配置选择已实现的compact v3完整示例/短引用协议验证。两臂均已终止，无ETA，无可部署知识产物；不按一次结果宣称预算或算法优劣。后续记忆分叉会改变提示，不能把各组当成完全相同输入的因果对照。历史312次来源调用只计一次。

结构化结果见`ace-budget-diagnostic-result.json`。下一步需分别验证协议遵循、规则语义与消费收益，协议通过不能代替系统精度。

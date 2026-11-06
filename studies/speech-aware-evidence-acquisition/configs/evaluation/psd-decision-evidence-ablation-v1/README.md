# Decision conditional evidence：单节点候选

只读CPU已验，尚未stage/prepare/FIT/register/admit/dispatch。仅一个D正常instruction候选；不是移除Slot输入的架构ablation。

父：`/var/tmp/omni-psd-rails-preparation-v1/complete-coverage-on-workload-v1.json`，SHA `1c502d1d8d3db034f6dbf1bbde02630d95ae654d14a220551f08891f95c4971c`。

标准配置：`decision-coverage-on-workload.json`；模板：`configs/probes/slurp_psd_decision_conditional_evidence_v1.txt`。现有`intent_slot_workload.NodeWorkload`已支持decision_policy并映射solver，无业务源码扩展。

父没有显式D policy。本候选将隐式D baseline thinking（3072 reasoning＋1024 answer）显式化，改变模板，其他字段和两节点模板/recovery保持父字节值。raw变化只有decision_policy/name/run_id，有效策略变化仅solver instruction。

主线程审查后将reviewable-assets的文件相对study根合入；应用patch前须核完整父SHA并要求decision_policy不存在，不能将RFC6902 add覆盖已有policy。现有workload入口负责之后的stage/verify-stage/prepare/register；本目录不是注册或准入凭据。不得原地替换已运行模板或派发到现有两实例。完整部署仍需常规source/asset闭包及新身份，静态loader解析不冒充完整material/derive或FIT通过。

本候选保留原D提示逐字节前缀，附加三句明确比较操作/对象/方向、Slot条件性、保留真实支持的实体。无gold/词表/新KB、不强制首选、不增加输出字段。完整FoldA1080失败分母验收，不以20例命中率宣称独立收益。

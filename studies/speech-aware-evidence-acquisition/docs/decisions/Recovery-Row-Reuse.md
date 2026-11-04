# 全量恢复启动效率：整行复用与逐请求接续

## 已核验根因

当前A0运行旧冻结实现。NativeFitGate._reuse 对每个已知响应调用 request_fit_inputs，后者调用 preparation_context → _prepare。_prepare 每次创建adapter、load_split并计算全量样本identity。历史profile已显示这一链路是CPU热点。配置缓存微基准2.06倍仅覆盖compose，未部署到当前冻结运行，不能宣称解决全量恢复。

## 优先方案及约束

已有整行checkpoint恢复（core/flight/resume.py）可复用完整答案和成本；当前runtime构造函数明确禁止native_contact_recovery与resume_from/checkpoint_compatibility混用。直接移除禁令不安全：整行恢复与逐请求恢复可能双重记账、部分样本请求被重复发送，且旧FIT绑定runtime源码。

设计为一个有明确分区的恢复输入：先一次性验证ABORTED来源、checkpoint hash链和当前配置兼容性；完整行集合只走行复用；其余样本才进入逐请求恢复。全量任务名单不变，不能把剩余子集当新测试集。原始失败及未形成完整行的费用必须保留；已返回请求不重发、unknown阻止相应样本执行。原生请求序号/逻辑contact身份必须验证不会因跳过完整行发生冲突。

配置和实现兼容目前只接受contact-executor-width-v1，需新增真实命名且严格限界的恢复模式，不能借旧kind冒充兼容。不能伪造旧runtime SHA或热改活跃树。应在独立发布树完成实现/回归，再停当前任务、对账、封存并交接。当前任务保持运行到安全替代方案可用，避免重启后再次从头重放。

## 必要验证与验收

完整行、部分行、未开始样本并存时：完整行0新FIT/0新generation；部分行只补缺失节点；费用与原记录相等且只计一次。加入unknown、来源改写、不同配置、重复sample/contact、重放中再次崩溃等拒绝/恢复用例。比较固定真实已完成块的准备墙钟和CPU时间，验收的是端到端恢复到首个新请求的时间，不以compose微基准替代。尚未实现或执行该方案。

## 当前证据

源码：core/flight/runtime.py preparation_context/request_fit_inputs/_prepare，core/native_fit_recovery.py _reuse，core/flight/checkpoint_compatibility.py，scripts/analysis/resume_registered_workload.py。最新后台快照历史6921/13078，恢复重建3172、0新请求；不能把3172叠加到6921。该报告是工程设计，不是部署结果。

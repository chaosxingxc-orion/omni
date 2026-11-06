# PSD P length恢复：调查交接

本次尚未实现P length恢复，也未运行该功能测试；不能用于生产或声称100%覆盖。辅助引用修复及预审修正分别提交 `ab28828a` 和 `0b94479a`，固定镜像相关集114项通过。

已有两条非thinking P length样本在distinction附近产生大量空白、未形成完整JSON；schema支持maxLength，当前提示已经要求简短JSON，证据不足以确定受限解码或模型为根因。现有answer-only路径仅在thinking保留预算且空响应时触发，不能覆盖非空length。

阻塞在正式来源链尚未完成：现有NativeRecoveryCompletion和bundle校验仅允许`:answer-only`子contact、父响应为空且特定解码变换；P修复需要一个新注册策略绑定原音频/知识和改写提示，记录父length的wire终态及两次FIT与成本，保证和Slot补问共享预算，并在SDK、bundle重放和异常/UNKNOWN下同时验证。直接套用旧路径会把非空length误记成answer-only或让正式bundle拒绝。尚未实现这些必要变更，因此没有声称可运行纠错。下一步在独立提交中完成上述闭环，并以thinking关闭的脚本化length、业务字段缺失、预算耗尽和UNKNOWN测试验收；沿既有PSD P→Slot→Decision，不重跑旧P→Solver历史基线。

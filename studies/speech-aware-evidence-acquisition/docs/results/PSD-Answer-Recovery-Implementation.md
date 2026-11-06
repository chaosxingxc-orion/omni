# PSD Slot 回答恢复：当前实现与验证

## 状态与范围

共享补问预算、有效响应复用和一次已知无效Slot后继已合main `dd92c65d`。固定镜像207项相关集成测试通过；主线程独立确认测试容器exit0、28个变更文件与受测快照摘要一致。当前生产基线没有启用这项新策略，尚无真实数据覆盖率或精度收益，不能声称已达到100%有效回答。

本文原位描述当前状态；早期“fallback必须关闭”“单次后继未实现”的说明已过时，不作为当前执行限制。

## 恢复行为

Slot返回完整JSON但未覆盖原候选集合时，逐分支验证schema、槽位类型和引用。保留合法分支，只为缺失候选补问；补问保留原音频和知识供给。Decision仍从原候选集合选ID，由程序复制对应实体，不凭空填充默认意图或空槽。

`budgeted-v1` answer-only fallback与Slot补问共享额外调用预算。每个新增子contact保留独立请求、FIT、响应、费用及父子关联。读回时核对多源响应及冻结策略，不伪造模型结果。

若已知无效补问导致前驱attempt终止，可由正式recovery receipt绑定来源checkpoint，选择新attempt并通过原生加载器进入SDK。已观察P、有效Slot和无效补问复用；仅新增尚需完成的Slot和Decision。UNKNOWN不得自动重发。普通checkpoint恢复不直接跳过pending行，必须使用原生后继入口。

## 配置与预算

沿用NodeWorkload的`pipeline`配置，示例为：

- `configs/examples/psd-slot-coverage-answer-only-v1.json`：启用共享answer-only与Slot补问预算。
- `configs/examples/psd-slot-coverage-recovery-v1.json`：关闭answer-only，仅保留Slot恢复。

`coverage_recovery.strategy=complete-original-candidate-set-v1`，`max_extra_contacts`允许1至3。每条样本正常P/Slot/Decision共3次，generation上限为`3 + max_extra_contacts`，而非再次乘以旧fallback系数。1080条、额外预算3时，正常3240次、generation上限6480次；raw-audio FIT上限6480次。预算是容量，不是实际消耗，注册由`contact_budgets()`统一派生。

已有运行身份、固定release和科学配置不热改；新策略通过独立候选进入同Fold A配对实验。

## 已完成验证

固定saea镜像，CPU、网络关闭、源码只读，207项相关测试覆盖NodeWorkload/GEPA（71）、知识适配（9）、TF-GRPO适配（12）及恢复/FIT/checkpoint（115），56.33秒，5项依赖弃用警告，无业务或教师调用。

正式入口组合经过`native_recovery_receipt.prepare`、`select_recovery_cell`、`load_native_recovery`、SDK runner和`verify_native_replay`。脚本响应场景中，后继重放前驱3次调用，仅新增Slot和Decision共2次；预算耗尽与UNKNOWN场景新增0次；checkpoint或重放费用篡改被拒绝。另覆盖同attempt内answer-only与缺项补问共用预算。

这是正常路径离线验证，不是完整真实工作负载注册或服务推理验收。不会为更多边界测试阻塞现有基线启动。

## 已知限制与下一步

有费用的多代祖先仍明确拒绝；混合完成行与后继、多代FIT/费用和跨attempt answer-only组合未验证。`replayed_predecessor_cost`仅统计实际重放contact，来源绑定另存前驱attempt总用量；不能将该重放指标当成所有历史成本总和。完整未重放费用明细汇总未验证，历史记录必须保留。

恢复不保证修复坏JSON、length、语义错误或服务失败。下一步在独立版本、原Fold名单上运行，报告有效回答覆盖率、Intent、官方SLU、Strict、额外调用和时间；程序恢复收益与TFRL算法收益分别统计。

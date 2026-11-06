两份候选分别只替换 Perception 或 Slot instruction，复用已有 compact P、reference B 的完整文本再追加语义指导。它们是系统基线提示对照，不是 TFRL 算法。当前状态为 UNBOUND_NOT_READY：完整回答覆盖父基线尚未确定，未做模型实验、FIT、注册或准入。

- `p-instruction.patch.json`：对标准 `saea-intent-slot-workload-v1` 执行 RFC6902 test/replace，仅改 `/perception_policy/instruction_template`。
- `slot-instruction.patch.json`：同样仅改 `/slot_policy/instruction_template`。
- `binding.json`：记录模板及父模板 pins、追加句子和固定条件；这是待绑定的审查元数据，不能直接当 workload 加载。disabled/parent 字段不是新增运行参数。

主线程选定完整 coverage 基线后，先填写其 workload/config/protocol/coverage-result pins。每个候选各复制同一份完整父 workload，再应用对应 patch；test 失败时必须审查模板差异，不能直接改 test hash。结果对象仍由既有 `scripts/analysis/intent_slot_workload.py` 的 NodeWorkload 加载，不需要新执行框架。若父节点没有显式 policy，应先在共同基线中把既有有效 instruction 与 thinking 显式化并核对等价，再比较候选。不能把参考 OFF workload 的 thinking 粘到 ON 父基线上。

仅模板 instruction 的科学差异允许变化；标准 staging 需分配独立 run/name 身份。保持原 FoldA1080 名单和音频、最大3候选、原知识及检索 k、其余节点、恢复预算、thinking/解码和 v4g 不变。不同时测试两份候选；不与 k16 召回数量实验混合。

静态核验已检查标准 NodeWorkload 构造、1080唯一ID、目标 instruction 以外字段完全相等、模板 SHA、父模板 test 及 thinking 保留。schema 示例只保存在私有审查目录，未作为完整 coverage 基线或 READY 配置交付。完整 material/renderer/FIT/注册检查仍需绑定最终父基线后完成。

先用既有同名单机制样例及有实体/无实体正确负控作小规模对照，再完整1080配对；统计有效回答、intent/SLU/strict、正确→错误/错误→正确与原始失败/费用。Slot 同时看 normalizer 前后的实体指标，不能把减少本已被 v4g 消除的错抽当作最终收益。不得按 gold、样本ID或词表例外在线过滤。没有实际收益结论，不部署。

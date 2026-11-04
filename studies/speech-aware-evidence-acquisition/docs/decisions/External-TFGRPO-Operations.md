# 外部TF-GRPO相对反馈与经验操作

## 问题与算法边界

旧外部模板返回自然语言intervention，缺少可确定应用的经验增删改、父库身份和逐组高低轨迹引用。它不能直接接入现有原生学习bundle/发布链；把自然语言直接改成提示变异也不是TF-GRPO经验学习。

当前实现是冻结actor的融合式离线相对反馈操作提案：完整组及连续奖励进入教师，一次返回多个独立经验库候选。它保留TF-GRPO组内对比、经验提炼和更新语义，但不是现有summary→advantage→group_update→batch_update多阶段参考算法，实验必须单独标记，不能混报基线。

## 配置与执行

1. `python scripts/analysis/tfgrpo_external_compile.py context --source SOURCE.json --out CONTEXT.json` 从固定源文档提取model、父库、rollout policy/kind及连续reward模式。可显式设置max-operations、max-items、max-characters。
2. 使用既有 `external_exploration.py prepare JOB.json`，profile选择 `configs/learners/tfrl/tfgrpo.json`，context指向上述文件pin；显式选择固定TRAIN query_ids。prepare核验context与源一致，按每条真实rollout计算均值、总体标准差、中心化/标准化reward；重复预测仍占独立权重，同分组优势为零。统计只是教师输入，不是梯度优化。
3. 既有run/collect单次发送并核验原始wire。候选必须引用同组严格高低reward的真实trajectory_id，使用ADD/UPDATE/DELETE；父库、边界、重复操作、未供给轨迹、样本ID泄漏均检查。文本语义与隐含答案泄漏仍须独立审查。
4. `python scripts/analysis/tfgrpo_external_compile.py compile --spec SPEC.json --out PREVIEW.json`。SPEC schema为`saea-tfgrpo-external-compile-v1`，含prepared、collected两个path/sha256与candidate_index。编译重验来源、父库、原始响应、完整相对统计，复用现有parse_operations/apply_operations产生经验库预览。

## 安全接续与成本

预览使用独立schema `saea-tfgrpo-external-preview-v1`，明确candidate_publishable=false、native_learning_seal=false；现有原生consumer会拒绝它。不得伪造原生call/FIT或移除发布门禁。所有教师看过的组都计入learned_query_ids，即使提案未引用；同一外部请求的多个候选按request_sha256共享一次成本。编译不调用模型。

下一步仍需独立语义批准、真实源封存及外部来源到消费配置的专门适配，才能进入NPU配对。本变更不宣称已经完成部署、原生多阶段TF-GRPO复现或稳定收益。现有完整训练轨迹和未知请求不重发约束不变。

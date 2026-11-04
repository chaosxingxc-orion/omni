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

外部适配器现在可将独立审查通过的预览绑定到共享Solver消费组件，进入固定真实TRAIN开发评估。它仍不代表部署接受、原生多阶段TF-GRPO复现或稳定收益。现有完整训练轨迹和未知请求不重发约束不变。


## 外部来源审查与消费

`rails/tfgrpo_external_release.py` 是显式外部来源适配器；原生seal校验不变。新增参数 `source_kind: external-tfgrpo-v1`，强制 `candidate_release`，只允许标准经验渲染。支持既有learned/lock配对；暂不支持短ID和三臂渲染。

发布文件schema为 `saea-tfgrpo-external-release-v1`，字段如下：

- `candidate`：预览文件pin；`source_manifest`：上述编译SPEC文件pin。
- `consumer_roster`：固定 `docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json` 的仓库pin。
- `learning_dataset`：固定 `configs/dataset/slurp-speech-native-stage3-construction-r01c.json` 的仓库pin。
- `consumer_ids`：预先选择并排序的真实H2小规模评估子集，不能使用TEST；`selection_ids`：显式排除的selection集合，没有时写空数组并由审查者确认。
- `review`：独立语义审查文件pin。

所有pin为 `{"relpath":"相对路径","sha256":"文件SHA256"}`；candidate、spec、release和review相对data根，roster和dataset相对repo根。编译SPEC中的prepared/collected仍使用原始绝对path/sha256，运行容器必须挂载原始证据目录。适配器在实际消费边界重新读取原始请求、响应、来源和操作，并与候选逐项比较，零模型调用。

审查schema为 `saea-tfgrpo-external-review-v1`，必须包含 `decision: eligible-for-development-evaluation`、非空`reviewer`和`procedure`。`bindings`复制发布文件除schema/review以外所有字段；`items`逐一覆盖候选保留经验，每项必须包含 `content_sha256`（`tfgrpo_learning.sha(content)`）、`decision: supported`、非空`reason`、非空`evidence`（query_id/group_sha256列表）。这些检查证明审查与证据绑定，不自动证明语义为真；审查者需实际检查支持度、条件、反例和隐含答案泄漏。

启动配置直接对应prepare参数，例如：

```json
{
  "candidate_path": "/data/external/preview.json",
  "source_manifest": "/data/external/compile-spec.json",
  "candidate_release": "/data/external/release.json",
  "repo": "/repo", "data": "/data",
  "base_template": "probes/solver-base.txt",
  "template": "probes/tfgrpo-external-reviewed.txt",
  "consumer_ids": ["<固定H2真实样本ID>"],
  "model_id": "<冻结actor模型ID>"
}
```

执行 `python scripts/analysis/tfgrpo_external_release.py --config CONFIG.json --out PARAMS.json`，将输出参数配置给既有 `tfgrpo-solver-experience` 组件。脚本不会提交NPU任务或调用教师；参数、候选、来源和模板在FlightRuntime消费时再次检查。query、原始utterance和audio_sha256三个层面都禁止learning/consumer/selection交叉，且保持固定actor模型和精确消费者顺序。

这条路径只是经验库消费适配，未为ACE/Reflexion冒用TF-GRPO规则，也未自动批准任何真实候选。当前发布范围是固定TRAIN625及H2子集；跨数据集或祖先来源无法在该视图中核验的候选需独立扩展来源协议，不能忽略校验。

## 多阶段参考链的模型身份

融合式单次提案之外，`ExperienceLearner.learn(..., actor_model_id=...)` 现在允许既有 summary→advantage→group_update→batch_update 链显式区分冻结 actor 与外部 learner。构造器的 model_id 仍是实际调用并核验响应的教师；actor_model_id 必须与 parent 及每条 rollout 的模型身份完全一致。candidate.model_id 保持 actor，另存 actor_model_id/learner_model_id；batch 绑定两者，断点恢复不允许更换身份。

省略 actor_model_id 时保留历史原生行为及 batch 格式。此接口没有改变算法阶段、奖励、来源规则或原生发布要求，也不意味着外部参考链已真实运行。已有外部 receipt 保留 provider 身份，不将教师响应伪装成 actor 输出。后继配置入口仍需固定数据、完整证据、每阶段预算及来源适配；融合式当前请求不受此更改影响。

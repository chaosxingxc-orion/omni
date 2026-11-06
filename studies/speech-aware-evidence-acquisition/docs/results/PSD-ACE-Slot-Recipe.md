# ACE 启发的单卡知识优化配方

## 本轮交付

固定教师模板、配置化动态材料拼装脚本及使用文档已合main（`bfc04663`、`96ce68ef`）。解决的是通用ACE提案格式与现有单卡知识编辑编译器不一致的问题；不新增知识类型，不修改SDK或模型参数。

- 固定模板：`configs/prompts/learner/tfrl/ace-slot-construction-v1.md`。
- 动态拼装：`scripts/analysis/ace_slot_recipe.py --config <私有配置>`。
- 配置示例：`configs/learning/recipes/ace-slot-construction-v1.example.json`；示例仍需真实数据pin，不能直接调用教师。
- 执行说明：`docs/recipes/ace-slot-construction-v1.md`。

教师按原始定义、实际已送达的卡、真实H1轨迹及反馈分析根因、竞争解释、反例和风险，可提出多组独立候选。每个候选只改一张已有槽位卡的说明字段：extraction_cues、admissible_formats、missing_behavior、ambiguity_behavior、limitations。原槽位类型、意图关联与来源保持约束；证据不足允许无候选。提案先编译成候选快照，经实验验证后才可能用于运行知识库。

## 已验证与边界

指定固定镜像、CPU、无网络、源码只读的新增回归通过（1项，4.19秒）：实际运行拼装CLI，核对messages/dynamic/contract及文件摘要；现有编辑器接受单卡工程fixture，并拒绝同一话语不同录音的禁用重叠。主线程核交付5份文件摘要、拼装器与测试的受测快照一致。静态模板、示例与文档独立审读；没有声称执行过真实教师请求。

此前16项既有编辑器测试使用了不同镜像tag，不作为本次指定镜像验证证据。回归fixture不是业务轨迹、候选质量或实验收益。

拼装器保留外部供给的完整材料，但不能单独证明样本属于H1、轨迹真实或祖先完整；这些由既有动态材料来源链核对。H2、Fold A/B和TEST不能作为新的教师反馈。

## 下一步

等待当前基线锁定及construction真实轨迹，优先选择H1中“目标卡确已送达、原始定义可追溯”的失败与反例。沿既有外部教师、单卡compiler、PSD知识adapter及NPU评估入口执行；固定模板与动态轨迹保持分离。

这是ACE启发的反思与单卡解释优化，完整Curator迭代、合并和冲突处理尚未实现。当前真实教师调用0、NPU调用0，尚无收益。它不阻塞已具备条件的GEPA和TF-GRPO实验。

# PSD 的 TF-GRPO 节点学习与迭代

## 已实现范围

main `ca57b2c7` 接入 Slot/Decision 显式经验消费、节点专用外部教师模板与 profile，以及有来源记录的非空父经验库继承。模型参数冻结，知识库与非目标节点不变；Decision 仍只选择已有 candidate ID，实体从 Slot 复制。该实现同时保留融合式单次相对反馈提案和参考 summary→advantage→group update→batch update，两者不能混称为同一算法实现。

固定模板位于 `configs/prompts/learner/tfrl/tfgrpo-psd-{slot,decision}.md`，profile 位于 `configs/learners/tfrl/tfgrpo-psd-{slot,decision}.json`；动态输入由脚本注入来源固定的同题同策略组、奖励、原始请求/响应和父库。模板输出对应目标节点经验，不再要求先写旧 Solver 经验再人工改写。

## 配置与消费

示例集中在 `configs/evaluation/psd-tfgrpo-v1/`：learner-context、reference-overlay、descriptor-direct、parent-binding。真实路径、摘要和来源材料需按已有发布记录绑定，示例本身不是已就绪实验。

`psd_tfgrpo_adapter.py --repo ... --data ... --workload ... --descriptor ... --output ...` 验证发布来源后，生成新模板和带 `tfgrpo_experience` 的 NodeWorkload。必须使用独立配置副本与新输出路径；失败会清理本次生成文件，不覆盖旧模板。已来源验证的 PSD 候选按原文消费；旧 Solver 经验仍需显式语义映射，记录为人工派生，不能当作算法直接学习成果。

## 持续迭代与来源

首轮可用空父库。后继轮必须引用同 actor、同目标节点且已审阅的父候选；核验父库正文、累计 learned_query_ids、来源回执和模板，继承祖先费用身份。不能把任意非空字典当作父库，不能跨 Slot/Decision 偷换经验来源，不能丢弃祖先样本曝光或重复计算共享教师费用。

学习材料仍须是真实同题同策略多轨迹；两个评估臂不是一个组。重复轨迹保留权重，参考答案改写不算真实 rollout。当前构建池采集入口正在实现，不以工程测试夹具代替业务学习数据。

## 验证与限制

主线程在固定镜像、CPU、无网络、源码只读的独立集成树验证143项通过（28.57秒），研究树20,573个文件摘要与测试副本一致。四种组合（融合式/参考式×Slot/Decision）均覆盖两轮模拟传输、真实编译、审阅发布、来源重放、NodeWorkload与最终静态节点渲染；目标、父曝光、来源和响应篡改会失败。

测试中轨迹、教师响应和材料/服务绑定是fixture，不是业务推理；来源校验器与编译链未mock。尚无真实当前PSD分组学习、Fold A/B收益或生产部署证据。恢复新配置的集成也尚未完成；正在运行的实验代码release保持原版本，不因本次合入热更新。

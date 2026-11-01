# Training-free RL 原始方法核对与落地选择

## 结论

主算法应明确选择 **Training-Free Group Relative Policy Optimization（Training-Free GRPO）**，而不是把多候选推理、模板蒸馏或ACE统称为TFRL。论文为 [arXiv:2510.08191v1](https://arxiv.org/html/2510.08191v1)，2025-10-09；论文直接链接的官方代码为 [TencentCloudADP/youtu-agent training_free_GRPO](https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO)。本次已核查其main集成版本 `c2caa539f4c95ae1c39ed24dc8a99cb3651e1d5d`（2026-03-21）及MIT许可全文。

模型确实承担“自我改进”：冻结的同一模型不仅生成任务轨迹，还总结轨迹、比较奖励高低的原因、提出经验修改，并在下一轮使用更新后的经验。模型不负责凭主观判断宣布自己变强；外部任务奖励、独立开发评价和版本回滚负责检验。我们的当前BF16实验尚未跑通这个学习链，原625条K4不能叫TF-GRPO复现。

## 三种方法各自的角色

| 方法 | 原始机制 | 在本项目的角色 |
| --- | --- | --- |
| [Training-Free GRPO](https://arxiv.org/abs/2510.08191) | 同一问题多轨迹→奖励→模型生成组相对语义优势→模型增删改经验→下一批使用经验 | 主要算法基线，先忠实实现核心更新，再做RSI适配 |
| [Reflexion](https://arxiv.org/abs/2303.11366)；[官方代码](https://github.com/noahshinn/reflexion) | 根据任务反馈生成反思文本，保存在episodic memory，指导后续尝试，无权重更新 | 简单单轨迹反思对照，检验组内对比是否值得额外开销 |
| [ACE](https://arxiv.org/abs/2510.04618) | generator/reflector/curator，增量更新playbook，grow-and-refine减少上下文塌缩 | 上下文记忆组织与增量合并对照；不能仅引用ACE就宣称实现TF-GRPO |

## 论文与官方代码实际做什么

论文第2节：保持模型参数θ冻结，维护经验库E，rollout分布为πθ(output | query,E)。同一query生成G条轨迹并由reward评分；有明确赢家/输家的group才学习。**同一LLM**先分别总结每条轨迹，再比较成功和失败，产出自然语言semantic advantage。随后模型给出ADD/DELETE/MODIFY/KEEP，更新E，而不是反向传播、PPO clipping或计算梯度。所谓token prior是注入上下文的文本经验，不是可训练embedding或logit偏置。

官方实现已逐文件核对：

- `utu/practice/training_free_grpo.py`：epoch/batch循环、rollout、experience_updater、持久化、下一轮经验注入、评估。
- `utu/practice/experience_updater.py`：`_single_rollout_summary`、`_group_advantage`、`_group_update`、`_batch_update`。前四步调用LLM；程序应用结构化操作。不是人工预写规则。
- `utu/prompts/practice/experience.yaml`：上述角色的实际提示；包含学习目标与任务目标。
- `configs/practice/math_reasoning.yaml`：main样例是epochs=1、batch=50、G=5、并发128、temperature=.7；不能把样例当论文3epochs设置。

论文数学实验用100个DAPO样本、3epochs、G=5；web用100个AFM样本、G=3。它不是保证收益的定理：论文Table5中QwQ-32B的web pass@1从27.5降至25.5。原论文自生成经验对照也无收益，支持“不能用无反馈的自我总结替代奖励驱动更新”。论文成本/提升不直接外推我们的语音任务或NPU吞吐。

## 必须公开的RSI适配差异

1. **奖励与group语义。** 官方实现以 `0 < mean(reward) < 1` 筛选部分成功组，隐含binary reward。若直接把连续SLU-F1接入，全组0.5也会进入学习，违背论文无相对优势不学习的意图。SLURP应明确记录intent与官方slot sufficient counts；组比较需要真实reward差异或意图修正，平局不编造语义优势。最接近原版的主对照可用严格任务成功binary reward，连续/字段级reward作为独立适配消融。
2. **候选生成。** 标准TF-GRPO应同一配置/经验版本采样完整轨迹。现有K4共享P且改变部分策略，仅可用于离线经验候选诊断，不能冒称完整同策略rollout或独立难度重复。共享P是可单独验证的提效变体。
3. **难例机制。** 用户要求通常正确样本不学习。mixed-success组适合原算法；全失败难例不能永久丢弃，应先定位知识/检索/感知缺口并做定向探索，取得可比较反馈后再学。不要让模型凭空从全失败组“总结正确答案”。
4. **明确更新部件。** 将经验条目分型：ORG知识内容与key/value描述；SUPPLY检索key、排序/过滤；USE提示与证据消费；SELECT结果选择。模型输出受schema约束的delta而非任意代码；先固定其余部件，验证单项与交互，再接纳组合。
5. **跨样本验证。** 规则来自一折，在另一折应用；同一条查询不能消费从自身gold提取的答案式经验。通常正确样本留在代表性开发评估监测回退。开发集用于版本选择，完整test在冻结后汇报。
6. **算力与预算。** 总成本包含rollout、逐轨迹summary、group advantage、group merge、batch merge，不只统计task调用。缓存不可变输入、checkpoint每次模型响应；相同canonical输出可复用摘要但须列为改进变体而非原版。无收益group不做昂贵reflector调用。
7. **奖励外置，模型驱动改进。** 由同一冻结Qwen3-Omni模型提出自然语言经验和操作；程序负责身份、预算、输入边界和原子版本发布，CPU官方评分负责效果，主线程负责实验策略/失败归因。不能让主线程手写几条经验再称模型自驱学习。

## 当前实现差距与最小顺序

已有：task rollout、官方score、原始轨迹、持久化工程、模板distiller及ACE适配素材。缺少当前BF16下实际执行的trajectory summary→semantic advantage→LLM操作→经验版本发布→下一轮真实消费闭环。新不稳定credit审计只是候选选择改进，不应取代上述学习算子。

先绑定原论文/代码版本、奖励与group协议；复用可比的现有轨迹做首批语义优势提取，明确其共享P适配身份；使用冻结core完成反思及增量操作，做跨折消费与开发评估，再以更新经验运行下一轮。主对照保持无经验A0、TF-GRPO核心流程，模板distiller作为已有低成本对照；ACE/Reflexion随后按预算分阶段比较。先证明算法真正闭环，再声称改进算法。

## 来源凭证

本文来源均为论文/官方仓而非二手综述。私有目录保留TF-GRPO论文HTML、官方docs、上述源码和prompt、LICENSE以及逐文件SHA256清单。下载按顺序完成；没有模型调用、服务变更、运行时编辑或公开提交。

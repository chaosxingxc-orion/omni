# GEPA 节点学习材料准备结果

## 已完成

Decision 与 Slot 已分别生成真实 seed、修改合同和父提示，使用现有 `gepa_v2_node_adapter.py seed`。固定镜像、断网、源码只读，实际 Qwen3-Omni tokenizer；模型和教师调用均为0。本轮没有执行学习、候选评估或NPU推理，不构成精度收益。

来源为已提交的 construction625 OFF K2 第一成员（temperature=0.7、seed=1001），节点策略及知识沿用当前预备基线。基线选型尚未完成，因此只作为OFF条件候选；最终选ON或改变pipeline时应按真实来源重新派生，不能直接套用。

|产物|Decision|Slot|
|---|---|---|
|允许反馈名单|原H1 313|原H1 313|
|排除名单|原H2 312|原H2 312|
|名单交集|0|0|
|原指令tokens|7|9|
|完整模板tokens|75|121|
|真实学习轨迹|尚未采集|尚未采集|

这些token数仅是固定模板长度，不含知识、音频、动态上下文或模型思考，不能用于估算整条推理成本。H1名单是允许反馈的范围，不代表313条都是难例或都已提供反馈。

主线程核对11份交付文件摘要、workload pin、合同名单和拆分后的parent/context均一致。Decision成功复算字节一致；Slot成功退出。首个Decision进程曾未返回，保留这一观察限制，没有将它当作失败后重发模型请求。

## 可复用入口与下一步

固定模板采用 `configs/prompts/learner/tfrl/gepa-v2-{decision,slot}.md`；节点profile采用 `configs/learning/examples/gepa-v2-{decision,slot}-profile.json`，要求一次2至6个提案、输出预算131072。实际服务容量仍以API返回为准。固定镜像内有效CPU解释器为 `/usr/local/python3.12.13/bin/python3`；本次未修改镜像或运行源码。

真实H1轨迹到达后，脚本供给原任务定义、节点请求响应、实际知识供给、反馈及反例；gold只进入合法学习与评分侧。分别沿 `external_exploration.py prepare/run/collect` 调用本机外部教师，再用 `gepa_v2_node_adapter.py compile` 编译来源绑定候选，进入现有stage/prepare/register/admission/FIT队列。动态轨迹不存在时没有生成占位反馈或空教师任务。

Decision只修改选择指令，实体仍从既有Slot候选复制；Slot只修改抽取指令。其他节点、知识、召回和解码固定。先单轴配对，再保留正向变化；Fold A选型、Fold B冻结确认、TEST不调优。

## 产物身份

- decision seed文件SHA256：`074c300b33d12a3febaaa077bd947fc9a68c60ee92aef893bebe9819034e7f4b`；合同内容SHA256：`7821dbcc8b22c2dab2f025848ecfa4fdb4884ab180c754da2f72adc3c318d091`。
- slot seed文件SHA256：`ee9741c3265ce4d1de097321f7fe2fd5decf27910c11114f11441e8080efa766`；合同内容SHA256：`a775673243304fb4c13736903b7bfba60c3d3d778e164e671b399d23c8cbc8a7`。

真实输入、输出和完整命令已归档私有交接，不发布内部路径和样本轨迹。当前缺口是同策略真实轨迹及后续学习评估，不是继续加固通用框架。

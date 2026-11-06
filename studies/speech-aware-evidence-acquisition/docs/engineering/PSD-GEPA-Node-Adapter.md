# PSD v2 的 GEPA 节点适配

## 已实现的能力

代码 `b551793e` 已合 main。外部教师的 GEPA Slot/Decision 提案经过既有请求/响应回执校验、父候选校验、模板发布后，编译到实际 `NodeWorkload` 的 `slot_policy` 或 `decision_policy`。每个节点有独立教师模板和候选合同，动态证据仍由脚本采样注入。

源 workload 已有的 thinking 和模板可以直接使用。seed编译后与源模板逐字节一致；当前可编辑面是目标模板首句，其余协议文本保留。候选编译前后比较完整派生基线，只允许目标模板路径变化，不改变其他节点、知识库、供给策略、解码及名单。

Slot负责对全部原候选分别提取；Decision只选择已有candidate ID，实体由程序从Slot结果复制。该功能属于“激发理解能力”方向的配置入口，不等于已完成GEPA搜索、种群选择或稳定收益。

## 使用入口

`python scripts/analysis/gepa_v2_node_adapter.py seed|compile --repo STUDY --data DATA --config CONFIG --tokenizer TOKENIZER --out OUTPUT`

在独立可写配置树运行，禁止修改正在执行的发布树。示例位于`configs/learning/examples/gepa-v2-{slot,decision}-{seed,compile,profile}.json`；替换占位符并绑定真实来源。seed输出contract和parent，compile使用既有external exploration的prepared/collected回执及candidate_index。输出workload仍需正常stage、prepare、注册、准入、FIT和推理。

教师模板位于`configs/prompts/learner/tfrl/gepa-v2-{slot,decision}.md`，profile输出上限131072，是否可用仍以实际教师容量与请求结果为准。本次未调用教师或业务模型。

## 验证与限制

主线程在固定镜像、断网CPU容器、只读源码中验证合并版本：六组相关测试共68项通过，33.19秒，无跳过。此前旧solver/pair兼容回归116项通过。原perception夹具重复参数错误已在`99e78df5`修复，相关43项测试通过。

测试覆盖候选真实编译/模板渲染、已有节点策略保留、seed字节一致、跨节点合同与知识身份篡改拒绝，以及Fold A1080字段组合。向量、服务回执和外部响应使用离线fixture，未验证完整生产资产、真实FIT或模型质量。旧分支曾有69通过/2失败，其两项夹具失败在主线程集成验证中已消除；测试集合范围不同，不能只比较通过数量。

下一步先锁定PSD系统基线，再以稳定TRAIN轨迹产生候选、Fold A比较、冻结Fold B确认；提示候选收益与回答恢复收益分别统计。

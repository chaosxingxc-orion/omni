# TFRL探索队列与异步接续

更新：2026-11-03 11:46 UTC。此页记录可核验状态；任务图不等于实际入队。

## 当前执行状态

| 资源 | 最近核验状态 | 下一有效任务 | 尚待满足 |
|---|---|---|---|
| 卡0–3，TP4/C96 | 原A0 P-off全量容器仍运行；10:40:25 UTC样本快照3343/13078 | 当前完整测试继续 | 完整结果封存与评分；不以测试调参 |
| 卡4–5，TP2/C48 | ACE18/51失败已清算；65,536上下文迁移完成 | 三条独立ACE完整证据/任务/诊断候选 | 三臂51组真实包装prepare通过；完成注册/FIT与真实队列装载 |
| 卡6–7，TP2/C48 | Reflexion51/51生成封存，callback程序错误已严格清算；64k迁移准备 | 两个focus×两组反馈，4条mutation→paired16 | B64k真实身份、原生FIT、配对材料与队列装载 |

已核验多后继执行队列数仍为0，不能把planned、prepared或测试通过报成queued。当前两研究lane存在接续空档，不能称8卡满载。

## 排队与依赖规则

固定脚本configured_job_queue.py驱动固定镜像launcher。depends_on表示真实成功依赖，after_terminal用于失败清算；后继after_handoff接受已核验生成释放或对应严格清算成功。unless_generation_released防止成功后重复清算。正常生成释放后CPU评分可继续，下一NPU任务不等评分；未知请求或来源漂移仍阻止对应lane接续。

同一lane资源互斥不等于算法依赖。三构建候选只共享来源；每条消费配对只依赖自己的mutation，分支互不等待对方评分。失败候选保留全部成本，其依赖评估被阻塞；清算成功后独立任务继续。活跃队列已绑定摘要，不原位修改，通过后继队列调整优先级。

后台每600秒或终态事件监测；模型只处理需要判断的变化，避免频繁对话轮询。两线各维护多个有效任务和完整报告，不用无科学价值的重复推理凑负载。

## 本次解除的工程阻塞

1. ACE提示4条与compact上限2条矛盾：prepare拒绝错误配置，新配置2/2。
2. source-only容量低估：全51组改用真实native wrapper；新prepare三分支最大44,139/44,360/44,789，原生64k、输出16384、reserve512，静态extra reserve3072。库增长仍每次FIT。
3. 同步artifact封存误套task评分格式：引入独立可验证交接schema，核验manifest、outputs、FINALIZED attempt及phases，不生成虚假score文件。131项固定镜像测试及真实51组封存零调用验证通过。
4. 无release但已FINALIZED的Reflexion：严格清算真实51响应、零重发，生成完成与科学收益分开。
5. 服务输出上限与节点预算混淆：配对核验原生解析后的完整wire decoding，允许服务16k上限而P/S维持历史4096总输出/3072 reasoning，不改科学配置身份。

## 验收与结果

异步DAG已通过115项固定镜像测试，旧8节点仅通过只读plan，从未dispatch；其容量不合格，不能复用启动。上下文迁移40项、同步artifact/GEPA清算最终102项、native wrapper容量231项、pair预算67项分别通过；测试范围不同，不相加冒充唯一总数。新release通过131项回归及真实51组封存零调用验证；源文件未改变，未补写历史release。

下一次汇报包含各任务planned/blocked/ready/queued/running、完成样本/总量、秒/条或秒/组、卡数/实例、ETA及全部成本。详情与精度基线见[研究现状](Research-Status)。

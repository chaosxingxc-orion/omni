# 研究与运行现态

## 当前结论

本轮核验：A0完整13078已封存，SLU F1为74.2334%、意图准确率86.3894%、strict7525/13078。ACE0与旧joint均完成80/80，见[两TP4诊断](ACE-Joint-TP4-Diagnostics.md)。ACE SLU提升但strict净损1条，尚无稳定TFRL收益。Reflexion2 selection已完成80/80，ACE selection于03:04:57自动接续，03:13:58快照16/80。GEPA1于03:08:04因旧执行树遗漏aa分支修复退出，完整0/80；修复与请求清算中，未恢复第二实例有效负载。

## 实验结果与资源

|任务|已核验结果或状态|资源、效率与剩余|
|---|---|---|
|A0完整测试|13078/13078，官方评分封存完成|4卡TP4；剩余0。恢复campaign含历史重建，不能用其总时长冒充纯推理吞吐|
|TF-GRPO合同经验确认|80/80；经验/同合同无经验SLU为65.9813%/68.5772%，strict27/31；不晋升|2卡TP2、C32、64K；25.05秒/完整三臂样本；剩余0|
|GEPA候选0|80/80；child/lock SLU为64.9136%/68.9716%，strict均32；不晋升|4卡TP4、C16、48K；52.15秒/完整四臂样本；剩余0|
|Reflexion来源约束候选2|80/80；learned/lock独立评分SLU为69.1159%/69.1922%，strict均27；修4破4，不晋升|2卡TP2、C32、64K；campaign18.829秒/配对样本，240生成；输入1,583,946、输出491,991 tokens；剩余0|
|Top-K意图→Slot→Decision|旧归档含36条seal_refusal记录，完整P→Slot→Decision轨迹为0/80|旧2卡TP2任务已退出；历史响应及成本保留。恢复工作量不能按80−36计算，效率及ETA待修复核验|
|ACE候选0诊断|80/80；learned/lock SLU70.2074%/66.1634%，strict27/28，修4破5；不晋升|4卡TP4、C16、48K；20.93秒/配对条；240生成，剩余0|
|旧joint消费参考|80/80；SLU66.8266%、intent77.5%、strict31；P候选覆盖正确意图65/80|4卡TP4、C32、64K；12.47容器秒/条，160生成，剩余0；不是AA或A0 P-off|
|Reflexion2 selection确认|80/80，生成exit0；独立双臂最终评分待主线程验收|4卡TP4、C16、48K；22.35容器秒/条，240正常生成预算；后继已自动接续|

详细基线与配对指标见[A0完整测试](A0-Full-Test-Baseline.md)、[确认实验](TFRL-Confirmation-Results.md)及[Reflexion双臂结果](Reflexion-Grounding-Paired-Results.md)。已有完整结果不重复生成。

## 错误归因与证据更正

旧页面“新节点已完成36/80”把checkpoint完成标志误作完整业务轨迹。主线程重新核验归档字节哈希和36条episode：均为seal_refusal、role_visits为空。恢复需逐请求核验已知响应，保留失败成本，不能盲重发。相关节点修复由另一个负责人处理，本线程不重复开发。

增强joint与split的P输入尚不等价：split额外收到pipeline evidence inventory；不能据此隔离节点拆分收益。后续对照须锁定相同P供给、知识和解码，并审计真实请求。新基座开发集锁定后重新测完整TEST，不能继承旧基座分数。

此前“Reflexion只有lock评分”也已纠正：已有独立双臂官方评分与来源凭据，不需重新推理。本机ACE/Reflexion教师各完成一次请求，共8候选；ACE0和Reflexion2获诊断放行，其余6候选继续语义审查hold。教师仅在本机调用，远端消费离线产物，本轮不追加教师请求。

## 下一阶段与分工

1. 两TP4部署完成；第一实例Reflexion确认接续ACE，第二实例GEPA1 selection接续旧joint。逐项核真实请求和checkpoint。
2. 按Astra依赖图转换持久执行配置。没有科学依赖的后继不等待CPU评分，生成释放后接续；评分与错误分析异步。
3. Astra已交付7个新执行单元，正常生成预算2800、FIT另列2800，正在转换运行配置；当前策略TF-GRPO组采集仍缺绑定与子集入口，不冒称ready。GEPA现有P节点四臂实验不依赖native修复。
4. 汇报样本覆盖、秒/条、卡数/实例、ETA、全部调用成本以及strict修复/破坏；未核验项明确标注。候选须有稳定收益才晋升，TEST不参与调优。

工程与文档由请求模型Terra承担，消费参考已从草稿推进到实际启动；Sol负责ACE及下一轮TF-GRPO反馈证据分析，Astra负责规划评价。部署负责人已完成交付，空出的分析槽用于TF-GRPO CPU任务。暂无新增用户决策。main与真正Wiki通过已恢复的推送通道持续同步；页面状态原位更新。

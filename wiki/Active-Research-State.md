# 研究与运行现态

## 当前优先级

用户明确先锁定新P→Slot→Decision实验基线，再开展TFRL。新基座尚未锁定。旧H2调度与在途生成已停止并保留断点，GEPA旧图续推与旧K4采集不再执行；CPU清算和新图修复并行。服务保留，当前不能宣称NPU满载。详见新基座优先决策页（Wiki：New-Baseline-First）。下表历史结果不代表新图效果。

## 当前结论

本轮核验：A0完整13078已封存，SLU F1为74.2334%、意图准确率86.3894%、strict7525/13078。ACE0与旧joint均完成80/80，见[两TP4诊断](ACE-Joint-TP4-Diagnostics.md)。ACE SLU提升但strict净损1条，尚无稳定TFRL收益。Reflexion2 与 ACE0 selection 均已完成80/80，均不晋升。ACE0 SLU仅+0.4442pp、Intent−3.75pp、strict24/26；见[ACE确认结果](ACE-Selection-Results.md)。第一实例曾自动接续H2第一片，现按用户新优先级于03:51 UTC停止；检查点与未完成请求待清算。GEPA1旧树aa路由已修，但恢复被prepared源码身份绑定拒绝，保留输出repin已完成，但协议身份变动阻止旧响应接续；旧图生成现已hold，工程执行者转向新基座。

## 实验结果与资源

|任务|已核验结果或状态|资源、效率与剩余|
|---|---|---|
|A0完整测试|13078/13078，官方评分封存完成|4卡TP4；剩余0。恢复campaign含历史重建，不能用其总时长冒充纯推理吞吐|
|TF-GRPO合同经验确认|80/80；经验/同合同无经验SLU为65.9813%/68.5772%，strict27/31；不晋升|2卡TP2、C32、64K；25.05秒/完整三臂样本；剩余0|
|GEPA候选0|80/80；child/lock SLU为64.9136%/68.9716%，strict均32；不晋升|4卡TP4、C16、48K；52.15秒/完整四臂样本；剩余0|
|Reflexion来源约束候选2|80/80；learned/lock独立评分SLU为69.1159%/69.1922%，strict均27；修4破4，不晋升|2卡TP2、C32、64K；campaign18.829秒/配对样本，240生成；输入1,583,946、输出491,991 tokens；剩余0|
|Top-K意图→Slot→Decision|旧归档含36条seal_refusal记录，完整P→Slot→Decision轨迹为0/80|旧2卡TP2任务已退出；历史响应及成本保留。恢复工作量不能按80−36计算，效率及ETA待修复核验|
|ACE候选0诊断|80/80；learned/lock SLU70.2074%/66.1634%，strict27/28，修4破5；不晋升|4卡TP4、C16、48K；20.93秒/配对条；240生成，剩余0|
|ACE0 selection确认|80/80，SLU65.6352%/65.1911%，intent72.5%/76.25%，strict24/26；修1破3，不晋升|4卡TP4、C16、48K；21.70容器秒/配对条，240生成，剩余0|
|旧joint消费参考|80/80；SLU66.8266%、intent77.5%、strict31；P候选覆盖正确意图65/80|4卡TP4、C32、64K；12.47容器秒/条，160生成，剩余0；不是AA或A0 P-off|
|Reflexion2 selection确认|80/80，SLU learned/lock69.0315%/67.7252%，strict31/30修3破2；intent75%/76.25%，不晋升|4卡TP4、C16、48K；22.35容器秒/条，240正常生成预算；后继已自动接续|

详细基线与配对指标见[A0完整测试](A0-Full-Test-Baseline.md)、[确认实验](TFRL-Confirmation-Results.md)及[Reflexion双臂结果](Reflexion-Grounding-Paired-Results.md)。已有完整结果不重复生成。

## 错误归因与证据更正

旧页面“新节点已完成36/80”把checkpoint完成标志误作完整业务轨迹。主线程重新核验归档字节哈希和36条episode：均为seal_refusal、role_visits为空。恢复需逐请求核验已知响应，保留失败成本，不能盲重发。节点恢复仍需补齐，已要求当前工程执行者核对可复用的通用修复与split准备状态，未重复启动新实验。

增强joint与split的P输入尚不等价：split额外收到pipeline evidence inventory；不能据此隔离节点拆分收益。后续对照须锁定相同P供给、知识和解码，并审计真实请求。新基座开发集锁定后重新测完整TEST，不能继承旧基座分数。

此前“Reflexion只有lock评分”也已纠正：已有独立双臂官方评分与来源凭据，不需重新推理。本机ACE/Reflexion教师各完成一次请求，共8候选；ACE0和Reflexion2获诊断放行，其余6候选继续语义审查hold。教师仅在本机调用，远端消费离线产物，本轮不追加教师请求。

## 下一阶段与分工

1. 修复新图首次跨节点重复供给，完成匹配P输入、正式功能与恢复测试；沿原固定DEV80启动真实P→Slot→Decision验证。
2. 第一实例旧任务做终态/未知请求清算并保留全部成本，第二实例只运行新基座验证。旧TFRL候选和旧joint K4保持hold，不为占卡继续旧方向。
3. 核验80条节点轨迹、失败分母、官方指标、节点耗时和配置来源，冻结一个实验baseline digest。当前尚未锁定。
4. 锁定之后，再从同一基线开展知识构建、P、Slot、Decision的单项TFRL与组合回归；完整TEST仅最终报告，可与开发队列分实例异步运行。

当前三个Codex任务分别负责新图工程与第二实例DEV80、旧任务清算与冻结契约、历史节点归因。通过文件交付，主线程验收并及时同步main与Wiki。暂无新增用户决策。

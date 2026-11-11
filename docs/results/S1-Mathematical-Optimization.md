# S1数学优化合同：固定模板＋本轮实测反馈

## 任务函数与可优化变量

令音频为x_i，参考类型集合Y_i⊆T，|T|=55，Fold为F_1…F_10，互不重叠且并集T。

K=锁定来源的知识内容与边界；R=示例/规则检索和消费策略；p=任务提示词；q=输出协议；θ=固定模型参数。输入z_ij=C(x_i,K,R,F_j)，输出Z_i=∪_j Rails_q(M_θ(x_i,z_ij;p,q))。

优化ω=(K,R,p,q)，θ不更新。语音内容辨认和语义角色理解是M内部潜过程，不可直接由最终标签观测；不得把ΔR直接命名为“理解能力提升”。没有attention隔离时“逐类型输出”也不等于各类型概率条件独立。

## 完整分母与互补目标

y_it=1[t∈Y_i]，z_it=1[t∈Z_i]，a_i=1[所有Fold完整且协议有效]。
TP=Σ_i,t y_it z_it；FN=Σ_i,t y_it(1−z_it)；FP=Σ_i,t(1−y_it)z_it。
R=TP/(TP+FN)，P=TP/(TP+FP)，F1=2PR/(P+R)。无预测时P未定义，不填成1。
C=Σ_{i:|Y_i|>0} a_i·1[Y_i⊆Z_i]/N_+。
B=FP/N，E0=Σ_{i:|Y_i|=0}a_i·1[Z_i=∅]/N_0，X=Σ_i a_i·1[Z_i=Y_i]/N。
失败仍保完整分母；可用部分类型计observed PR，但失败不能挣C/E0/X正确分。
槽值未知不改变y_it：本阶段不预测值，不用值缺失将类型置0。

Fold j 的g_ij=1[Y_i∩F_j≠∅]；误空率FE=Σg_ij·a_ij·1[Z_ij=∅]/Σg_ij；负Fold误非空FNNE=Σ(1−g_ij)·1[Z_ij≠∅]/Σ(1−g_ij)。另列失败，不能算正确空。

## 配对修复/退化：每次策略必须报告

设对照b、候选c：r_T=Σy(1−z_b)z_c，l_T=Σyz_b(1−z_c)，a_F=Σ(1−y)(1−z_b)z_c，d_F=Σ(1−y)z_b(1−z_c)。
ΔTP=r_T−l_T；ΔFP=a_F−d_F。覆盖ΔC=(修复覆盖样本数−退化覆盖样本数)/N_+。
当前grounding→supply：r_T=547，l_T=29，a_F=12301，d_F=3890；ΔTP=518，ΔFP=8411；ΔC=(516−26)/4501=10.8865pp。
净新增FP/净新增TP=8411/518≈16.24。它是此次组合干预的代价描述，不是未来边际导数，也不是16.24个已审定语义错误。
C:85.63→96.51%，R:89.46→97.50%，P:23.42→18.73%，B:2.82→4.08。下一步希望d_F>a_F且r_T≥l_T，至少满足预注册非劣界，不靠丢gold换表面精度。

## 各策略修改项和影响面

U：Δp≠0；ΔK=ΔR=Δq=Δθ=0。语义角色判断假说，预计影响全样本所有Fold。目标减少a_F、增加d_F，同时限制l_T；不能预设只影响FP。已有grounding反例提醒隐含角色可能漏标。
E：仅R.max_examples:4→1；其他ω不变。静态fallback使n_total=n_dynamic+n_static，Δn_total不等于−3，字节/类型支持集合也要量化。潜在d_F增加同时l_T增加；按原动态支持程度分层报告。
K：ΔK为来源规则的target_types投影，其他ω固定（缓存id的合法重绑定只属工程来源，不改向量/排序）。影响集合I_K={i,j:候选输入与对照不同}，再列来源屏蔽和预算原因。目标类型族FP下降不等于其他类型无风险。
B布尔：Δq及格式所必需Δp，知识/消费/θ不变；每Fold仍一次调用，成本可能上升。科学parent=None，与U/E/K的supply parent分开；不可把不同parent的差异归给单一机制。

实际影响率ρ=|I|/(N·10)，输入预算L_in、输出L_out、吞吐τ=wall/N、卡时H=Σcards·wall/3600。请求延迟与并行吞吐不能相加混称。

## 教师生成候选的必填数学字段

每候选写：changed_variables/frozen_variables、mechanism_hypothesis、primary_metric、expected_metric_signs、affected_input_scope、risk_metrics、predicted_repair_types、possible_regression_types、falsification、comparison_parent、budget_change。预测只写方向/带理由区间，不编造精确提升率。I_K等必须以实际输入核验。

教师用What/Why/How连接公式和案例：What说明哪项损失高；Why用请求供给/参考/响应支持机制及反例；How说明Δω如何可能改变(r_T,l_T,a_F,d_F)，如何用同parent对照证伪。禁止只写“提高F1”。公式是评估/策略合同，不把当前query的y或gold发给学生。

## 多目标TFRL更新

每轮由ACE式记忆修订/组相对策略搜索提出ω_c，用相同TRAIN评测得到v_c=(C,R,P,−B,E0,X,−H)。分维优势A_ck=(v_ck−mean_c v_ck)/(std_c v_ck+ε)，零方差标无信息，未知维度null；保对照在组内。
先按SELECTION-PROTOCOL的覆盖/召回非劣与P/B约束形成可行集，再比较真实收益与Pareto，不用加权平均掩盖C/R回退。具体边界已事前记录；DEV验证固定候选，TEST不调优。
策略更新ω_{r+1}=Select({ω_r}∪可行候选；实测修退、置信区间、成本)，无可行改进则保留ω_r，并将反证反馈教师，不宣称训练成功。此为参数冻结的策略/外部记忆学习，不是参数GRPO。

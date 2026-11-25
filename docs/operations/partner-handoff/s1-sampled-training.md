# S1 训练录音与 fold 筛选

固定镜像验证：官方 TRAIN 119,881 个录音引用里，来源和标签可用 106,646 条，静态页每满页 128 条，共 834 页；与 KBv2 的 106,018 个 top5 命中取交集剩 94,533 条。沿用已有特征和 S1 排名。未重新编码。训练准入不用于缩小 DEV/TEST 人口。

十个 fold 共有 945,330 个候选（录音，fold），67,376 个含该 fold 的金标类型，其余 877,954 个为空。另有 4 条录音带不在词表的 query_detail，单列，不算模型漏报。原 S1 scorer 对实际供给的类型评分；空 fold 也可能误报。

prepare_s1_recordings.py 配合 S1-RECORDINGS.template.json 和 --profile 生成 PAGES.json。sample_s1_folds.py 配合 S1-FOLD-PANEL.template.json 和 --profile 生成 PANEL.json。先随机分配一个 fold，再按真实/合成、fold 和该 fold 是否有金标分层无放回抽样。首个筛选面板为 480 个不同录音，40 层各 12 条，保留条件概率。此平衡样本的错误率不可直接当总体错误率。

render_s1_panel.py --panel PANEL.json --output inputs 用原 S1 source policy 和 StageRails 按原音频渲染选中 fold。offline supervision 单列。固定镜像已读完全部 480 份输入。audit_s1_screen_history.py 对一份封闭 TRAIN 采集里的 1,810 个历史 S1 回复检查，与面板的录音/fold 重合为零。下一步在冻结 S1 上获得真实回复后，区分漏报、误报、两者兼有、协议失败、正确对照；未知不计错例。全 DEV 保留全部 fold。

compile_training_scope.py 配合 scope 模板和 --profile 从同一份 learning 和实验几何编译 P/S1/S2/D/system。32 条录音 × G4 的单 fold S1 为 128 次请求。系统配置保留全部 10 个 fold、实际 S2 分支及 D。系统 Clock 缺 full-DEV-after-D 时拒绝训练组。CPU 范围编译、S1 真实输入制备已通过；五种目标通过同一正式启动器到达 actor 尚未验收。S2 需要版本化 S1 供给，D 需要优化后 P/S1/S2 的新轨迹，系统需要组合策略与新 TRAIN 难例。阶段配置不是策略部署或训练结果。

480 条录音经过正式 release 的 candidate 和 labels_for 生成 1,920 个互异的请求身份，同组 G4 的 fold、原始消息和供给类型一致，未请求模型。对 S1 的预算按录音、G 和实际供给的 fold 计。

在观察模型结果前，用 S1-PARTITION.template.json 按来源组件加 seed 切分。实际筛样侧 388 条（379 个组件），独立 TRAIN 检查侧 92 条（91 个组件），组件交集为 0。全 DEV/TEST 不使用该筛选。s1_panel_selection.py score 按版本读取已封闭的原始调用，核对 wire、原始响应字节、intent、模型和来源，并复用原始 StageRails 解析器。逐条分为漏报、误报、混合、正确非空、正确空、协议异常；无对应回执单列。S1-OUTCOMES.template.json 可配置训练 fold 和各来源/fold 的结果配额；面板未收齐时不输出候选 roster。配额只控制筛样分布，不作总体准确率；只在筛样侧选 fold 或调配比，检查侧用于冻结方案的配对验证，不回流训练。

历史集合 live02 的原始请求账本含 1,890 个 S1 调用（1,810 指另一份轨迹导出中的阶段输出），仍无新面板同录音/fold 的回复。独立历史 live01 的 10 条原始回复与官方 TRAIN 标签回放得到 6 正确空、2 正确非空、2 误报，10 个篡改 fold 的负例全部拒绝。单 fold 消费分支已经接入原并发派发和原解析器；真实 S1 对齐、当前冻结系统的 480 条面板推理、正式池封闭及 actor 消费尚待验收。约束：不可用当前 P adapter 的输出代替冻结 S1，不把 G4 组装或历史解析回放当作 NPU 同动作验收。

进一步用 live02 有已知回执的 1,810 次 S1 调用与官方 TRAIN 标注回放：仅误报 396、仅漏报 13、混合 2、正确非空 157、正确空 1,242。1,810 次改错 fold 的负例全部拒绝。历史语义错例中仅误报占 396/411；这个分布支持将误报和漏报分开采样，不能代替当前 480 条的观测，也不能直接用为总体权重。

正式入口新增 formal-target-start-v1；target、已登记的 baseline、实际 branch/Clock、TRAIN sampling 必须一致，配置传入原编译器。真实 P 站点的 --plan 通过；误选 P baseline 和把 P baseline 改名为 S1 的两个反例都在部署前拒绝。S1/S2/D/system 的实际 Clock 通过阶段匹配检查。站点目前仍只有 P 具备正式登记，不能把其 checkpoint、Adam 或 roster 交给 S1。

独立节点用 BASE-ADAPTER.template.json 和 prepare_base_adapter.py 从原始模型 config、权重索引和训练工程的 PEFT 结构创建初始 LoRA。固定训练镜像、CPU、无 NPU：S1 rank=256、alpha=512、dropout=0.1；48 层、192 个投影、384 个 BF16 张量，原训练 reader 读回与文件逐元素差为零，192 个 B 矩阵全零。未读 P 权重，未加载 30B 底座，未继承优化器。这是权重装载检查，不是同动作训推概率验收。

用 measure_panel_prompts.py 和 MEASURE-S1-PROMPTS.template.json 在同一训练镜像运行原 processor 对全部 480 个请求计数：最短 1,686，中位 2,210，P95 2,509，最长 2,557；加最大输出 256 后最长 2,813。当前 3,072 的 S1 输入限额及 8,192 的上下文均容纳本面板，超限 0。按实际请求内容保存长度，不能据此推断其它录音或 D 的长度。后续仍需独立分支发布和受 guard 的 NPU 同动作验收、冻结策略面板回复和正式池封闭。

checkpoint 选择已独立于训练目标。固定训练镜像 CPU 用同一个 S1 目标分别读取基座 adapter 和一份真实 P step 80 的 adapter，每份都通过训练 actor 初始化、step 0 游标和推理包重定位后的原始 policy reader，权重摘要逐份一致。6 个错误配置（恢复与初始化混用、摘要漂移、作业身份不符）全部拒绝。step 80 是读取器测试输入，没有作为正式 S1 策略部署。当前冻结面板仍为 0/480；未新开 NPU 作业。

候选分母：官方 TRAIN 119,881 条录音，来源 allowed 106,646、pending 966、excluded 12,269；allowed 均有可读标签。当前训练协议再要求真实 P 前动态召回包含金标，两个冻结名单合计 106,018。按录音 ID 取交集后剩 94,533（真实 42,507、合成 52,026），有 12,113 条来源允许的录音仅因不在 P 准入名单而未入此次面板候选。该限制来自当前训练协议，不是 S1 类型预测本身的要求。公开报告不得将这套口径表述为已对 94,533 条运行 S1。

prepare_s1_training_inputs.py 读取 S1-TRAIN-INPUTS.template.json，固定镜像已为全部 480 条准备训练 reader 所需的原始请求音频、官方标签和录音来源，未调用模型。筛样侧 388 条（该 fold 有金标 188、无金标 200），检查侧 92 条（52、40）。seal_s1_training_roster.py 读取 S1-TRAIN-ROSTER.template.json；当前 0/480 明确拒绝封存名单。10 条真实历史回复在同一条记录拼装和 fresh_material.node_screen_packet 路径回放通过，10 个错 fold 拒绝；历史回放不计当前面板结果，不代替正式训练。建议当前冻结策略先对锁定面板每条一个 fold 实测，再按层缺额扩大独立录音的样本。不以 10 个 fold 的笛卡尔积作为首轮采样要求。

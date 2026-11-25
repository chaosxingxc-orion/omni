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

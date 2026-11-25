# 训练配置自检（2026-11-25）

固定入口按目标读取实验 JSON，站点 profile 只提供主机、路径、版本摘要及现场回执。S1 的数据阶段和启动请求共用 `S1-EXPERIMENT.template.json`：各命令传同一配置，只解析自己需要的阶段。按阶段就绪的文件可以在后续绑定；改已封存的抽样条件必须开新输出目录。已有单阶段文件仍可读取。训练启动需要已登记的同目标基线。

| 调整项 | 配置及最终消费者 | 已核对的行为 |
|---|---|---|
| 抽样 seed、fold、每层配额、检查集占比、录音页大小、邻居深度 | parameters → 录音准备、分 fold 采样、按结果封存 | 固定镜像，同一批 94,533 候选从 480 条切到单 fold 10 条；再换 seed 后名单变化 |
| P 的错对配比、训练和 DEV 分母 | 已封存的 sampling、pool、原请求绑定的数据集 → 控制器、窗口、DEV 验证 | 正式窗口不再覆盖错对比，S1 不套 P 比例。面板在窗口记录自己的范围；正式 DEV 仍要求配置所指的完整官方数据集 |
| 训练与推理协同 | 启动配置 experiment → rank profile、graph、policy、worker | CPU 配置变体：rank 128，推理 max rank 128；每步 16 条录音，推理图上限 64。已编译正式基座包，未在 NPU 运行变体 |
| 精度容差 | same_action_max_abs → 启动预检和发布门限 | 1e-8 边界通过，超界、NaN 拒绝；历史无此字段仍按精确相等读取。publication_norm_atol 是独立的发布权重范数门限 |
| S1 解析器与缓存 | 发布配置中的两份解析器文件及摘要 → S1 capture；prepared_cache_bytes → 正式 rollout/处理器缓存 | CPU 编译实际发布包，读回两个原始模块并装载 S1 解析器。旧快照兼容值不作为新配置的参数源 |

验证入口：`check_experiment_injection.py` 按 `EXPERIMENT-CONFIG-AUDIT.json` 的 20 个文件检查已发现的常量回归，并执行真实采样变体；这个检查不等于覆盖所有硬编码。另用 `check_s1_controller_window.py --branch ...` 读实际独立 S1 Clock 和历史原始回包，10 条筛样计 21,646 tokens，9 条入选、三个窗口。正式面板 0/480，当前 P 作业未切换。

系统池与知识供给的补充核查：

- 旧系统池将当前 TRAIN 数量和错误/正确 80/20 写在验收器里。新发布器读取有摘要的 TRAIN manifest，核对两个域、元数据、实际记录数和完整性；抽样比例由 sampling.error_success 指定。当前清单为真实 50,628、合成 69,253，合计 119,881。用隔离的 16 条测试池分别配置 25/75 和 75/25，成品包的实际发布器和消费者每次读 8 条，分别为 2 错 6 对、6 错 2 对。比例未归一、分母错误、元数据引用冲突、局部 manifest 或不完整池均拒绝。测试池只证明配置传递，不是生产数据。
- P 的常设卡从知识策略的 `policy.permanent_fixed_only` 读取；材料库、字段修复器和请求渲染器必须认同同一名单与顺序。当前是 general_quirky 和 qa_factoid，两张。用真实封存供给调用 CPU 渲染器，新包和冻结版的完整输出一致（wire SHA-256：32eaa62cd80047b45936081222fdf72195240e6315c4b9167417c4467b223f68）。漏卡、顺序交换、重复卡均在启动时拒绝。要改变常设卡集合，需配置匹配的 renderer、seed 和知识策略版本；不能据此宣称旧渲染器支持任意名单。每条录音的动态卡、示例和召回指令照常独立供给。
- 一个站点 profile 中，录音页消费的召回清单引用与筛样消费的结果摘要现在可以同时提供：`RECALL` 传引用，`RECALL_RESULTS_SHA256` 传摘要；旧面板专用 profile 仍可用。数据集阶段与训练启动共用同一份实验模板。
- 正式构建器使用原精度锁，在固定 CPU 镜像锁定、构建并核对 126 个安装目标。跨模块源码只允许定点名单；从装好的包再次执行上述采样和渲染。运行中的 P 作业未切换版本。

未通过或尚未覆盖：

- 历史 live01 合并重放通过了全量原始清单与元数据核对，在核对历史作业的源码摘要时被拒绝（ACTUAL_RUN_CODE_DRIFT）。举例：unified-learning 的 dataset、scoring、test_unified 已不是该作业当时的内容。未更改旧凭据或放宽校验；恢复须提供匹配原摘要的封存代码。旧数据未被晋级。
- 正式 S1 面板为 0/480。仍需接通绑定实际冻结策略、服务实例和请求的采集回执，随后封存实测名单、验收 NPU 同动作及首次更新。S2、D、system 未做真实端到端训练。TP4/FSDP8 等已验收能力也不会随配置修改而自动扩展。

复验入口：`check_experiment_injection.py`（定点源码及真实候选变体）、`check_s1_controller_window.py`（历史窗口）、`collection-dispatch-integration/check_pool_config.py`（比例及消费者）、`p-card-field-repair/check_card_policy.py`（请求与错误知识名单）。均不需要发起模型请求。

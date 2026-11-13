# 反复实验修复的根因与整改边界

结论：这些反复失败主要来自执行与部署契约分散，而不是已有证据证明必须替换推理核心。实验专用启动器不断复制并叠加恢复补丁，导致配置路径、源码、控制器和评分来源不再共享同一事实来源。测试多数在局部 fixture 上通过，未覆盖实际的新解释器、容器挂载与阶段交接。

以下为独立审查的 12 类机制。E1–E5 是内部有限证据编号，分别对应本地实现验收、来源接口缺口、真实部署部分验收、评分观察者断网失败、完整 D 终态与 S2 bind 失败；原始身份与轨迹不公开。相对路径是历史交付中的证据定位，不是推荐运行入口。

## Why / What / How：12 个独立故障机制

### F01：准备命令复用了错误的网络能力

- **Why／根因：** `s2-reliable-shared-runtime/host.py:47` 的 `create_argv` 只有 run 使用 host 网络，其余模式均使用 none；`:76` 的 bind 却复用 load 参数，而实际 bind 会调用 live 检查读取服务 socket 身份。
- **触发与证据：** E5 明确记录该因果链和已知失败退出；不是 SSH 或模型质量问题。
- **漏测：** 仅验证静态 load 或 FIT 成功，没有在实际 bind 命令的网络命名空间执行 live gate。
- **What／已有修复：** `s2-reliable-c175-successor/bind-network-repair/v4/host.py:15` 将 bind 显式纳入 host 网络；保留原失败 FIT 的来源，不假造请求历史。E3 只部分接受真实启动。
- **How／长期约束：** 用“需访问本机服务身份”的能力决定网络模式；纯 CPU 评分继续 none。固定镜像验收须检查实际 create argv、inspect NetworkMode 和 live gate，且在新增模型调用前完成。

### F02：把容器 hostname 错当成 CID 前缀

- **Why／根因：** `d-reliable-candidate-operations/ops.py:101` 用 `actual['Id'].startswith(socket.gethostname())` 证明身份；`:222` 的 host 网络 Docker 命令没有显式 hostname。该等式不受 host 网络容器保证。
- **触发与证据：** E2、E3 确认现场观察与此假设冲突；对应 `recovered-source-layout/MAIN-REVIEW.md` 记录未来多个 D 入口都会受影响。
- **漏测：** fixture 使用恰好像 CID 的 hostname，或者只在当前 Python 解释器覆盖校验函数。
- **What／已有修复：** `recovery-source-adapter/operations_patch.py:115`、`:191` 显式设置唯一 name/hostname，并校验完整 CID、inspect Name/Hostname 与容器内 hostname；E1 接受该本地实现。
- **How／长期约束：** 名称负责唯一命名，完整 CID 负责实例身份，二者分别校验。验收须覆盖 system、control、mention 的新解释器与实际 Docker inspect。

### F03：控制器类型与历史状态 schema 被假定为统一

- **Why／根因：** `d-reliable-candidate-operations/ops.py:81` 强制查 systemd unit，实际前驱来自 detached Popen。原 `s2-reliable-shared-runtime/host.py:59` 的 START 仅含 pid/at，`:68` 的 EXIT 仅含 exit_code/at；恢复启动记录同样没有历史 start_ticks/boot 快照。把新 schema 强加给旧证据会永久拒绝合法接续；用不存在 unit 的 inactive 状态放行又会误判。
- **触发与证据：** E2；`recovered-source-layout/rework/MAIN-REVIEW.md` 明确指出早期适配对真实字段的错误要求。
- **漏测：** 自造“字段齐全”的新 schema fixture，未使用真实旧 START/EXIT 结构；仅看 PID 是否存在，未区分 PID 重用。
- **What／已有修复：** `recovery-source-adapter/operations_patch.py:34` 起区分 controller kind、startup_snapshot 和 terminal_reconcile。旧记录通过真实 EXIT、同目录 CONTAINER/FINAL、当前 inspect、进程消失或有证据的 PID 回收、unknown=0、原 guard 与释放凭证接续；不补写虚构启动身份。
- **How／长期约束：** systemd 须验证 loaded unit；Popen 不查不存在的 unit。启动时记录 PID/start_ticks/boot/CID 关联；缺历史字段的兼容路径须显式声明。SSH 断开、unit failed、PID 消失均不能单独证明业务终态。E5 还显示业务全分母完成后停止客户端可产生非零进程退出，业务终态与控制器退出须分开表达。

### F04：逻辑实验身份绑死在一次恢复的物理目录上

- **Why／根因：** `d-system-production-entry/production.py:42`、`:43`、`:51`、`:57`、`:73` 固定绑定旧 runtime/controller/archive 布局；只改 producer 的路径仍会被评分端遗漏击穿。`score_cpu.py:67` 固定 objects，`:82` 固定 run/rows。
- **触发与证据：** E2；关联 `rework/MAIN-REVIEW.md` 明确定位评分端这两处遗漏。真实当次 run/archive 名称未变，掩盖了通用性缺陷。
- **漏测：** 只测试 recovery resolver 或当前恢复目录，没有真正调用展开后的 `score_archive` 和 `baseline.entry`。
- **What／已有修复：** `recovery-source-adapter/source_adapter.py:26` 将完整 source_paths、evidence_roles、expected_identity 置于配置；`load.py:59` 至 `:67` 补全评分物理映射；`check_score_paths.py` 使用实际函数 AST 验证重命名路径。
- **How／长期约束：** 从 intake、attach、configure、feed、runtime 到 scorer 共用一个冻结来源见证。物理键可映射，官方评分接口的 canonical refs 键保持原义；不能通过重命名科学来源掩盖部署变化。

### F05：部署闭包依赖大量历史目录，摘要通过未证明可部署

- **Why／根因：** `d-system-production-entry/host.py:24` 创建的 accepted 是指向父 code/accepted 的 symlink，`:27` 继承父 mount 列表；源码又通过 `load.source` 和多个固定父根加载。依赖闭包包含旧代码、配置、来源和挂载别名，而非只有新目录。
- **触发与证据：** E1 的 187 个 mount、170 个本地不可观测依赖，以及固定镜像未验证，是明确的依赖和验收缺口；不能据此断言 187 个 mount 已全部实际失败。关联 MAIN-REVIEW 曾纠正“本机没有 accepted 就判部署缺陷”的错误诊断。
- **漏测：** 将 pins 无差异等同于容器可达；或者只检查 mount target 无重复，未验证目录类型、嵌套覆盖、symlink 终点和实际读权限。
- **What／已有修复：** `recovery-source-adapter/host.py` 增加只读父根，`source_adapter.merge_mounts` 拒绝重复/冲突 target；E1 接受本地实现，仍未证实真实固定镜像闭包。
- **How／长期约束：** 生成可复核的最小依赖清单：每项记录角色、source/target、ro/rw、文件/目录、symlink 终点及 hash。先证明原闭包可达，再在独立后继树去除无用历史依赖，禁止删除活动依赖来“精简”。项目 cwd、tmp/cache/home、日志、attempt/checkpoint/queue 和容器可写项目路径都须位于获准共享根；共享盘故障不得落本地。Docker 项目日志/可写层单独审计，不迁移全机 Docker 根。

### F06：把运行输出 confinement 误用于跨根只读源码

- **Why／根因：** `source_adapter.confined` 会 resolve symlink 后要求仍在运行根内；实际复用源码通过跨根只读 workspace symlink 提供。文件 hash 正确也不能证明 resolver 或容器 mount 可达；反过来全局放松为整个共享盘会削弱运行输出约束。
- **触发与证据：** 关联 `rework/MAIN-REVIEW.md` 记录真实 resolver 18 项中 17 项摘要一致、1 项 FIT 权限不可读；E1 明确将 supplemental source roots 纳入接受范围并保留 FIT 权限 pending。
- **漏测：** 只做 lexical prefix 或只从宿主直接 hash 文件；未执行真实 resolver、physical scorer 映射和容器权限读取。
- **What／已有修复：** `source_adapter.py:62`、`:76`、`:94` 用显式 origin_roots、supplemental_roots、alias 和只读 mount 区分源码 namespace；运行结果 custody 仍严格 confined。
- **How／长期约束：** 仅允许配置列出的只读源码根，禁止递归 alias、越界 symlink 和输出逃逸；FIT 应由实际运行权限在固定镜像读取，不 chmod 活资产、不复制整棵 workspace。

### F07：统一 pin 接口混淆字节校验与 JSON 反序列化

- **Why／根因：** 恢复 source 和 implementation_pins 包含 `.py` 原字节，早期适配却对每个 pinned 引用执行 json.loads。
- **触发与证据：** `recovered-source-layout/MAIN-REVIEW.md` 对初稿指出必然 JSONDecodeError；现 `source_adapter.py:145` 至 `:155` 已分开保存 bytes 和解析 JSON，并在 `:166`、`:189` 校验源码 pin。
- **漏测：** 全部引用均使用 JSON fixture，未包含真实源码字节。
- **What／已有修复：** 当前 pinned 对 JSON 引用解析，对源码返回原字节。E1 的本地实现接受覆盖该接缝，但本审查未重跑父交付测试。
- **How／长期约束：** 长期接口宜显式声明 artifact kind/content type；先验证原字节 hash/size，再由调用者请求 JSON。扩展名判断是现有兼容接缝，不能作为任意 artifact 类型的完整契约。

### F08：动态 exec 与源码替换未自动传播到后续进程

- **Why／根因：** `d-system-production-entry/ops.py:28`、恢复 `load.py`、`operations_patch.transform` 多层拼接源码执行；父 operations 的 launch/controller/follow 又以脚本路径启动新解释器。当前 globals 覆盖或 import 成功，不会改变下一个解释器导入的 ops、runtime、storage 或 scorer。
- **触发与证据：** 关联 `rework/MAIN-REVIEW.md` 明确记录旧 runtime_adapter import 的依赖缺失，并要求覆盖 continuous.load_parent。`operations_patch.py:142` 至 `:150` 对 runtime/storage 和三个 detached 入口逐项改路由，说明原来不是天然继承。
- **漏测：** 只导入 ops、只检查字符串出现次数，未跨解释器运行真实入口；临时 symlink 和修改 HERE 的 fixture 也不能证明真实部署目录已接好。
- **What／已有修复：** 当前 system 与 control/mention wrapper 在执行 CLI 前装入 override，固定新入口 pin，并保留已接受 scorer 选择；`check_new_gaps.py:128` 起覆盖新 CLI 和实际 runtime/storage/continuous import。
- **How／长期约束：** 短期保留 SHA 和精确 seam 次数校验，另外保存展开源码 hash、模块来源与子进程 argv。长期在新后继树改为正常模块接口和显式参数，不能在活动树中逐层热改 exec；新入口仅是部署适配层，不替代 NPU 核心。

### F09：网络观测失败被提升为观察者死亡

- **Why／根因：** `worker.ssh` 对非零 SSH 状态抛 RuntimeError；`terminal_score.py:84` 在循环读取原 EXIT 时没有捕获，且部分直接 SSH 调用没有总执行时限。ConnectTimeout 只限制建连，不等于完整命令 deadline。
- **触发与证据：** E4 记录确切未捕获异常、本地进程消失和远端评分未知；E3 明确 SSH timeout 不是 runtime terminal。
- **漏测：** 只测启动连通和正常完成，没有覆盖成功 launch 后持续断网、半包传输及 inspect 暂不可见。
- **What／已有修复：** 关联 `final-score/collection-only-repair/collector.py` 固定同一任务与原 guard，设置有界 SSH observation，捕获不可观测状态并约 600 秒继续，仅收集原结果。对应 MAIN-REVIEW 记录了实际交接接受；本轮不重新观察远端。
- **How／长期约束：** 分离 business_state 与 observation_state；unknown/unobservable 不重派、不释放租约、不报终态。恢复读操作可重试，launch/score 操作不得借网络重试盲目重复；成本与已知响应累积保留。

### F10：文件通信只确认写入动作，未保证整段交接闭环

- **Why／根因：** START、CONTAINER、CREATED、AUDIT、FINAL、EXIT、archive terminal、score COMPLETE、COLLECTED 分属不同生产者。文件“有了”不等于引用可读且对应同一实例。`ops.py:312` 在容器启动后写 audit，runtime 最多等待约 30 秒；这也是必须定义的接收协议。
- **触发与证据：** E3 已观察到 post-bind pin 命令失败后 shell 仍继续，后续 observe 声称成功却未独立取回 receipt。评分收集返修 MAIN-REVIEW 还指出 command[-8:] 对真实 7 项后缀恒不相等，造成成功结果也拒收；当前 collector 已改为正确后缀长度。
- **漏测：** 手写 fixture 与验证器同错；只检查脚本退出码或最后一条命令，没有读取实际阶段 receipt；未覆盖延迟文件、缺失 receipt 和部分传输。
- **What／已有修复：** 当前 collector 同核原 EXIT、CID/inspect、COMMAND、source/input seal、完整输出与原评分 schema；部署来源也增加 flow 见证。E3/E1 的 pending 仍不能靠这些代码静态存在而清零。
- **How／长期约束：** 每个阶段必须声明产生文件、消费文件、schema、来源 hash、写入/原子发布方式、deadline 和失败态。关键命令失败必须使阶段失败；成功以实际 receipt 集合为准。业务 COMPLETE、进程 EXIT、归档完成和评分完成分别报告，缺任一项只阻止依赖阶段，不虚构全局完成。

### F11：科学一致性与部署一致性采用同一个含混身份口径

- **Why／根因：** `production.py:45` 的 science 比较排除 runtime/dispatch_allowed，而 `source_adapter.py:232` 附近的 protocol_hash 又基于除 runtime 外的配置；capture 另绑定 runtime。路径、hostname、controller 迁移可以科学等价，却不可能保持全部部署字节一致。若只核 science，又会漏模型服务、source、scorer 或实际 mount 的部署漂移。
- **触发与证据：** E1 对部署修复接受明确 `scientific_promotion=false`；E3 对真实启动接受仍 `no_quality_promotion=true`。源码证明多身份口径并存；没有证据证明本轮已经发生科学污染，本项是已证实的身份接口缺口与失效风险。
- **漏测：** 把完整配置 hash 变化一概解释为科学变更，或把 science 相等解释为部署全等；未固定相同业务模型与 thinking 状态。
- **What／已有修复：** 现 source witness、capture、implementation pins、actual inspect 分别补足部分身份；仍需要主方案明确字段归属和跨版本映射。
- **How／长期约束：** 同时记录科学身份（模型/参数冻结、thinking、数据、知识、检索、提示、评测）、执行身份（run/attempt/request journal）和部署身份（代码/image/service/session/CID/mount/controller/source）。恢复回执显式连接三者。业务推理统一已授权 Instruct 模型与 thinking=OFF；不得随节点默换模型。字段归属变更必须版本化，不能靠新的排除列表偷偷降低校验。

### F12：容器名属于共享 daemon，目录隔离不能避免名字冲突

- **Why／根因：** 同一 Docker daemon 的容器名与 attempt 目录不是同一 namespace；失败容器保留后，重新使用通用名称可在创建阶段失败，且可能已有 DISPATCH 记录。
- **触发与证据：** `s2-reliable-c175-successor/bind-network-repair/v4/recover03.py:14` 至 `:24` 明确要求旧容器已知退出、CREATE.stderr 含 already-in-use、前次没有 CID、没有请求 custody，并保留原双 dispatch 标记再采用唯一 name；不是把该恢复本身另算一轮科学实验。
- **漏测：** 只在空 daemon 或干净 attempt 目录验启动，没有保留旧容器的冲突场景。
- **What／已有修复：** 新恢复名称绑定来源和实例；不删除旧容器伪造“干净环境”，不重新 claim 科学 dispatch。
- **How／长期约束：** create name 包含受控的运行/attempt/阶段唯一身份；创建成功后以完整 CID 为准。create 回执丢失须先 reconcile，不仅凭“目录无文件”重新创建；已知零请求失败和未知启动严格分开。

## 现有 FlightRuntime 应如何复用

已只读检查公共仓与共享权威 src 的 `core/flight/{factory,runtime,cli,lease,native_recovery}.py` 和 `core/contact_journal.py`，没有执行其业务入口。两树均已有：

1. `build_flight_runtime`：按已注册 model/dataset/baseline/experiment 构建同一 FlightRuntime，carrier 通过 TaskAdapter 接入。
2. `FlightRuntime.preflight/run_phases/seal`：承接准备核验、执行和封存；应评估把未来来源适配接在该既有生命周期中。
3. `GenerationLease` / `WorkerExecutionPolicy`：稳定 guard inode、协调器持有租约、worker 只验证对应 run/attempt/session，不得另造一套互斥代替旧 guard。
4. `ContactJournal` / `NativeContactReplay` / `load_native_recovery`：保留原始响应、区分 unknown 与 observed、基于固定来源做恢复，不因断链重新调用。

**复用结论有版本边界。** 本轮有界比较发现 7 项中 1 项相同（cli），5 项内容不同（runtime、factory、lease、native_recovery、contact_journal），另 1 项 `runtime_overlay.py` 仅存在于公开仓。公开 factory 提供外部运行配置校验；权威 factory 本轮所见未包含同一接口。因此不能把公共仓较新接口当作权威现场已经可用，不能仅凭同名类就直接切换。差异原因和完整兼容性本轮未审计，不能判任一树天然错误。

主方案应固定源树与版本，沿现有 TaskAdapter、工厂、租约、journal、官方评分器接入有限 deployment/source adapter，并以兼容性证据规划后继迁移。当前 R6 adapter 与 FlightRuntime 是否可直接无损互换仍待接口测试；不建议在活动实验中移植。任何新的 CLI、preflight、配置生成器或 supervisor 都只能承担工程外壳职责，不能宣称替代模型服务、NPU 执行、特征管线或既有科学核心。

## 主方案目前仍不充分的地方

- 77 项本地 checks、pins 零差和目录 staged 不能替代固定镜像真实 mount/权限/新解释器检查；187 个 mount 的有效闭包仍需逐角色说明。先完成有限部署验收，再谈“统一入口已解决”。
- 三个局部补丁（路径、hostname、controller kind）还需覆盖 producer→publisher→feed→controller→continuous→storage→archive→scorer→collector 的同一来源链；只修启动端不足。
- 动态 exec 的 hash 与 seam 次数能拒绝父源码漂移，却不能证明替换后的语义、子进程继承和实际 mount 正确；不能把“配置化”当消除这种风险。
- schema 不能只增加字段然后要求旧任务满足。真实旧记录缺失历史快照时，须有不造史的有限 reconciliation；未知请求和费用必须原样保留。
- 监测应有独立生命周期，断网不结束任务；观测重试和产生业务效果的重试必须分开。一个入口同时容纳 upload/launch/score/collect 会增加误重派风险。
- 共享存储政策须在实际容器和所有可写路径执行；仅配置最终 output 在共享盘不够。旧本地热状态条文不能继续驱动新部署。
- 公共和权威 FlightRuntime 版本不同，复用计划必须给出明确版本、API、恢复兼容边界。不能另建 Python 执行器后将其包装成“NPU 核心替换完成”。
- 完整评分与科学收益仍待原任务产物；不能因工程恢复重新抽样、改变原科学配置或启动下一代候选。现有已接受交付保持原封存。


## 本次整改状态

| 改动 | 已交付 | 验证边界 |
| --- | --- | --- |
| 公共配置入口 | validate / plan / prepare / run / status / reconcile，严格 schema、DAG、配置锁 | 本地真实子进程验证 |
| 启动与恢复 | detached controller/worker，dispatch-before-spawn，PID+boot+start_ticks，未知不重发 | 包含杀控制器、超时、非零 NPU 返回测试 |
| 资源协同 | CPU/NPU 独立并发，跨实验 lane，依赖就绪才占配额 | 无模型 fixture；保留原生卡准入和 generation lease |
| 容器复用 | 原 configured_job_launch 新增 foreground run 与显式 hostname | Docker 协议 mock；真实固定镜像待双机恢复 |
| 镜像构建 | 共享根与 CPU/内存配置化；发布清单包含公共入口 | 实际 shell + recording Docker 测试，不冒充镜像已构建 |
| 输入身份 | 小控制文件核内容，大型明确不可变资产复用缓存 | 复现同大小/同时间改写漏检并锁定回归 |
| 阶段交接 | 统一回执、声明输出、失败按 DAG 传播，600 秒心跳 | 不自动接管旧任务，不把退出码代替科学评分 |

**尚未完成的迁移**：旧恢复 adapter 的动态源码补丁、历史挂载闭包和权威源树版本差异没有因新增入口而自动消失。它们留在旧任务证据和接续路径内；后继业务适配必须迁入正常模块接口，并在选定 release 内核验实际 source→runtime→score。当前不能宣称双机完整工程基线已部署。

验收记录：91 项测试通过，其中公共执行器/构建入口 20 项、既有容器入口 71 项；另有独立三任务 CPU DAG 完成输入生产、并行分析和后继消费，模型调用为 0。详见 [运行约定](Experiment-Base.md)。

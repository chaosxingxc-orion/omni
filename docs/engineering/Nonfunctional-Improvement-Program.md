# 非功能改进实施台账

本页持续原位更新。依据[工程规范](development-and-quality-contract-zh.md)，先锁定功能行为，再以真实故障和测量结果选择改进。当前已交付环境预检/后继绑定、配置化异步归档与零调用终态租约核验，并已用于真实Direct任务。连续多任务可靠性和吞吐收益仍须实测，不能由单项通过推断。

## 已交付：推理前发现评分环境缺项

问题：现有配置化启动器检查提交、输入和挂载，但不验证容器内 Python 依赖。官方评分器优先使用 `SAEA_SLURP_OFFICIAL_PYTHON`，缺失时可回退到业务解释器；业务解释器的可用性并不保证官方评分依赖存在。

新增固定工具 `studies/speech-aware-evidence-acquisition/scripts/environment/configured_environment_preflight.py`，以 JSON 声明解释器、必须匹配的环境变量、真实导入模块及精确包版本。检查在独立 CPU 容器中执行，异常、超时、解释器或依赖缺失均写失败结果且不产生成功凭据。每次使用新的输出目录，旧结果不覆盖；不安装依赖，不自动重试。

锁定的行为：所有检查通过才产生 `ready.json`；其中绑定配置与结果摘要，结果包含脚本摘要、实际解释器与 Python 版本。后继任务利用原启动器的 `required_inputs` 固定成功凭据 SHA256。失败或凭据漂移时，原启动器输入检查拒绝执行。

模板：

- [检查内容配置](../../studies/speech-aware-evidence-acquisition/configs/launcher/environment-preflight.example.json)
- [固定镜像 CPU 启动配置](../../studies/speech-aware-evidence-acquisition/configs/launcher/environment-preflight-job.example.json)

配置中的路径及全零摘要是占位符；独立评分环境挂载需按已有封存环境替换。没有请求网络和设备，不运行推理或评分任务。CPU 容器复用[固定启动器](configured-job-launch-zh.md)的 `layout/plan/launch/status` 流程，不增加启动器 schema 或恢复分支。业务阶段仍执行自己的注册、准入和互斥。

部署顺序：固定执行提交和环境；填充预检 JSON；在与后继相同的固定镜像、依赖挂载、解释器和必要环境下运行预检；核对成功结果；把 `ready.json` 路径及 SHA256 加入后继的 `required_inputs`。再用下述绑定工具验证两份启动配置并启动后继。环境、镜像、依赖挂载或选择变量变化必须重新预检；模板不自动生成后继配置。

## 已交付：预检凭据与后继环境绑定

新增独立 `scripts/environment/configured_preflight_binding.py`，原 `configured_job_launch.py` 的文件、schema、runner 摘要及既有恢复行为均未改变。它提供 `bind/plan/launch`：先验证来源和环境，随后直接委托原启动器，继续使用原有 guard、启动意图、幂等状态及未知启动不重发机制。

`bind` 要求源是直接执行固定预检脚本的 CPU 容器（network none、无 host PID），且其 Docker 身份与启动 intent 一致、已经 exit 0；ready 必须来自该配置输出挂载，绑定真实检查配置、脚本和结果摘要。后继必须固定 ready 摘要。兼容性严格要求镜像、全部显式环境变量、工作目录、执行仓及提交相同，业务入口解释器已经探测；源输出挂载外的挂载全部只读，后继保留它们的 source/target/readonly。新增挂载不得覆盖依赖、解释器、系统 Python 路径、工作目录或绝对环境路径；业务网络和不重叠的结果挂载可不同。

绑定文件固定两份完整配置、ready/result/检查配置/脚本/intent 摘要、源容器 ID 及 helper/启动器/检查模块版本。之后 `plan/launch` 每次重新校验上述文件、兼容性和精确后继配置；任一漂移在委托启动前失败。绑定文件只创建一次并 fsync，不覆盖旧证据。示例从 study 根目录运行，所有部署路径均为私有配置：

```bash
python3 scripts/environment/configured_preflight_binding.py bind --source cpu-launch.json --ready /owned/preflight/attempt-v1/ready.json --successor successor.json --binding binding.json
python3 scripts/environment/configured_preflight_binding.py plan --successor successor.json --binding binding.json
python3 scripts/environment/configured_preflight_binding.py launch --successor successor.json --binding binding.json
```

目录创建、status/watch 仍走原启动器。采用预检绑定的后继必须通过此工具 `plan/launch`；直接调用原启动器只执行原有检查，不会强制读取绑定文件。严格相同环境和执行提交是本版刻意保守的兼容范围，存在差异时准备匹配预检，不降低判断条件。

此工具比较挂载配置，不冻结外部宿主机目录内容；容器 readonly 不阻止宿主机原位修改。依赖来源必须保持冻结，发生内容变化需重新预检；绑定不抵抗人为伪造凭据，也不替代原生科学准入。固定镜像回归已通过，真实Direct启动及恢复均已完成环境绑定；实际完整测试结果仍待完成。

本机绑定测试23项通过（1.928秒），原预检13项回归通过（0.704秒）；Docker终态使用替身，子解释器导入真实执行。覆盖正确绑定、缺失/失败/异源凭据、镜像/环境/解释器/依赖挂载漂移、嵌套遮蔽、绑定后改配置、实现身份改变和原未知启动结果保留。后续运行路径扩展的固定镜像66项与7个子用例通过，真实绑定也已验收；分批用例重叠，不相加计数。

## 验证与限制

本机标准库测试覆盖真实子解释器导入、传递依赖缺失、包版本不符、缺少解释器、环境选择冲突、超时、重复输出保留、CLI 退出码和原启动器输入门禁。测试使用的临时包仅用于失败注入，不用于估算实验成本。固定 saea 镜像中，新预检与原启动器回归共24项通过（2.13秒）；真实同镜像 CPU 预检已通过，业务 openjiuwen、独立评分解释器及精确依赖版本均可用，耗时0.745891秒、0模型调用。真实Direct原启动预检4.374秒、恢复预检3.592秒均通过并绑定后继；吞吐改善及节省卡时仍未知。

本机执行 `python3 studies/speech-aware-evidence-acquisition/tests/unit/test_configured_environment_preflight.py -v`：13项通过，0.692秒。系统 Python 无 pytest，未安装新依赖；标准库测试可由固定镜像 pytest 收集。JSON 检查模板 schema 和 `git diff --check` 通过。

| 后续项目 | 当前证据与验收要求 | 状态 |
| --- | --- | --- |
| 预检接入真实任务 | Direct及恢复后继已绑定真实ready；完整任务评分仍待完成 | 启动绑定已验收，完整评分待验收 |
| 终态驱动后继 | 前驱封存、未知请求为零、guard 正确交接；测量任务供给空档 | 未交付 |
| 评分与推理异步 | 保留轨迹/成本，CPU 失败不重发已知模型调用；测量到结果时间 | 未交付 |
| 长任务稳定性 | 多任务连续交接，记录队列、资源和失败率；以实际负载设目标 | 待测量 |

以上后续项目不构成已完成的功能声明，保留原始响应、历史成本和唯一运行互斥是所有变更的前提。

## 已交付：异步归档与零调用失败交接

`configured_remote_archive.py once/watch --config ...`在开发机执行；配置限定项目自有目录、明确的共享attempt及容器身份。每600秒或终态归档，流式SHA256、fsync，保留失败残件和变化中快照；tar不是事务快照，不自动恢复。固定镜像16项与12子用例通过，真实Direct终态归档798,720字节、176成员通过清单校验，generation attempt收录、独立FIT attempt缺失显式记录。模板及操作见[归档配置说明](../../studies/speech-aware-evidence-acquisition/configs/launcher/remote-archive-example.md)。

`reconcile_zero_contact.py --config ...`仅处理精确终态、ABORTED且实际调用/音频均为0的失败；核验原生来源、native/FIT无记录、目标未创建和双实例静默，持有原guard后保全旧租约。不适用于已消费或未知请求，不自动发起恢复。租约在原locks目录原子改名，避免不同容器挂载之间的EXDEV；原始字节另纳入私有归档。最终版固定镜像16项与6子用例通过，包含真实NativeContactReplay及final-report allowance。

真实Direct旧租约已由较早的受验快照完成交接（11项与6子用例），新增模型调用0；最终版额外加强baseline、容器生命周期/环境与guard inode绑定，不能把新测试冒称该版已经现场执行。生成恢复仍复用`resume_registered_workload.py`和原注册科学配置。

本次通用音频FIT修复另有53项正式回归通过；完整13,078条实际FLAC字节哈希及头部时长全部一致，核验6.931秒、0模型调用。依赖导入预检不替代真实数据格式契约测试，这项缺陷须归入功能测试基线。

## 约束、节奏与优先级

固定负责人维护此页，主线程整合审查、固定镜像验收、提交和发布。每次只推进一个有证据的小改进，先完成当前验收再接受下一项；没有新的故障或测量依据，不扩展框架。日常监测仍为终态事件或每600秒，不添加高频轮询。新代码只部署到后继冻结执行树，运行树、共享guard、历史响应与成本不得修改。

优先级 P0 为正确性及防重复：未知启动不重发、正确环境选择、失败凭据不作成功凭据。P1 为已有真实故障造成的阻塞：此次依赖预检、保留生成结果后的独立评分恢复。P2 为量化后再排序的性能及可观测性改进；不让其抢占功能闭环与有用 NPU 工作。

## 实验依据与指标

触发证据来自主线程核验的 H1 条件复测：313条生成、626调用全部成功，但评分因业务解释器缺少 `progress` 失败；独立评分 Python 已存在，启动配置遗漏了选择变量。本改进只提前发现此类环境错误，不能声称回收既有调用成本。原始轨迹和内部路径不进入公开台账。

| 指标 | 口径与当前值 | 验收标准 |
| --- | --- | --- |
| 错误环境阻断 | 缺解释器、模块、传递依赖、版本及选择变量用例 | 推理前失败，保留 result，无 ready |
| 新模型调用 | 独立 CPU 导入检查；测试未调用模型 | 0；模板无网络、无设备挂载 |
| 重复执行 | 输出目录重复或已存在 | 不重写结果、不再次发起探测；启动器仍保持原幂等语义 |
| 预检开销 | result.elapsed_seconds 为实际本次预检耗时 | 真实同镜像0.745891秒；不由单次推断长期性能 |
| 任务进度 | 当前为 CPU 工程验证，不是研究样本完成量 | 报告测试用例数与耗时；卡数/实例均0 |
| 节省时间、卡时、成本 | 已实际部署，尚无独立对照结果 | 未知；使用固定真实任务比较，不用重复合成样本推算 |

## 验收与回退

本机及固定镜像测试锁定上述行为；真实 CPU 预检已成功产生凭据。H1 封存结果也已用既有独立评分环境恢复为 FINALIZED，新增模型调用0。相关后继必须绑定该 ready 摘要，且复核固定镜像、依赖挂载、业务及评分解释器、选择变量一致。成功导入不替代实际评分回归，不代表训练成功或完整测试集结果。

需要回退时，仅从尚未启动的后继配置移除预检前置绑定或采用上一版独立执行树；保留已生成的预检与启动凭据。不得编辑已绑定的运行树或原配置、删除 intent、修改已有 attempt 身份、自动重启或重发未知请求。已有评分失败由独立恢复入口消费封存输出，不重跑模型生成。


## 全量准备中的重复核验

真实Direct日志显示进程maps检查占约81%的request_render嵌套耗时，音频读取并非主瓶颈。通用字节精确解析缓存和路径去重已通过129项正式测试；每次重读live bytes，scope退出清理，未知/漂移仍拒绝。本地真实maps微基准仅证明CPU组件下降，线上端到端收益另测。已有准备接近完成时，应比较剩余成本和重启重复成本，不把局部加速比直接当中断依据。

通用队列扩展已通过104项正式测试，支持固定learning JSON命令与单run挂载下的原始sealed artifact核验。部署新helper目录并重新绑定摘要，禁止修改活动队列的固定实现。独立准备/学习任务必须在启动前验证全部可写attempt、cache和日志路径；预检imports成功不能替代这些真实写入接口。

## 归档等待故障修复

2026-11-02发现Direct与A0编码的开发机异步归档在首次快照后因等待超时退出，实验容器继续运行。共享归档脚本增加远端5秒强制终止宽限，将等待超时/SSH断链作为未知观测处理，保持600秒节奏；下一轮重新核验原容器ID，身份变化拒绝，非预期命令错误仍失败。未重发模型请求。固定镜像19项测试及14个子用例通过（0.37秒）。这次修复覆盖wait阶段，inspect和快照失败后的通用恢复仍需后续完善。

两个归档watch已用原JSON恢复。14:42实时归档：Direct 587,479,040字节、9,385成员、清单核验通过，tar标记changed-during-read；A0编码61,870,080字节、3,459成员、清单核验通过。实时tar均非事务一致快照，不能作为已封存终态。原失败日志与快照保留。

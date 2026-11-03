# 配置化任务启动基线

## 固定入口与职责

启动器版本 `1.0.0`：`studies/speech-aware-evidence-acquisition/scripts/environment/configured_job_launch.py`。固定镜像、执行提交、挂载、输入摘要、资源、环境及参数均由JSON配置声明。每个阶段使用独立配置和容器名，复用现有 `prepare/register/admit/run` 业务入口；启动器不替代其预算、来源、FIT和互斥校验。

新增实验交付配置，不复制Python启动器或拼装临时shell。配置不包含凭据，真实部署配置保存在私有控制目录；公开[模板](../../studies/speech-aware-evidence-acquisition/configs/launcher/job.example.json)故意使用占位摘要，必须替换后才能运行。模型、数据、预算由业务配置固定，启动配置以 `required_inputs` 绑定它们及前置事件。

## 镜像与外部脚本的边界

允许固定版本的通用脚本和业务实现从外部只读挂载。镜像摘要锁定依赖环境，`repo_path`与`expected_commit`锁定脚本来源，`required_inputs`绑定外部配置。新增实验复用同一脚本提交，只调整配置、数据引用与独立输出目录。源码维护发布与日常运行分开；不要求将外挂脚本重新打入镜像。旧注册/准入仍依赖写执行树的部分需要迁移到配置与状态目录，不能把它称为已完成的配置化交付。

## 操作顺序

从仓库根目录执行，`job.json`是已填好的私有配置：

```bash
python3 studies/speech-aware-evidence-acquisition/scripts/environment/configured_job_launch.py layout --config job.json
python3 studies/speech-aware-evidence-acquisition/scripts/environment/configured_job_launch.py plan --config job.json
python3 studies/speech-aware-evidence-acquisition/scripts/environment/configured_job_launch.py launch --config job.json
python3 studies/speech-aware-evidence-acquisition/scripts/environment/configured_job_launch.py status --config job.json
```

`layout`只创建明确声明、位于项目自有根内的目录，并检查只读父挂载下的嵌套挂载点；`plan`检查干净提交、输入摘要及挂载，不启动任务；`launch`先持久化启动意图，再以参数数组启动后台Docker；`status`核对配置和启动器摘要及容器身份。`watch`可作为后台CPU任务运行，等待容器终态或600秒超时，原位更新状态。

业务运行固定使用 `/opt/saea-venv/bin/python`，该环境包含openjiuwen；系统Python仅用于确实不依赖该框架的工具，不可互换。配置中的工作目录建议使用可写的 `/tmp`；数据锁、归档来源、模型及历史运行的每个逻辑路径都必须显式挂载。服务调整是独立任务，不在启动器中隐式重启服务。

## 断点与失败边界

相同配置重复调用只读取原容器状态。存在启动意图但无法找到容器时，返回 `unresolved-launch-intent`，禁止自动重发。配置变更、同名外来容器、输入变化和脏执行树均拒绝。不能用新容器名绕过业务attempt和未知请求核验。

确认尚未进入模型请求的启动错误，可保留旧记录，以新配置修复路径或挂载；已有请求则走原生恢复入口。容器退出0不等于实验成功，仍须检查共享运行时封存和评分证据。

历史任务漏写释放事件时，可用 `scripts/analysis/finalized_generation_release.py` 在共享guard下验证FINALIZED包并生成可追溯的终态交接事件，不改历史执行树，不伪造原释放时刻。未来分片/融合入口直接通过campaign释放回调写事件。

## 基线与未完成项

固定镜像32项相关测试通过（5.41秒）：含幂等启动、响应丢失/启动失败不重发、配置漂移、目录缺失、已封存前驱验证及释放回调。真实CPU交接任务已通过此入口完成；真实融合执行的业务结果另行验收。

此版本锁定单任务启动和观察，尚不是全部任务类型的统一流水线：自动建工作树、配置生成、register/admit间提交编排、多任务优先级和故障恢复调度仍待迁移验收。后续在这个入口增量完善，不能另造逐实验启动脚本。改执行器需要版本、缺陷说明、相关回归及真实任务验收；无新证据时冻结执行器，把精力用于算法实验。

原生语音任务在发送前失败、attempt明确ABORTED且调用/音频均为0时，允许通过显式原生恢复声明接续，保留零成本前驱。32项相关恢复回归通过（4.53秒）；存在未知或未核验成本时仍不能重发。

## 恢复与归档的通用入口

原始Direct音频可能为FLAC；准入使用真实wire头部读取时长，不转码或相信外带duration。完整输入格式校验与环境依赖预检分别验收。预检的业务输出目录由预检脚本独占创建，launcher layout只准备父目录；required_environment仅声明必须存在的非空变量。

零调用故障使用`reconcile_zero_contact.py --config ...`核验并保全准确旧租约；通过后由`resume_registered_workload.py`创建明确的新attempt。原attempt、历史成本与执行树不改，generation.guard不删除或替换。带请求的恢复必须使用对应原生来源契约，不能套用零调用工具。

归档统一使用`configured_remote_archive.py watch --config ...`，实例、路径与attempt清单只改配置。监测、归档及CPU分析脱离SSH，默认600秒或终态事件。固定启动器不自动编排多个业务阶段；后继是否就绪必须以来源、预算、预检和恢复证据判断。

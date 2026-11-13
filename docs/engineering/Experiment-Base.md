# 配置驱动的实验基座

公共入口管理配置冻结、依赖调度、进程回执与状态观察；业务继续使用已有 FlightRuntime、TaskAdapter、ContactJournal 和官方评分器。它不替代模型服务，也不绕过 card_queue_guardian 或 generation lease。新增实验应提交配置差异，不能在 delivery 中复制旧 launcher、用源码字符串替换生成新程序。

## 日常接口

```bash
python3 scripts/experiments/experiment.py validate --config experiment.json
python3 scripts/experiments/experiment.py plan --config experiment.json
python3 scripts/experiments/experiment.py prepare --config experiment.json
python3 scripts/experiments/experiment.py run --config experiment.json
python3 scripts/experiments/experiment.py status --config experiment.json
python3 scripts/experiments/experiment.py reconcile --config experiment.json
```

`validate/plan` 只读。`prepare` 固化完整配置、数据清单和输入身份；配置变更必须使用新 experiment.id。`run` 在执行主机启动独立会话，不依赖 SSH 持续连接。`status/reconcile` 区分历史状态与当前可观测的进程身份；reconcile 只观察，不能自动修复、清租约或重发。`--phase cpu|npu|all` 可以分阶段执行，依赖尚未完成的阶段不抢 NPU 配额。

配置只支持显式 `${shared_root}`、`${run_root}`、`${config_dir}`，未知变量或字段立即报错。argv 是参数数组，无 shell 插值；环境变量显式配置，不隐式继承代理、凭据和缓存路径。配置不能内嵌凭据。解释器、源码、业务配置和清单作为 release/task artifacts 固定。小控制文件及数据清单按内容核验；仅大于 1 MiB 且在 artifacts 中显式设置 `immutable: true` 的发布资产复用已验证记录，不能把可变数据伪装成 immutable 资产。

`storage.run_root` 是共享执行命名空间，实验在 `experiments/<id>` 下隔离，输出在其 `outputs` 子目录，公共 lane 在 `lanes` 子目录。所有使用相同模型服务的后继实验必须使用相同 lane 配置；lane 是调度配额，不是物理卡授权。业务执行继续保留原生资源准入。任意 argv 本身不是沙箱：程序及容器仍须遵循声明的读写路径。

## CPU / NPU 协同

CPU 数据准备、分析、评分分别作为任务；NPU 任务只在依赖成功后进入 lane。无依赖 CPU 任务可以与 NPU 推理并行。独立任务不因另一路失败而停止。失败会沿 DAG 阻塞后继；状态不明的请求不会自动重发。

每个 task argv 必须保持前台直到其业务结束。不能把 `docker run -d`、只派发后退出的脚本或 `configured_job_launch.py launch` 当成完整推理节点。容器复用现有 `configured_job_launch.py run --config launch.json`：它沿用原 launch custody，随后等待容器终态，透传退出码；不能观察到终态返回 75。NPU 非零退出仍可能留下服务端请求，因此保留未知状态和 lane，不自动接续。

共享盘状态不使用每秒 fsync 轮询；调度检查和状态发布分开。远程观测按既有 600 秒或终态事件执行，网络不可达不等于业务失败。

## 固定镜像与部署

复用 [release 构建工具](../../docker/saea-base/release/README.md)，不在线升级基础依赖，不复制模型、数据或整棵历史工作目录到新实验。外部源码挂载只读，配置与数据只读，输出/日志/缓存/临时目录显式挂入获准共享根。

容器名称、hostname、CID 分别记录，禁止由 hostname 推导 CID。网络能力按任务配置：访问模型服务需要可达服务的网络，纯离线处理可为 none。控制器类型须与真实 Popen/systemd 一致，不能用一个不存在的 systemd unit 判断 Popen 任务结束。

现有固定 parent 的内容身份为 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`。这是已有本地 image ID，不是新发布镜像或公共下载地址。本次公共代码尚未完成该镜像内的双机部署验收。

## 迁移边界

当前 S2 活动请求和 D 已启动评分保留原配置、journal、guard 和观察者；不能用新 experiment.id 再跑一遍来代替恢复。历史 START 缺 boot/start_ticks 的记录不能补造字段。后继任务在原终态确认后采用公共入口；模板见 [existing-container.template.json](../../configs/experiments/existing-container.template.json)。模板不是已获准的科学实验，不会自动移交旧租约。

公共仓和共享权威源树的 FlightRuntime 版本存在差异，必须选定单一 source release 后再部署，不能把两棵树按文件拼装。现有 recovery adapter 的动态 exec 与历史挂载闭包仍留作旧任务接续，尚未宣称已整体消除。后续将业务适配接入已有工厂和 TaskAdapter，须用真实 source/score 接口验证，不能仅把旧脚本包一层就称完成迁移。

业务基线保持 Instruct、thinking OFF，TRAIN6681 / DEV1850 / TEST13078；科学身份（数据、知识、提示、评分）与执行部署身份（路径、实例、进程、镜像）分别记录，工程修复不构成指标晋级。

## 验收与可复用非功能约束

先验证功能：真实 CLI 准备与启动、依赖执行、CPU/NPU 配额并行、共享 lane 互斥、产物消费、失败传播、状态恢复。然后验非功能：SSH 断链独立、PID 重用识别、未知请求不重发、原子回执、配置不可原位漂移、共享盘无本地降级、低频状态写入。

```bash
python3 -m pytest -q tests/experiments \
  studies/speech-aware-evidence-acquisition/tests/unit/test_configured_job_launch.py
```

本地测试中的 NPU 标记节点运行无模型 CPU fixture，仅验证调度语义，不能证明 NPU 性能、Ceph 跨主机锁行为或模型精度。远端恢复后的最后验收是：固定镜像内实际 import/挂载/权限检查、原任务只读核验，再以既定业务配置做小规模端到端接续。通过后扩大运行，不重复已完成数据准备和全量哈希。

## 多轮对抗审查后的补充契约

同一 experiment 只允许一个 controller。CPU/NPU 持续并行使用 `run --phase all`；`--phase cpu` 与 `--phase npu` 是串行分阶段入口，不能同时各启动一个 controller。控制器换代时，可观测仍存活的旧 worker 继续计入并发配额。

退出回执必须与原 DISPATCH、配置身份一致；不匹配视为 UNKNOWN。status 使用原冻结配置，不能用修改后的模型设置给旧结果换身份。worker 在执行前复核输入 pin；PREPARE 必须与 CONFIG-LOCK 的原始输入摘要一致。产物不得多任务重复声明，验收时重新检查实际路径。此方案不把可写源树变为安全沙箱：部署仍须使用不可变只读 release。

`execution.bindings` 显式绑定下层配置：每项含 `path`、`sha256` 和 `matches`，后者将下层 JSON pointer 映射到当前实验 JSON pointer。例如 `/image` 对 `/release/image_digest`，业务配置中的真实模型/thinking 字段对 `/experiment/model` 和 `/experiment/thinking`。prepare 和 worker 核 pin 与字段相等；未配置 bindings 的通用 argv 只记录元数据，不证明业务身份。模板包含待填写的字段映射，不能假定不同 workload 有统一字段名。

release 的 CPU stage/prepare/register 模板使用 `workspace_role: prepare`，仅允许固定发布目录中的离线 preparation 脚本及有限模式，network 必须 none。它与后续 generation 的 admission 分开；generation 仍要求完整原准入，不得用 preparation 角色运行推理。

新发布模板使用 `runtime_storage`，将 `/run/saea` 和 `/tmp` 映射到获准共享根，设置 HOME/tmp/cache，stdout/stderr 落共享文件，Docker log driver 为 none。旧配置保持兼容，其原 tmpfs 行为不等于符合新存储约束。真实容器的用户写权限、Ascend 导入和挂载仍待远端验收。固定镜像继承的驱动库路径保留，禁止通过任务环境覆盖 LD_LIBRARY_PATH；需要不同依赖应发布单独、可核验的镜像。

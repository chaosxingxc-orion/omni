# 新协作者接入检查单

目标：一个新进入项目的人，用**少量本地配置**在固定镜像内启动一次实验，不改任何源码，不需要知道任何开发机路径。

本检查单的每一步都由 `scripts/experiments/onboarding_drill.py` 自动执行并给出证据；文档是它的说明，脚本是它的可执行版本。两者不一致时以脚本输出为准，并回头修脚本或修文档，不允许只改一边。

```bash
python3 scripts/experiments/onboarding_drill.py --root /tmp/onboarding --keep
```

最近一次实测（本地 CPU，无 NPU、无共享盘）：10 步全部通过，耗时 3.2 秒，新人手填 7 个字段。

## 只有两步是人做的

### 1. 复制本地部署配置并填 7 个字段

```bash
cp configs/local/deployment.example.json configs/local/deployment.json
```

`configs/local/deployment.json` 已在 `.gitignore` 中，永远不提交。必需字段共 7 个，示例文件已经把它们都写出来了（值仍是 `REPLACE_WITH_*` 占位）：

| 键 | 含义 |
|---|---|
| `storage.shared_root` | 已存在的共享根（模型、数据、归档的来源） |
| `storage.hot_root` | 本地持久盘上的热状态根（必须真实本地文件系统） |
| `storage.archive_root` | 归档根 |
| `roots.model` | 模型目录 |
| `roots.data` | 数据目录 |
| `image.reference` | 人读的镜像引用（tag） |
| `image.digest` | `sha256:<64 hex>`，校验用；与 reference 成对出现 |

其余字段（`hosts.*`、`services.*`、`identity.*`、`paths.*`、`ports.*`）按需填写，不填不影响启动；只有实验真正引用到某个键时才会报错，并给出该键的 `${local.<key>}` 写法。

规则（全部由解析器强制执行，不是约定）：

- 值必须是字面量，**禁止**在本地文件里写 `${...}`；本地文件不做第二层解析。
- 路径必须绝对且不含 `..`。
- 未知键报错，防止拼写错误静默失效。
- 键名命中 token/password/secret/key/credential 模式即报错；本文件不进版本库，但也不该放凭据。

### 2. 复制一个实验模板并改 `experiment.id`

```bash
cp configs/experiments/onboarding-drill.template.json configs/experiments/<新 id>.json
```

模板里所有本地信息都写成宏，例如 `${local.storage.shared_root}`、`${local.roots.data}`、`${local.image.digest}`。新人只需要改 `experiment.id`（以及自己的数据集路径）；不需要改任何绝对路径。

`onboarding-drill.template.json` 是故意做成 CPU-only、`/bin/true` 的两节点 DAG：它证明基座能从本地配置启动，**不产生任何科学结果**，`science.purpose` 明确写了这一点。

## 之后四步是固定 CLI

```bash
python3 scripts/experiments/experiment.py validate  --config configs/experiments/<id>.json --local-config configs/local/deployment.json
python3 scripts/experiments/experiment.py plan      --config configs/experiments/<id>.json --local-config configs/local/deployment.json
python3 scripts/experiments/experiment.py prepare   --config configs/experiments/<id>.json --local-config configs/local/deployment.json
python3 scripts/experiments/experiment.py run       --config configs/experiments/<id>.json --local-config configs/local/deployment.json --phase cpu
python3 scripts/experiments/experiment.py status    --config configs/experiments/<id>.json --local-config configs/local/deployment.json
```

在固定镜像内执行时，命令不变，只是把 `python3` 换成镜像入口（见 [固定镜像与部署](Experiment-Base.md)）。服务、实验和队列都必须脱离 SSH 运行。

## 通过条件

1. `validate → plan → prepare → run` 四步全绿，任务到达终态 `SUCCEEDED`。
2. `prepare` 落下 `CONFIG.json`、`CONFIG-LOCK.json`、`LOCAL.json`、`PREPARE.json`。
3. `LOCAL.json` 只含本次实际引用的键 + `file_sha256` + `local_sha256`，不整份复制本地文件。
4. 删掉本地配置文件后，`status --from-snapshot <run_dir>` 仍能读出这次运行（跨机观察走这条）。
5. 本地信息零泄漏：`check_local_leak.py` 在基座目录内除 allowlist 外零命中。

## 反向验收（新人一定会踩的四种）

| 情形 | 期望 | 实测错误信息 |
|---|---|---|
| 故意删掉 `roots.data` | `validate` 报错，指出缺哪个 `local.<key>`、该改哪个文件 | `missing required fields ['roots.data']; add them in the deployment profile (${local.roots.data}) and pass it with --local-config` |
| `shared_root` 指向不存在的目录 | `prepare` 报错且**不留半成品** | `shared_root must already exist; no fallback storage` |
| 已 `prepare` 后改动被引用的本地值（即使解析结果相同，例如换成软链） | `prepare` 拒绝复用 | `local deployment profile changed for this experiment.id; prepared run keeps its frozen values` |
| 上一个 controller 还没退出就再次 `prepare`/`run` | 明确提示互斥锁被占用，而不是裸 errno | `run is busy: another prepare/controller holds <run_dir>/RUN.lock; observe it with status or reconcile, then retry` |

第三条是本设计的关键：实验配置的摘要看不出来源值变化，只有冻结的 `LOCAL.json` / `CONFIG-LOCK.json` 里的 `local_sha256` 能看出来，所以它单独拦住。

## 演练脚本做了什么

`onboarding_drill.py` 把基座复制到一个全新绝对路径（不是任何人的 checkout、不碰 `/mnt`），只写一个本地配置和一个实验配置，然后驱动真实 CLI 走完上面全部正反例，最后输出 JSON 报告：

```json
{"steps": [...], "profile_fields": 7, "ok": true, "failed": [], "elapsed_seconds": 3.209}
```

它是**证据生成器**，不是替代品：脚本里出现任何失败，一律按真实缺陷处理（修 `tools/experiments/` 的可复用实现并补测试），不允许在演练脚本里绕过。

已知边界：演练在本地文件系统上运行，证明的是配置面与 CLI 语义；镜像内 import/挂载/权限、NPU 性能和 Ceph 跨主机锁行为仍需在真实主机上另行验收，不能由本演练代替。

## 不在本地配置里的东西

`8 卡训练 / 4 卡 rollout`、TP 拓扑、门禁阈值、采样比例这些是**已批准的资源布局与策略常量**，不是某个人的本机信息；它们属于版本化策略配置，不写进 `configs/local/deployment.json`。区分标准：换一台机器会变的写本地配置；换一个人也不该变的写策略配置。

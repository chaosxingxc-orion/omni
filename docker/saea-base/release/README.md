# 版本化运行镜像与外部实验配置

同一个 release 镜像可以分别运行 CPU 预检、stage、prepare、register、queue、observer、score，以及既有 NPU 服务部署入口。业务源码在 `/opt/saea-release`，环境沿用固定 parent；配置、数据、模型、知识和输出通过外部挂载提供。CPU 角色不需要 NPU 设备。

本目录的 [examples](examples/) 是**不可直接执行的配置模板**。`<…>`、`/REPLACE_WITH_OWNED_ROOT`、`example-*`、容器名和条数都必须替换；占位 hash 不是真实 pin，字符串条数也不是合法整数。样例不包含真实数据、名单、模型文件、私有主机配置或凭据，不提供新开发集。必须使用已有获准 workload 的原名单、顺序、音频身份、知识与科学参数；`workload.json` 由实验自身既有合同提供，不要用样例伪造。

## 镜像、源码和依赖身份

已验证的历史 v8 身份如下；这些内容摘要不含运行环境位置：

| 项目 | 身份 |
| --- | --- |
| 固定 parent 本地 image ID | `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026` |
| v8 构建源码 commit | `8b4740251cbb491e8d6a24bae02a8f1fac26c5c6` |
| v8 本地 image ID | `sha256:b342a3f9c4af4e4f3c085b3bd1fd739aaa7618b6842b871f4b47426e226ddb8b` |
| v8 source-manifest 文件 SHA256 | `f1f9463f34a47148c0eca708c3a4dcb94c746cfc2b047d32fbe99c71fbfcd2ac` |

相关功能已集成到公共源码 `ee280aec`，但该 commit 与 v8 构建 commit 身份不同。文档提交也不会改变已有镜像的源码身份。不能把 v8 image 宣称为从公共 main 或本文档 commit 构建；后续效率版须另行构建并绑定自己的实际 commit、manifest 与 image。

parent 必须预先存在；release 层不会在线 pip、拉取或升级引擎。系统 Python、vLLM/NPU 栈和 `/opt/saea-venv` 继承 parent，构建中重定向项目 editable hooks，清理旧业务目录。两个解释器应核实际 import 路径与文件 hash，不能仅凭 PYTHONPATH 声称版本正确。parent 底层供应链尚不能只由本仓全量重建；这里提供的是基于固定现存 parent 的可重建 release 上下文。image ID 是本地内容身份，不能当已发布 registry digest 或可用的公共下载地址。

## 从已提交源码准备构建

以下命令都是替换变量后的操作说明，本轮文档交付不执行构建。使用已提交 Git 对象，不混入 dirty helper：

```bash
python3 docker/saea-base/release/context.py \
  --repo "$CHECKOUT" --commit "$RELEASE_COMMIT" --out "$NEW_CONTEXT"
```

`context.py` 输出实际 commit、文件数、manifest SHA256；上下文包含同 commit 的 Dockerfile、校验器与构建脚本。检查 allowlist 和选入内容，不放入凭据、私有数据、历史 runs 或请求轨迹。保留上下文清单与构建日志。

将上下文放至获准构建主机的独立新目录 `$BUILD_ROOT/context`。现有 `build-release.sh` 要求 `$BUILD_ROOT` 为 `/var/tmp/omni-unified-release-<unique-version>`，tag 为 `saea-unified-release:<unique-version>`；该前缀是脚本接口，不是既有部署目录。脚本当前固定构建 CPU 240–243、builder 内存 8GiB、IO 权重 100，必须先确认构建主机拥有这些已分配资源，不要直接在任意主机执行。

```bash
bash "$BUILD_ROOT/context/build-release.sh" "$BUILD_ROOT" "$UNIQUE_TAG"
```

构建 detached、无网络、`--pull=false`，保存唯一容器 handle、`launch-custody/container-id.txt`、`build.log`、`exit-code.txt`、`image-id.txt`。SSH 或调用超时后先 inspect 原 handle；未终态不重建。Docker build 成功不等于业务验收。

## 配置与外部资产

[assets.template.json](examples/assets.template.json) 使用 `saea-experiment-assets-v1`：仓库相对 `configs/` 或 `docs/` 路径映射到容器内外部绝对文件路径及其**原始字节 SHA256**。它用于外置片段、模板、roster、冻结来源 receipt；样例六项不是完整依赖闭包。应从真实 workload 的锁、片段和 `preparation_source_receipts` 补齐全部引用，保持原科学内容；数据侧依赖保留 data-root 下原相对路径及 confinement。

读取 manifest 本身的 SHA256 填入 `SAEA_EXPERIMENT_MANIFEST_SHA256`；launcher 的 `required_inputs` 也绑定主机侧同一文件。容器内 `/experiment` 只读。数据和知识是外部资产，不能挂到 `/opt`、`/usr` 等位置覆盖镜像源码或依赖；不能通过外部 manifest 替换 Python 代码。

权限按实际文件 owner 配置：launcher 可选 `user` 是数字 `uid:gid`，不要照抄别的环境身份。原 0600 文件应由匹配身份读取。需要同 hash 私有副本时，使用 [asset-stage.template.json](examples/asset-stage.template.json) 对应的既有 `configured_asset_stage.py --config …`；该脚本仅复制清单所列原字节到独立新目录，目录 0700、文件 0600，重复目录拒绝。输出父目录由部署者为该 UID 配置写权限，不扩大原 ACL，不加入凭据。

## 通用 launcher 与零模型预检

[launch-preflight.template.json](examples/launch-preflight.template.json) 与 [preflight.template.json](examples/preflight.template.json) 沿用已验格式。填写真实 image ID、该 image 的 source commit/manifest SHA、分配的 cpuset、主机目录、原始文件 hashes 和精确 roster 整数条数。两解释器 import、真实 compose、图/模板与媒体 hash 检查均有独立声明；没有真实数据就只能做静态 JSON 检查，不能称业务预检通过。

薄 launcher 取自匹配 release 的 `scripts/environment/configured_job_launch.py`；它在主机上只编排 Docker，业务始终执行镜像内脚本。release 模式不要求主机保留整个 Git repo。下列变量需由部署者填写：

```bash
python3 "$LAUNCHER" plan   --config "$LAUNCH_CONFIG"
python3 "$LAUNCHER" layout --config "$LAUNCH_CONFIG"
python3 "$LAUNCHER" launch --config "$LAUNCH_CONFIG"
python3 "$LAUNCHER" watch  --config "$LAUNCH_CONFIG"
```

`plan` 只描述配置，不代表业务可运行；`layout` 创建获准输出目录；`launch` 在发送前校验 image、完整源码 inventory、mount、输入 pin 和启用的租约检查，持久化 launch intent；重复同身份启动复用原容器。配置变化使用独立新 custody，不覆盖旧 intent、guard 或响应。`watch` 每600秒或容器终态观察。CPU预检成功不授权生成；generation仍需原生 register/admit/preflight、资源、guard/lease和unknown核验。本模板没有 generation 派发配置。

## stage → prepare → register 的 pin 生命周期

[launch-stage](examples/launch-stage.template.json)、[launch-prepare](examples/launch-prepare.template.json)、[launch-register](examples/launch-register.template.json) 均调用已有 `scripts/analysis/intent_slot_workload.py`，无需为每个实验写 launcher。其他既有 workload 使用自身标准入口与配置合同。三个阶段分别独立容器名/custody，冻结 image、源码、workload 与输入 assets；代码只读，可写 `/workspace` 和 `/data` 状态独立，原始 datasets 子挂载只读。

| 阶段 | 输入和输出 pin |
| --- | --- |
| stage | 在空的新 workspace 生成片段/roster，写 stage `complete.json`；记录其文件 SHA，以及返回的 `workspace_manifest.path/sha256`。首次空 workspace 不注入不存在的清单。 |
| verify-stage | 同入口 `verify-stage --repo … --data … --config … --stage-record …`；使用 stage 返回的 workspace pin 和原 stage record，在 prepare/register 前核原始阶段产物。 |
| prepare | 注入 stage 的 `SAEA_EXPERIMENT_WORKSPACE_MANIFEST` 和 `SAEA_EXPERIMENT_WORKSPACE_SHA256`，绑定 stage record 文件 SHA；CPU准备写 supply 和 prepared `complete.json`，保留这两个新 hash。不发模型/FIT。 |
| register | 使用仍匹配的 stage workspace pin、原 stage record 和 prepared record；内部核对后更新 baseline/experiment、readiness，再返回**新的** workspace manifest pin。`readiness=held` 不代表生成获准。 |
| 注册后消费 | 后续读取必须使用 register 返回的新 workspace pin，并重新绑定最终配置/协议/源码。旧 stage pin 在 workspace 已变化时应拒绝；不能重写旧清单使其“通过”。 |

stage record、prepared record/supply、workspace manifest 是不同对象，不能混用 SHA。样例的 `<STAGE_RECORD_SHA256>` 等指完整文件字节摘要；required_inputs 是主机路径，argv/env 的路径在容器内。register 样例在开始时仍用 stage pin，因为它内部先验证阶段状态，再封新清单。已pin的workspace文件缺失或悬空链接必须拒绝；仅未被workspace声明且不存在的路径可以回退到其他合法来源，仍核清单身份。配置变化不允许盲复用不同请求的旧响应。

## queue、observer、score 与服务 plan

[queue.template.json](examples/queue.template.json) 为同一既有 `configured_job_queue.py` 的最小 CPU 队列格式；绑定完整 launcher JSON 的文件 SHA，queue状态与各job custody分开。执行 `watch --config …` 前先验证每个job，未知派发状态不自动重提。队列容器若需Docker socket，使用匹配主机身份和Docker组；socket访问具控制Docker的权限，不能称只读保证。通用launcher不含自定义Docker组字段，需在获准的queue容器部署层配置。

只读 `configured_progress_observer.py watch --config …` 和 `configured_remote_archive.py watch --config …` 沿用既有配置合同，每600秒/终态观察归档。`audit_intent_slot_checkpoint.py CHECKPOINT --output …` 审计真实三阶段链；终态checkpoint行数不等于成功三阶段条数，refusal/finite等分类分开。生成/FIT/教师费用单列，FIT custody文件数不能当实际FIT调用，未知请求成本不能当0。

官方 SLURP 评分入口可由同镜像调用，但仍依赖只读外部固定 Python3.9 环境和 upstream evaluator commit `8eb16545762be97ace75334109d73824217311f1`；外置代码/环境的实际pins和权限须部署者提供。设置既有 `SAEA_SLURP_OFFICIAL_PYTHON` 指向该容器内外部解释器。当前镜像未内含整个官方评分环境，不能称全依赖自包含；已完成封存数据一致性评分不应重复执行以补文档。

服务使用现成 `scripts/npu/npu_deploy.sh` 与外部已验 deployment profile/model fragment，不通过普通CPU launcher虚构服务role。以下只是**plan**操作说明；profile、model和所有变量需部署者提供，禁止复制另一run/port/lease身份：

```bash
SAEA_DEPLOY_MODE=release \
SAEA_RELEASE_IMAGE_ID="$IMAGE_ID" \
SAEA_RELEASE_SOURCE_MANIFEST_SHA256="$SOURCE_MANIFEST_SHA256" \
SAEA_DEPLOY_MODEL_FRAGMENT="$MODEL_FRAGMENT_FILE" \
bash scripts/npu/npu_deploy.sh plan "$DEPLOYMENT_PROFILE_FILE"
```

检查profile image pin、卡/端口/容量、两实例命令和busy-card拒绝。plan通过不代表up或新镜像NPU生成验证；这里不执行up/down，也不迁移正在运行实验。

## 验收与迁移边界

示例本轮只做静态 JSON/字段检查，没有新业务、构建、模型、FIT或教师请求。已有release业务验收不能代替旧版本resume、新效率实现或新镜像真实吞吐验收。旧→release checkpoint兼容桥与后续效率最终版尚有单独验收边界；不得凭一般preflight改旧run身份或复用不等价请求。

将来获准迁移的顺序是 stop admission → drain → reconcile unknown → checkpoint seal → 镜像/配置替换 → 经兼容合同验证的resume。不能假定容器退出即请求终态；保留共享guard、租约、已知响应与全部历史成本，unknown不盲重发。本文不授权停止任何活实验或更换服务。

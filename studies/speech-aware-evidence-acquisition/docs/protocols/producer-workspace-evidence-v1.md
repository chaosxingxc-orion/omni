# producer workspace evidence v1

`ImageReleaseCompatibility` 可选 `evidence.producer_workspace`。它只允许恢复 producer 清单列出的 `configs/` 和 `docs/` 文件；`src/` 永远来自已认证 Git archive。

字段为：`schema`、绝对 `root`、以及 `manifest`、`source_manifest`、`launch`、`container` 四个 `{path,sha256}` pin。manifest 必须是 `saea-experiment-workspace-v1`，固定 `source_commit`、`source_manifest_sha256` 和每个文件 SHA。launch 必须有 `saea-configured-job-launch-v1`、规范化 JSON SHA 与 container label `saea.launch.config` 相等；launch 的 source identity、workspace manifest 环境变量及只读 mount 必须和 container `Mounts` 一致。路径拒绝绝对穿越、symlink、未列清单文件、shadow mount、错误 source commit 或改变字节。

示例（省略具体 pin 值）：

```json
{"schema":"saea-producer-workspace-evidence-v1","root":"/capture/workspace",
 "manifest":{"path":"/capture/manifests/<sha>.json","sha256":"<sha>"},
 "source_manifest":{"path":"/capture/source-manifest.json","sha256":"<sha>"},
 "launch":{"path":"/capture/launch.json","sha256":"<sha>"},
 "container":{"path":"/capture/container.json","sha256":"<sha>"}}
```

`checkpoint_terminal_custody.validate_terminal_custody` 在读取 predecessor 前先验证这些 evidence pins；未知响应、未绑定 producer 或任何校验失败都保持拒绝。


`root` 是明确指定的归档副本根，可与 producer 主机路径不同；必须包含
`manifests/<manifest sha256>.json`，其字节必须等于 manifest pin。
producer 原始路径取自 pinned launch 环境变量，独立核对 launch 与 Docker inspect
中的最长路径只读 bind mount，禁止子挂载遮蔽。不会对归档机器解析已不存在的
producer 主机目录，也不会将归档目录名称当作来源证明。
`source_manifest` 是 **旧 producer** 的 release inventory，不能用 successor 的
`evidence.release_manifest` 替代；其中全部 Git 源码 SHA 必须匹配已认证捕获。

`old_source(run_id=..., predecessor_attempt_id=...)` 可显式绑定 producer 身份；
precheck 与 terminal custody 自动传入。terminal 的 `evidence` 列表必须同时登记上述
四个 pin；workspace 文件由嵌套 manifest 清单认证，无需逐个添加 terminal pin。
未设置 `producer_workspace` 时继续走原 Git-only 路径。

固定镜像内建议从 study 根执行：

```bash
python -m pytest -q tests/unit/test_producer_workspace_bridge.py tests/unit/test_checkpoint_compatibility.py tests/unit/test_experiment_assets.py tests/unit/test_checkpoint_terminal_custody.py tests/unit/test_contact_reconciliation.py tests/unit/test_configured_job_launch.py
```

这些是工程来源验证，不授权终态交接、重发 unknown 请求或改写成本。

## 第二层外部资产（返工扩展）

旧 producer 的解析优先级为 **workspace > SAEA_EXPERIMENT_MANIFEST > Git**。
`evidence.producer_assets` 与 `producer_workspace` 独立可选；两者同时存在时，
`launch`、`container`、`source_manifest` 三个 pin 的 SHA 必须分别一致。
未设置第二层时保留旧 Git-only/workspace-only 行为；显式设置第二层但验证失败时绝不回退。

```json
{
  "producer_assets": {
    "schema": "saea-producer-assets-evidence-v1",
    "root": "/capture/producer-assets",
    "manifest": {"path": "/capture/experiment-assets.json", "sha256": "<原始 manifest SHA256>"},
    "source_manifest": {"path": "/capture/source-manifest.json", "sha256": "<旧 release SHA256>"},
    "launch": {"path": "/capture/launch.json", "sha256": "<原始文件 SHA256>"},
    "container": {"path": "/capture/container.json", "sha256": "<原始文件 SHA256>"}
  }
}
```

该 manifest 原始字节保持不变，schema 为 `saea-experiment-assets-v1`；`files`
以 `configs/` 或 `docs/` 相对路径为键，以 `{path, sha256}` 为值。
`path` 是 producer 容器可见的绝对路径。例如 `/producer/final/docs/receipts/runtime.json`
对应归档文件 `/capture/producer-assets/producer/final/docs/receipts/runtime.json`，
即归档根拼接原路径去前导 `/`。容器路径经 launch/Docker 中一致的最长只读 bind mount
映射到 producer host source；归档机不需要原主机路径存在。路径不可含 traversal、symlink，
同一只读 mount 下有嵌套遮蔽时拒绝。整个 manifest 清单逐项校验 SHA，随后按层优先级合并；
清单外文件不参与恢复。源码 `src/` 始终保留已认证 Git 字节。

manifest 自身路径及 SHA 必须来自 pinned launch 的 `SAEA_EXPERIMENT_MANIFEST` 和
`SAEA_EXPERIMENT_MANIFEST_SHA256`，并与 Docker `Config.Env` 一致；其路径也必须受
只读 bind mount 保护。归档 manifest 由独立 pin 提供，无需在资产归档根再复制一份。
恢复过程不调用当前环境的 `resolve_asset`，不读取当前环境的实验变量。

terminal `evidence` 列表登记两层全部 pin；共享同路径的 pin 只登记一次。
完整 custody 仍需单独登记 runtime receipt 的实际归档 `{path, sha256}` pin，
与 bridge 恢复出的 receipt 字节相等，现有 FIT/session 来源核验继续执行。

## prepare 的嵌套搬迁

`saea-terminal-package-prepare-v1` 的 `bridge_paths` 继续支持原来的平面 pin 映射，
并允许按 evidence 结构递归嵌套。仅列出 pin 和 archive `root`，不要放入 `schema`、
`startup_data_root` 等不搬迁元数据。每个需搬迁分支都必须完整列出，拒绝遗漏或多余字段。

```json
{
  "bridge_paths": {
    "commit_object": {"path": "/new/commit", "sha256": "<保持原 SHA>"},
    "tree_listing": {"path": "/new/tree", "sha256": "<保持原 SHA>"},
    "source_archive": {"path": "/new/source.tar", "sha256": "<保持原 SHA>"},
    "release_manifest": {"path": "/new/release.json", "sha256": "<保持原 SHA>"},
    "source_review": {"path": "/new/review.json", "sha256": "<保持原 SHA>"},
    "reference_pins": {"path": "/new/references.json", "sha256": "<保持原 SHA>"},
    "producer_assets": {
      "root": "/new/producer-assets",
      "manifest": {"path": "/new/assets.json", "sha256": "<保持原 SHA>"},
      "source_manifest": {"path": "/new/source-manifest.json", "sha256": "<保持原 SHA>"},
      "launch": {"path": "/new/launch.json", "sha256": "<保持原 SHA>"},
      "container": {"path": "/new/container.json", "sha256": "<保持原 SHA>"}
    }
  }
}
```

若含 `producer_workspace`，同样提供对应五个字段（`root` 和四个 pin）。
调用方先按原布局复制证据和归档树；prepare 只重定向配置并验证新位置字节，不复制、
不重写原证据、不重签清单。新 pin SHA 必须等于旧 pin SHA；对含外部层的 bridge，
prepare 调用 `old_source()` 认证新根内所有清单文件后才发布 HELD 模板。
输出 bridge JSON 因路径变化产生新的包级 SHA，原证据文件 SHA 均保持不变。
`source_paths` 原有 successor 输入清单搬迁语义保持不变，不得把它产生的新清单当作
历史 producer 的 manifest。

返工新增测试：`test_producer_asset_bridge.py`、`test_terminal_package_relocation.py`；
`test_configured_terminal_custody_package.py` 增加无 Git runtime receipt 的完整 custody 变体。
这些都是离线工程测试，真实 919+2 完整恢复仍需主线程验证。

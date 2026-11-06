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

# 跨版本队列前序身份

队列可在外部 `after` 依赖中配置 `launcher_artifact`，指向创建前序容器时使用的固定 `configured_job_launch.py` 原件及其 SHA256：

```json
{
  "condition": "container-exit-zero",
  "launch": {"path": "/owned/previous/launch.json", "sha256": "<reviewed-launch-file-sha256>"},
  "container_id": "<full-immutable-container-id>",
  "launcher_artifact": {"path": "/owned/frozen-release/scripts/environment/configured_job_launch.py", "sha256": "<reviewed-historical-launcher-file-sha256>"}
}
```

同一字段也适用于既有 `generation-released` 依赖；原 release/seal 核验不变。省略字段仍要求前序由当前 launcher 版本创建，与旧配置行为一致。需要将历史原件以只读方式挂载到队列；配置发布者应从已冻结发布树核验文件哈希，不从正在观察的 receipt 复制自报 hash 作为信任依据。

历史文件只用于版本身份承诺，既不导入、执行，也不调用旧代码。当前队列与启动器继续核验 launch 配置哈希、durable intent、镜像、Docker runner/config 标签和完整容器 ID；只是期望的 runner 哈希来自独立固定的历史文件。launch、readiness、队列内部任务观察仍使用当前启动器。配置与 pinned 文件任何变化均被拒绝；旧未知 launch 不会重新派发。

该机制解决“合法历史前序因为新启动器源文件变化而无法被后继观察”，不支持任意未来 launcher 配置格式。历史配置仍须符合当前 reader 的受支持 schema；若格式不兼容，需要显式 reader 迁移，不能绕过检查。也不恢复丢失的历史文件、不修复未知请求。信任根仍是经审核并冻结的队列配置与文件，哈希本身不证明任意配置具有授权。

实际事故发生在新 `generation_lease` 前置检查升级后：旧 cooking 前序 runner SHA 与新队列当前 SHA 不同，导致 `launch receipt identity changed`，此时没有派发新推理。业务已由旧外层依赖 gate 和新内层 queue 接续。本修复只供后续配置使用，不热修改那些活跃队列。

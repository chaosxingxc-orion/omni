# PSD B / ON K2 successor 准入状态

## 结论

当前不能启动或接续 ON K2 的 K001/K002。8202 的 B TP4 服务健康，B resident preflight 已通过；K2 资产完成 CPU `stage/prepare`，但仍是 held 状态，旧 ON queue `psd-on-e8e11567-v9` 已失败，不能复用或重发。

## 已核实事实

- 181/A800I-03：固定镜像 `sha256:7e60208f...`。
- B 服务：`saea-psd-aligned-b-tp4-v1`，PID `1397553`，端口 `8202`，TP4，`max_num_seqs=128`，`max_num_batched_tokens=2048`，上下文 65536。
- B resident preflight 在固定镜像、只读挂载、host PID 下 exit 0；runtime receipt `62898fa4...`、session receipt `c7a1fbfd...`、live socket 均匹配，0 模型调用。
- 当前 B profile `/var/tmp/psd-b-profile.json` 的 raw SHA 是 `d6cf89b8...`，model fragment 为 `bf16-thinking-psd-b-v1`；K2 campaign v2 与 K001/K002 runtime binding 也使用该 fragment 和 profile。此前拿来对比的 A fragment 文件不是当前 B profile，已排除该误判。
- K001/K002 各 625 条；checkpoint 均 `generation_authorized=false`、`new_model_calls=0`。K001 supply SHA `15e8b239...`，K002 supply SHA `72914a90...`。
- K2 编译、stage、diag 文件目前只存在 `/var/tmp/psd-tfrl-on-ready-v7/`，由 `root:root 0600` 持有；未形成仓库 release 下的 `docs/readiness/<candidate>/candidate.json`、experiment registration 和可供 admission 读取的完整 successor 树。
- 旧 ON queue 已 failed，shared `flight.lock` 仍绑定旧 TFGRPO 任务；不能直接从旧 queue 重发。
- 8200 当前有约 63 个运行请求；8202 当前空闲。本次没有停止或重启服务，没有新模型调用。

## 最小解锁动作

1. 在固定 release 和当前 B profile/runtime/session 身份下，把 K001/K002 的现有 held 资产通过正式配置化注册入口写入唯一 successor candidate/experiment/readiness 树；不重做 roster，不重采样，不改内容哈希。
2. 以该 successor 树运行 `admit_real_engine_candidate.py` 的只读/准入前置检查，确认 candidate、source、policy、checkpoint、runtime receipt、session、profile 和 PID 一致。
3. 在共享 guard 下完成旧 failed queue 的终态与 lease 交接；不删除锁、不盲重发未知请求。
4. 逐一为 K001、K002 生成新的配置化 queue/launcher pins，`monitor_interval_seconds=600`，从 held checkpoint 接续。

在第 1 步完成之前，不能声称“已准入”或让 8202 接收 K2 请求。手工复制 root 文件、修改旧 queue 或复用 A register 都不满足身份和结果可信度要求。

# CPU overlay 评分终态接续

`configured_overlay_score_handoff.py` 是配置化 CPU 接续工具，不参与 NPU 租约或生成队列。以 `configs/examples/overlay-score-handoff-v1.json` 为字段样例；所有零值、路径和 run 身份必须按实际已准入任务替换并固定 SHA。producer launcher 与 launch 必须使用该任务实际的只读 pins；`container_id` 已知时填精确 ID，尚未启动时可留 null，脚本从原 launcher 的 intent/result 和 Docker 身份读取，不猜 ID。

部署使用完整独立评分工具 release、原生成 release、固定镜像与可用的官方 Python。`worktree` 是 host 上干净 git worktree，仅满足现有 launcher 的 clean-git preflight；其提交与 `worktree_commit` 必须一致，不代替容器内原生成 release 或 `/score-tool`。已评 run 的 score-ready 仅核对 run/attempt/candidate/1080 及 scores 附件 digest，表示可以进入独立 CPU 正式验收，不等于完整科学验收。仅已知 overlay failure 可调用现有恢复器，它会独立验证两套 release 和 sealed bundle。

以私有持久 state 目录执行 `python3 configured_overlay_score_handoff.py watch --config /absolute/path.json` 并脱离 SSH；同 run 只能启动一个 adapter。轮询600秒。检查 `status.json`：`waiting-producer` 等原 worker 终态，`recovery-running` 表示原生 launcher 已提交，`score-ready` 为附件身份已核；`held` 要人工核对保留的 intent、失败证据和现场，不能清理后盲重试。配置文件与 launcher receipt 不得改写，score-ready 不重新评分。实际生成 release 与 CPU 评分互不依赖。该目录只能写入业务自有本地盘，原运行树不热改。

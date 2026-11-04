# 配置化进度观测与队列交接

两个入口都由固定镜像内的 `configured_job_launch.py` 启动。只通过配置增加实验，不改已运行队列及其冻结工具树。

## 进度观测

`configured_progress_observer.py watch --config /config/progress.json` 每 600 秒原位更新 `state_dir/status.json` 和心跳。`once` 用于一次验证。配置顶层是 `schema: saea-configured-progress-observer-v1`、`state_dir`、`interval_seconds: 600`、`lanes`。

每个 lane 声明 `id`、`cards`、`instances`、`jobs`。每个 job 声明 `id`、`kind`、`launch`、`launcher`、`source_plan`、`logical_data`、`arm`、`total_rows`、`roster`、`learner_job`。文件引用用 `{path, sha256}` 固定。

- `row-checkpoint` 使用固定 `roster`，`learner_job: null`。复制当前 checkpoint 有限前缀、验证 hash chain/样本范围后计数，临时副本读取后删除。
- `learning-updates` 使用 prepared `complete.json` 作为 `learner_job`，`roster: null`。目前支持 ACE/Reflexion，通过 batch、连续 update、learner response 和 native journal 桥接及确定性 CPU 重放统计完成组数。
- 数量不是 calls/2。进程 EXIT0 仅报告 stopped；完整科学评测仍需独立 seal/scorer。
- 已知生成 token 成本、未知请求数、失败成本分别报告；FIT 仅统计 custody 条目，其 endpoint 成本另计。源样本音频长度不冒充实际传输音频成本。
- 分配卡数不是实测利用率；同 lane 各 job 不重复累加卡数或独立样本数。

观测器只读挂载源数据，输出目录须独立。一个 job 读取失败报告 observation-error，不伪造零进度，也不阻塞其他 job。GEPA 格式尚未接入，不能套用 ACE 计数。

## 独立后继交接

`configured_queue_handoff.py watch --config /config/handoff.json` 等待旧队列终态并验证安全资源交接；`plan` 不修改状态，`once` 执行一次检查。

顶层字段：`schema: saea-configured-queue-handoff-v1`、`queue_config`、`queue_tool`、`watch_launch`、`watch_launcher`、`settlements`、`state_dir`、`monitor_interval_seconds: 600`。前四个文件均固定 `{path, sha256}`；`settlements` 将每个 generation job ID 映射到其固定 settlement 配置。

必须固定旧队列实现。旧 watcher EXIT0 可能对应 failed state，不能单凭退出码交接。所有 generation 容器、原始请求、release 或已完成 settlement、同 inode guard、无活动 lease 与服务 quiescence 都要核验。未知请求、pending/blocked job 或 guard 替换阻止放行。

后继配置以 barrier 的确切 container ID 作为 `container-exit-zero` 前置条件，仍须自己的科学准入、readiness 和 lease。该证明不恢复失败实验，也不表示算法有收益。

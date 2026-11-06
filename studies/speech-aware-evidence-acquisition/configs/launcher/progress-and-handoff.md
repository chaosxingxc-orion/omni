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

## 输入可读性盘点

共享存储异常时，先用既有启动配置检查全部挂载根与required_inputs：

```bash
python scripts/environment/configured_input_inventory.py --config /config/launch.json --output /state/input-inventory.json
```

只读探测、逐项保留errno，文件pin流式计算完整SHA256；报告不包含环境变量值，不启动任务。输出必须是新文件；存在失败返回1，原报告不覆盖。目录检查仅验证目录可遍历，普通未pin文件只读首字节；不能据此声称整个数据集已缓存、完整或READY。原始资产路径可能含内部信息，公开只发布脱敏汇总。

真实构建配置98个挂载根有26个EACCES；消费配对配置122个有36个EACCES。两者包含继承的历史run挂载，后续应按manifest依赖分析确认最小集合，不能仅因本次未直接使用就删挂载。固定镜像3项针对性回归通过，真实盘点零模型调用。

## 只读基座的框架日志与启动检查

使用完整只读源码的工作流任务，在启动配置中显式声明：

```json
"framework_runtime": {"log_directory": "/state/framework/logs"}
```

`/state` 必须映射到项目所属的本地可写目录，并在 `allowed_create_roots` 范围内。
源码和依赖目录继续只读，不允许把 `logs` 子目录挂进源码树。
启动器通过 openJiuwen 公共日志配置 API 设置目录，然后导入实际业务工作流，
最后执行原 `argv` 中的固定源码脚本。日志、完整源码、科学配置分别管理；
无需改依赖镜像或实验业务实现。未设置此字段的任务保持既有入口。

先使用与正式 worker 一致的镜像、源码、用户和挂载配置运行：

```bash
python scripts/environment/configured_job_launch.py layout --config /config/worker.json
python scripts/environment/configured_job_launch.py runtime-check --config /config/worker.json
docker wait <container_name>-runtime-check
docker logs <container_name>-runtime-check
```

检查会写框架日志，并导入真实工作流；强制 `network=none`，不会执行实验入口、
预留 attempt 或发送模型请求。返回 `launched` 只表示检查容器启动，必须等待
exit 0 和 `framework_runtime_ready`。失败检查保留容器、intent和输出，不原地重提。
该检查不证明完整业务路径、断点接续或推理吞吐通过；正式 launch 仍做自己的身份、
输入、输出冲突和共享 guard 检查。运行中的旧任务不受影响。

# 配置化终态交接

## 目的与边界

`configured_terminal_settlement.py --config /config/settlement.json` 验证一个已终止的学习生产者，保留过期租约并输出可追溯的 `complete.json`。工具不启动模型、不重发请求、不改 attempt、不授权科学结果，也不替后继获得租约或 readiness。

支持 `learning_strategy_job.py --config` 的 `run` 模式，以及已注册的严格 `gepa_mutation_job.py run` 参数契约；两者限定 `prepared-context-text`。恢复 attempt、panel lease、缺失 launch-result 的不确定启动均须另行适配；不可用宽松默认值绕过。

## 配置

所有 pin 均为 `{"path":"/absolute/file","sha256":"64位原始文件SHA256"}`，禁止软链接。顶层字段必须完整且没有额外字段：

| 字段 | 来源 |
| --- | --- |
| `schema` | `saea-configured-terminal-settlement-v1` |
| `producer_launch` | 前驱 `configured_job_launch` JSON pin |
| `producer_launcher` | 实际启动前驱的 `configured_job_launch.py` pin；匹配 launch intent 与 Docker label |
| `handoff_verifier` | 公共 `scripts/npu/generation_release_handoff.py` pin |
| `source_plan` | 前驱 host repo 下 `docs/readiness/<candidate>/generation-plan.json` pin |
| `runtime_receipt` | 对应 resident session 的 deployment receipt pin |
| `arm` | `prepared-context-text` |
| `state_dir` | 独立、本地持久盘的输出目录，不得占用运行数据目录 |
| `generation_release` | `{path, logical_data, expected}`；`path` 为 host 上 release 文件路径，`logical_data` 为前驱容器数据根 |

`expected` 必须包含 `candidate`、`run_id`、`attempt_id`（当前为 `learning-v1`）、`study_commit`、`profile`、`service_session`、`rows`。数据映射继承生产者的显式 mounts，不重复手填容器 ID 或 PID。

交接容器应使用固定镜像、host PID/network、Docker CLI/socket；按相同绝对路径挂载 producer 配置和证据，`locks` 与交接输出可写，其余证据只读。启动仍通过现有配置化 launcher。

## 判定与断点恢复

1. 前驱配置、launch intent/result、Docker ID/image/labels/environment、退出状态逐项一致。没有启动结果不推断成功。
2. 固定 generation guard inode 下验证 registered plan、manifest、terminal attempt 和每个原始请求/响应；未知请求、孤立响应、次数不一致均阻断。
3. 存在租约时核验真实 lease PID 已死、run/attempt/head/session 与容器存续时间匹配，再验证 resident 队列和 socket 均空。
4. 没有租约时验证真实 `generation-released.json`；若历史回调失败未发布release，仅允许严格同步artifact清算分支：真实FINALIZED attempt、stage/hash完整flight receipts、COMPLETED artifact manifest、完整数据行/成本、learner与native响应显式request-id桥接、候选和轨迹文件全部一致。异步task或缺失证据仍阻断，不以“无锁”替代证明。
5. 先 fsync `audit.json`，原子 rename 过期租约为共享锁目录中的 `settled-flight-<config hash>.json`，fsync 目录，再写 `complete.json`。同配置重入恢复未完成步骤；已完成记录返回原结果，不触碰后继租约。

同步artifact分支不追补历史release、不修改attempt、不再次调用模型。它只记录 `mode=finalized-artifact` 的audit/complete，保留producer原EXIT状态和 `scientific_success=false`；候选是否提高任务精度必须由后续独立评测证明。learner envelope文件hash与native journal文件hash不是同一概念，不得直接比较；通过receipt_id/request_sha256/response_sha256逐条关联。

## 与队列的关系

失败交接 DAG：前驱设置 `on_fail=continue`；交接节点设置 `after_terminal=[前驱]`；后继设置 `depends_on=[交接]`。交接成功不抹去前驱失败。

正常生成的提前交接继续使用队列既有 `generation-released` 依赖。`grounded_synthesis_job.run` 在 campaign 的 generation-release callback 内发布 seal，CPU 评分尚可继续；不能强制等待容器退出才给下一轮 NPU 工作。终态交接是失败/最终清理工具，不替代这条异步路径。

## 验证与报告

固定镜像运行 `tests/unit/test_configured_terminal_settlement.py`，同时回归 launcher、queue、terminal lease。保留 launch JSON、镜像/commit/hash、容器终态、JUnit、完整日志及失败尝试。通过单元测试不等于已在生产 lane 执行，更不代表算法收益。

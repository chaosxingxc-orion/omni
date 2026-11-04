# 配置化生成前置检查

新生成启动配置可声明 `generation_lease: {"lock_directory": "/owned/lane/locks"}`。目录必须与该配置 `SPEECHRL_DATA_DIR` 的实际最深挂载映射一致，不能检查无关空目录。既有配置兼容；只在新发布配置启用，不修改已冻结运行树。

`plan` 报告 `clear`、`lease-held`、`unreconciled-metadata`、`unfinished-engine-transition` 或 `unreadable`，允许在前序任务运行时准备后继计划。队列先验证依赖，再在 dispatch intent 前要求状态可启动；launcher 在 Docker launch intent 前再次检查。guard 文件存在本身不会阻塞；已有容器/未知启动意图继续走原有对账路径，不因新增检查被重复启动。

检查以只读 fd 短暂尝试同一 guard inode 的非阻塞 flock，不创建、删除或替换 guard，不读取泄露 lease token，不清理 metadata。死 PID 也不能证明请求静止。该检查是早期诊断，不能消除检查至启动间的竞争；runtime 原子 lease 仍是最终互斥权威。当前运行队列不热更新。

失败需要已有 PID/socket/contact quiescence 审计与原 metadata 保留；不自动重试 unknown。队列以 `held`（CLI 非零退出）记录这种已知前置阻塞，不记为未知请求。完成审计修复后再次执行同一队列可接续未 dispatch 的任务，不生成新的模糊启动意图。

## 冻结树接续

已有 `scripts/analysis/resume_registered_workload.py` 支持相同注册 candidate、相同冻结树、指定 `--source-attempt` 和全新 `--target-attempt`。它调用 `native_recovery_receipt.prepare`、共享恢复准入和原 Flight，保留历史成本，未知 contact 不自动重发。必须先有明确 ABORTED custody、可验证原始 ExecutionPlan/来源与已清算的共享租约；没有 source attempt 状态时不能冒造恢复凭据。

配置化 argv 示例（替换为已核验路径/身份，非现行任务启动指令）：

```json
["/opt/saea-venv/bin/python", "/repo/scripts/analysis/resume_registered_workload.py",
 "--repo", "/repo", "--data", "/data", "--candidate", "registered-candidate",
 "--source-attempt", "generation-v1", "--target-attempt", "generation-recovery-v1",
 "--out", "/data/work/codex/registered-candidate-recovery-v1"]
```

仅改变新 launch 配置的名称、receipt/output 和 attempt，不需要复制模型或整棵树。名单/科学契约改变需要新发布，不能利用恢复绕过来源审核。

本次两个失败 v2 的实际共享 attempt 路径均不存在：错误发生在 attempt 建立前，不能直接走要求 ABORTED 的标准 resume，也不能编造 ABORTED。若核验原始身份下 attempt、run、journal 均未建立且所有请求已知未发，清算共享 lease 后可用原冻结树/注册身份执行尚未开始的首次 generation，新建 launch/receipt/out 身份保存失败启动历史；若不能证明该条件则停止并补证。单纯为了绕过旧 launch intent 不需要复制整棵树并重做 material/stage/prepare。当前活跃 v3 不回退、不重启。

## 新启动的独占输出检查

新配置可声明`fresh_outputs`，列出**容器内**必须尚不存在的输出路径，例如：

```json
{"fresh_outputs":["/data/work/codex/job/generation-new", "/telemetry/timing-generation.jsonl", "/telemetry/timing-generation.jsonl.config.json", "/data/attempts/RUN/generation-v1.json", "/data/runs/RUN/generation-v1"]}
```

名单根据实际启动入口填写：首次生成要求新的attempt/run；正式resume只列目标attempt/run，不列已存在的来源attempt。启动器按最长目标路径匹配到真实可写bind mount，验证所属目录，集中报告所有已存在路径（含悬空链接）。在写launch intent和调用Docker前拒绝，不创建、删除或归档这些路径。已有容器或未知launch intent继续原对账路径，不被fresh检查误触发重复运行。

该检查为可选配置，旧冻结任务不变；旧配置未声明则没有此项覆盖。它是启动前诊断，不能消除检查后的并发竞争，运行时O_EXCL、attempt与lease仍为最终权威。不要因为path冲突自动删目录或重发请求。实际恢复中出现“仅preflight/preprocess receipt已写、attempt尚未建立”的状态时，需要同guard下证明终态、无请求/准备产物、服务静默，原目录完整归档后再接续；不伪造ABORTED，也不反复重prepare。

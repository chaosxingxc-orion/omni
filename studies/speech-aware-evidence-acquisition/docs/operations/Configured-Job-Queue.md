# 配置化任务队列

`scripts/environment/configured_job_queue.py` 是现有 `configured_job_launch.py` 的有限队列调度层。任务差异放 JSON；算法不复制启动器。CPU 数据准备、评分、分析可以独立运行，不需要进入这条推理队列。

## 配置与启动

参考 `configs/environment/configured-job-queue.example.json`。示例路径和全零摘要均为占位符，部署时替换为真实文件 SHA256 和完整容器 ID。每个任务固定 `id`、启动配置文件 pin、可选预检 binding pin，以及可选 `after`。任务默认按数组顺序串行。`after` 用于已经运行的前驱，必须同时固定其 configured-launch 配置与真实容器 ID。队列内默认前一任务仍需退出0；需要生成/评分重叠时，配置明确的 `generation-released` 前驱依赖，而不要将仍在评分的前驱重复作为队列头任务。

```bash
python3 scripts/environment/configured_job_queue.py plan --config /srv/project/control/queue.json
python3 scripts/environment/configured_job_queue.py once --config /srv/project/control/queue.json
python3 scripts/environment/configured_job_queue.py watch --config /srv/project/control/queue.json
```

`plan` 校验配置和启动预检，不创建队列状态；启动目录须已通过原启动器 `layout` 准备。`once` 执行一次调度或观察；`watch` 对终态依赖通过 `docker wait <完整ID>` 等待；对生成释放依赖通过 Linux inotify 监听发布目录，事件立即唤醒，每600秒兜底复核。正式 `watch` 应置于现有脱离 SSH 的服务/容器中运行。队列自身不新增 SSH 操作或实验专用 launcher。

## 一致性与失败处理

状态目录使用稳定 `queue.guard` flock 保证单调度者，`state.json` 原位原子替换并 fsync。状态固定原始配置字节、队列/启动器/预检模块摘要和每个容器 ID；更改配置或实现会拒绝旧状态。启动前记录持久 intent，实际 Docker 启动仍由原启动器处理其 guard、intent 和同名容器核对。

容器退出码非零立即停止后续任务，不重试。丢失启动响应时，只有原容器身份与原 launcher intent 能证明启动已发生，重启调度器才可继续观察；无容器且有 intent、PID/容器身份漂移、Docker状态不明均保留并停止，不重发。若崩溃发生在队列 intent 与原启动器 intent 之间，也保守停止，须核对后创建新明确恢复任务。不要删 intent 或改旧状态伪造成功。

## 结果边界

每项在 `state.json` 保存状态、精确容器 ID、观测和退出结果；退出0只表示容器成功结束，科学成功必须检查共享 runtime 产物。支持两种显式依赖：`container-exit-zero` 保持旧行为；`generation-released` 在生产者仍做CPU评分时核验释放事件后即可启动后继。此时退出码或评分失败不能被当作科学成功，后继仍需自己的共享租约和准入。无释放事件的终态容器保持停止，不推断已安全释放。

## 生成释放配置

顶层可选 `handoff_verifier:{path,sha256}` 指向既有 `scripts/npu/generation_release_handoff.py`（固定其原始摘要）；不复制 verifier，不导入业务运行时重依赖。`after` 结构：

```json
{
  "condition": "generation-released",
  "launch": {"path": "/srv/project/control/producer.json", "sha256": "<真实SHA256>"},
  "container_id": "<真实64位容器ID>",
  "release": {
    "path": "/srv/project/state/producer-output/generation-released.json",
    "logical_data": "/data",
    "expected": {
      "candidate": "producer", "run_id": "SAEA-PRODUCER", "attempt_id": "generation-v1",
      "study_commit": "<真实40位commit>", "profile": "profile", "service_session": "session", "rows": 312
    }
  }
}
```

生产者启动配置须有明确 `--out`、run/attempt observability env 和 deployment profile。队列按其实际mount将逻辑数据路径映射到本地持久盘，核验输出目录、`started.json` commit及存在的job/attempt/rows字段，再调用共享 verifier核验完整sealed manifest、score ledger、score jobs摘要与seal receipt。释放事件需精确声明generation sealed/lease released为true、scoring complete为false。接受后记录原文件SHA；后续任何变化都会停止调度。没有写回或改造生产者证据。

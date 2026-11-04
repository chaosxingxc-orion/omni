# 本机外部教师持久化队列

外部教师只由本机访问。181执行已交付候选的NPU队列；本机离线时新的教师请求等待，不声称学习仍继续。脚本只编排已有 `external_exploration.py` / `tfgrpo_external_reference.py`，不重写请求协议，也不自动批准候选。

## 配置与启动

先使用算法自己的prepare命令生成独立prepared目录。每项任务绑定CLI与prepared文件SHA256；不要在配置写API key、代理认证或环境快照。

```json
{
  "schema": "saea-local-teacher-queue-v1",
  "queue_id": "rsi-teachers-v1",
  "state_dir": "/absolute/private/teacher-queue-v1",
  "max_parallel": 2,
  "jobs": [{
    "id": "gepa-proposal",
    "kind": "external-exploration",
    "cli": {"path": "/absolute/repo/scripts/analysis/external_exploration.py", "sha256": "<actual SHA256>"},
    "prepared": {"path": "/absolute/prepared/gepa/prepared.json", "sha256": "<actual SHA256>"},
    "depends_on": []
  }]
}
```

`kind=tfgrpo-reference`绑定 `tfgrpo_external_reference.py`。各算法自己的模板、动态轨迹与预算仍由prepared manifest验证；队列不修改它们。不能将同一prepared加入两项任务。依赖DAG拒绝未知节点与环。

```bash
python scripts/environment/configured_local_teacher_queue.py start --config /absolute/queue.json
python scripts/environment/configured_local_teacher_queue.py status --config /absolute/queue.json
```

start创建独立session的supervisor和worker，启动shell/SSH退出不终止它们。状态目录必须0700，持久化文件0600并fsync；真实PID同时绑定boot ID和start ticks。进程完成事件推进队列；接管原有活进程时每600秒观察一次。已完成任务核验collected哈希后跳过，失败独立任务不会阻止其他独立节点；其依赖后继保留pending。

## 恢复边界

未知launch/dispatch不会自动重发。进程仍活跃时禁止collect；已死且无完成回执时保持unknown-held。仅执行已存在CLI的collect作零调用对账：

```bash
python scripts/environment/configured_local_teacher_queue.py reconcile --config /absolute/queue.json --job gepa-proposal
```

collect失败保留failed-held，禁止自动run重试。多阶段参考算法的部分batch接续仍需原算法audit及显式新编排身份，不把失败自动当可重试；原request guard继续保护未知请求。现版本不自动恢复掉电后尚未开始的launch窗口，需要操作者核验，宁可hold也不重复付费。恢复后start同一配置继续其他pending任务；配置/runner字节变化会拒绝旧队列身份。

## 候选交付

```bash
python scripts/environment/configured_local_teacher_queue.py export --config /absolute/queue.json --job gepa-proposal --output /absolute/private/handoff-v1
scp -r /absolute/private/handoff-v1 npu2:/var/tmp/owned-new-handoff-v1
```

export核验已完成collected并生成manifest；参考算法另带candidate原文件。接收端须对每项SHA复核，随后使用原来源审计、review/release/material流程。包不是完整source archive，不可直接部署，不可覆盖活树；完整prepared、wire与来源轨迹仍按现有私有归档方案保留，校验失败不可晋升。传输失败可以重传同一不可变包到独立临时目录，不重发教师请求。

## 验证

固定saea镜像、network none中11项零模型调用测试覆盖detached、已完成重启、worker中断/活child、unknown只collect、失败分支隔离、配置篡改和秘密不落盘。没有真实教师请求或新NPU实验；该测试证明编排进程行为，不证明供应商可用性或研究收益。

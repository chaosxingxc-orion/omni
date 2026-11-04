# 本机外部教师持久化队列

外部教师只由本机访问。181执行已交付候选的NPU队列；本机离线时新的教师请求等待，不声称学习仍继续。脚本只编排已有 `external_exploration.py` / `tfgrpo_external_reference.py`，不重写请求协议，也不自动批准候选。

## 配置与启动

先使用算法自己的prepare命令生成独立prepared目录。每项任务绑定CLI、prepared和implementation manifest文件SHA256；不要在配置写API key、代理认证或环境快照。

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
    "implementation": {"path": "/absolute/implementation.json", "sha256": "<actual SHA256>"},
    "prepared": {"path": "/absolute/prepared/gepa/prepared.json", "sha256": "<actual SHA256>"},
    "depends_on": []
  }]
}
```

先生成实现清单，再将清单SHA加入配置：

```bash
python scripts/environment/configured_local_teacher_queue.py pin-implementation --cli /absolute/repo/scripts/analysis/external_exploration.py --output /absolute/implementation.json
```

清单绑定整个第一方 `src/**/*.py`、环境脚本和CLI，覆盖动态导入；依赖库由固定镜像锁定。派发和collect均重新验证清单，文件增加、删除或变化均拒绝。不得修改运行中的源树；新版本使用新队列身份。

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

collect失败保留failed-held，禁止自动run重试。显式再次reconcile生成新的attempt ID，历史intent/result保留在reconciliations目录；成功后更新最新索引。状态判断先检查活worker/child，旧失败回执不能遮住活请求。队列worker锁和prepared共享锁保护并发编排；collect持有原provider request.guard，参考算法仍执行原learner runner.lock。多阶段参考算法的部分batch接续仍需原算法audit及显式新编排身份，不把失败自动当可重试；原request guard继续保护未知请求。现版本不自动恢复掉电后尚未开始的launch窗口，需要操作者核验，宁可hold也不重复付费。恢复后start同一配置继续其他pending任务；配置/runner字节变化会拒绝旧队列身份。

## 候选交付

```bash
python scripts/environment/configured_local_teacher_queue.py export --config /absolute/queue.json --job gepa-proposal --output /absolute/private/handoff-v1
scp -r /absolute/private/handoff-v1 npu2:/var/tmp/owned-new-handoff-v1
```

export包含完整prepared树（请求、wire、usage、候选、模板）及递归解析的全部显式文件pin，包括原始轨迹、知识、源batch与实现清单。源文件原字节保持不变；manifest记录原绝对路径到包内文件的映射。接收端验证SHA、完整文件集合和所有JSON文件pin的闭包：

```bash
python scripts/environment/configured_local_teacher_queue.py verify-handoff --output /absolute/private/handoff-v1
```

包内文件权限0600；认证字段、环境快照与任意HTTP headers被拒绝。仅允许历史headers.json中content_type/http_status/request_id三项响应元数据。校验失败不能发布部分包。用于回放时通过manifest映射恢复原路径或挂载到原路径，不能改写prepared中的pin。包仍需原语义review/release/material流程，不自动批准候选，不覆盖活树。传输失败可以重传同一不可变包到独立临时目录，不重发教师请求。

## 验证

固定saea镜像、network none测试覆盖detached、已完成重启、worker中断/活child、unknown只collect、失败分支隔离、导入实现篡改、多次显式对账、原provider互斥、递归来源闭包及凭据拒绝。另外使用真实已完成TF-GRPO prepared副本通过队列collect核对91次历史响应，要求collected和所有历史文件哈希保持不变、0新增请求。测试证明编排与回放行为，不证明新的供应商请求可用性或研究收益。

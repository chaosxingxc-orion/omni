# 三队列状态机原型

状态：CPU 原型已验证，尚未接入 live runner，也未授权 NPU 实验。

`reproduction/ojw/queue_state.py` 定义了独立的 `prepare → ready → generation` 状态机。只有 `complete_prepare` 能把任务放入 ready 队列；generation 只能消费指定阶段、实例和 prefix family 的 ready 项。已进入 generation 的任务取消后变为 `draining`，必须等待原请求终态，不能重新提交未知请求。checkpoint 保存所有任务状态、队列顺序和 inflight 集合，restore 会拒绝未知任务、队列状态不一致或 generation/inflight 不一致。

181 固定 `saea-base:20261027` 镜像中纯 CPU 测试 **3 passed**。当前实现没有替换 `NodePipelineCoordinator` 或 live runner，避免在尚未验证 barrier、公平性和实际异步 callback 语义前引入死锁。下一阶段应先用 fake callback 做 runner 适配测试，再以固定小批量 smoke 验证 queue 深度、prefix 命中、取消排空和断点恢复，全部通过后才可排入真实 NPU 队列。

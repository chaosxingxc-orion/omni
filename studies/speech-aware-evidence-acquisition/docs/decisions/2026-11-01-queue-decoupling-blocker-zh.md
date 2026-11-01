# 三队列解耦阻塞记录

状态：未合入生产代码。

现有 `NodePipelineCoordinator` 已支持有限 CPU prepare worker 和模型 contact slot，但它依赖全体成员的阶段 barrier，尚未形成独立的 `prepare queue → ready queue → generation queue`。我在 CPU 侧尝试加入 ready FIFO 和队列深度观测后，于 181 固定镜像运行 `test_native_node_pipeline.py` 与绑定测试；测试进程持续约 3 分钟未产生终态，随后受控停止。该改动已撤销，未进入提交、远端或运行中的 NPU 树。

阻塞原因是 ready FIFO 需要同时处理同一成员的阶段推进、过期队列项、组 barrier 和取消排空；简单把 pending 扫描替换为 FIFO 会造成等待条件无法满足。下一版必须先为队列定义显式状态机和公平规则，再用纯 Python fake callback 验证：

- prepare 完成才能进入 ready；P→S 依赖不能被历史答案绕过；
- generation 只消费当前 barrier 且保留实例/prefix 亲和；
- 取消会排空已启动 worker，未知请求不重发；
- checkpoint、失败传播和所有队列深度在恢复后可重建。

当前旧版骨架保持不变，未宣称三队列已完成；候选生成和 VerifiedImport 修复可独立使用。

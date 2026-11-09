# 四路12小时滚动运行记录

更新时间：2026-11-09 20:26 UTC。四个TP4实例（181 A/B，175 C/D）保持 detached 运行，16张卡、每路64并发；CPU准备、评分和归档由独立进程异步执行，监控周期600秒。

当前权威状态见 `delivery/s-single-intent-execution-v1/rolling-state/` 和 Wiki 的 `S-Rolling-Experiment-Queue.md`。窗口从当前运行时起连续12小时，到期停止新准入并排空在途请求。当前READY库存按实测吞吐估算约12小时，仍标记为 `NOT_YET_PROVEN`，直到窗口实际完成。

已处理的调度问题：B路预留不足已通过配置化lane分配修正；C路一次启动名冲突确认 `model_calls=0`，保留原blocked证据后以独立attempt恢复；D路未知终态继续执行不重发。所有任务仍使用固定镜像，真实数据集和固定配置可复用。

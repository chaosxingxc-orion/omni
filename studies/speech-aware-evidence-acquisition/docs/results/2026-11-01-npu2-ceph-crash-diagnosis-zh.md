# npu2 两次故障：Ceph fsync 内核异常

核查时间：2026-11-01 08:34 UTC。**两次持久化内核日志均指向同一 Ceph 客户端空指针异常；不是仅凭串口描述推测，也不是已证实的 NPU 故障。** 尚无 vmcore，不能从调用栈单独确定具体补丁或排除所有底层原因。

## 两次相同证据

| 项目 | 第一次 | 第二次 |
| --- | --- | --- |
| UTC时间 | 2026-10-31 10:23:49 | 2026-11-01 05:01:13 |
| 内核 | 5.15.0-25-generic #25-Ubuntu | 相同 |
| 异常 | kernel NULL pointer dereference，地址0x8 | 相同 |
| 错误码 | Internal error: Oops: 96000004 | 相同 |
| PC | unsafe_request_wait+0x1d0/0x564 [ceph] | 相同 |
| 调用路径 | ceph_fsync → vfs_fsync_range → do_fsync → __arm64_sys_fsync | 相同 |
| 当前任务 | python，CPU243 | python，CPU200 |
| 后续 | Ceph连接丢失记录 | 05:01:39、05:02:03 Ceph工作线程soft lockup，26秒/48秒 |

第二次锁死工作线程为`ceph-msgr/ceph_con_workfn`，栈包含`queued_spin_lock_slowpath → __unregister_request → handle_reply → mds_dispatch`。两个日期的原始日志已只读保存到本地受控证据目录；公开文档不包含主机地址、内核寄存器或完整日志。

## 当前主机与实验状态

当前boot ID已改变，08:17:10 UTC启动；旧学习进程不能跨主机重启存活。8张910B4-1均Health OK、无NPU计算进程。当前启动日志未见新的同类Oops/soft lockup/OOM或明确硬件错误；这不是长期健康保证。`/var/crash`、pstore及systemd coredump目录未发现转储，命令行没有crashkernel预留；未修改kdump或sysctl。

恢复attempt `learning-format-recovery-v1`留在OPENED，新增原生调用0，原生请求意图/响应均0；学习桥接记录46条意图、43条已复用响应。原146次模型响应和12次更早历史调用仍保留。不能把桥接意图当作新的未知模型调用。原租约仍在，后续要依据新boot与完整来源核对后交接，不能直接删除锁重跑。

旧两个TP4服务随主机故障退出。本轮只恢复原服务容器；新的PID/boot运行凭证必须重建，不能继续冒用旧驻留收据。学习和H2没有自动重新提交。

## 判断与处理

最直接的故障域是**主机内核Ceph客户端处理fsync时的异常与随后锁竞争**。Python是触发系统调用的进程，不能据此认定Python/模型代码破坏内核。未找到对应修复提交前，不命名CVE或承诺升级到某个版本必然解决。Ceph容量93%作为容量风险另列，不把它当作已证实的空指针根因。

近期避免重放相同高频Ceph文件/目录fsync负载。不能删去fsync来绕过断点可靠性：工程方案是把热请求日志、checkpoint和原子提交放在本地持久盘，CPU异步同步到Ceph；每条已提交记录保留哈希，崩溃恢复以本地完整提交为准，共享盘副本标明同步水位。镜像、模型和数据仍可只读共享。该方案尚未实现和验证，不称已解决主机重启恢复。

持久修复需要运维核查Ceph客户端与当前旧内核的已知修复、Ascend驱动支持矩阵，选择兼容内核并安排重启；启用kdump/串口持久日志也需单独规划。未擅自升级内核、驱动、固件或更改共享挂载。用户串口原文仍可用于进一步与日志签名对齐。

# 热状态本地持久化与独立客户端归档

## 当前生效的存储规则

权威修复结论：/home/xingchao/workspace/multimodal-jev/docs/engineering/runtime-storage-policy.md；175/181同一Ceph内核故障尚未修复。禁止受影响主机写Ceph运行态，覆盖应用和底层库；禁止fsync/fdatasync/syncfs/sync/O_SYNC/O_DSYNC绕过。普通共享写入也可能触发，另一线程或容器无效。

热状态必须真实本地ext4/xfs/btrfs，保留文件flush/fsync、原子rename及父目录fsync；不得全局禁用同步。显式配置并检查TMPDIR、XDG_CACHE_HOME、HF_HOME、TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、HOME/ascend/log、ASCEND_WORK_PATH和ASCEND_CACHE_PATH及日志路径。Ceph模型/数据/源码只读挂载。剩余容量必须同时至少20GiB及10%，可用inode至少5%，另满足工作负载预算；运行中检查，不足有序停止。

只能由已修复或已验收的独立客户端拉取已封闭不可变本地提交归档Ceph，记录hash、来源boot/run、提交号和水位，核验后发布完成标识。受影响主机本身不得执行共享归档写入；不把活跃目录快照或close成功当持久备份。归档失败保留本地权威副本。清理优先/var历史非活动项目资产，保护活动热目录、锁和唯一恢复证据。不得全局prune或生产复现故障。内核升级需支持矩阵、维护流程和隔离恢复验收，不自动安装重启。当前旧共享配置及容量不足配置均不得新派发。

当前实施边界：此前175/181上“不fsync的归档进程”方案已撤回；本地服务需满足空间预留，归档客户端尚待验收，不能宣称完成迁移。

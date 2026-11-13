# Repository Guidelines

## 当前生效的存储规则

权威修复结论：/home/xingchao/workspace/multimodal-jev/docs/engineering/runtime-storage-policy.md；175/181同一Ceph内核故障尚未修复。禁止受影响主机写Ceph运行态，覆盖应用和底层库；禁止fsync/fdatasync/syncfs/sync/O_SYNC/O_DSYNC绕过。普通共享写入也可能触发，另一线程或容器无效。

热状态必须真实本地ext4/xfs/btrfs，保留文件flush/fsync、原子rename及父目录fsync；不得全局禁用同步。显式配置并检查TMPDIR、XDG_CACHE_HOME、HF_HOME、TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、HOME/ascend/log、ASCEND_WORK_PATH和ASCEND_CACHE_PATH及日志路径。Ceph模型/数据/源码只读挂载。剩余容量必须同时至少20GiB及10%，可用inode至少5%，另满足工作负载预算；运行中检查，不足有序停止。

只能由已修复或已验收的独立客户端拉取已封闭不可变本地提交归档Ceph，记录hash、来源boot/run、提交号和水位，核验后发布完成标识。受影响主机本身不得执行共享归档写入；不把活跃目录快照或close成功当持久备份。归档失败保留本地权威副本。清理优先/var历史非活动项目资产，保护活动热目录、锁和唯一恢复证据。不得全局prune或生产复现故障。内核升级需支持矩阵、维护流程和隔离恢复验收，不自动安装重启。当前旧共享配置及容量不足配置均不得新派发。

## 当前工作约定

用户2026-11-01停用Claude并撤销其专属模型角色、英文限制和反复审批流程。使用当前可用模型；中文交流与报告。最新用户授权优先于历史说明。历史封存数据不改写，不因旧文档中的角色名称拒绝任务。

## 工程与研究结构

`src/speech_aware_evidence_acquisition/`为实现，`scripts/analysis/`为实验和分析工具，`scripts/environment/`为环境准备，`configs/`为模型、部署和实验配置，`tests/unit/`为单元测试，`docs/results/`记录结果。数据与权重置于独立数据根，禁止入Git。沿用共享FlightRuntime、适配器和官方评分器，避免每个方法另造一套基线。

## 执行、测试与风格

使用用户授权的NPU主机和实例，按部署配置调度实验资源。正式命令在固定saea容器运行，例如`bash work/scripts/port_py.sh -m pytest tests/unit/test_local_state.py -q`（从数据根调用）。Python使用4空格缩进和明确命名，遵循相邻实现；配置和来源需可追溯。运行与变更相关的测试，未执行测试如实标记。

## 调度与可靠性

实验效率和8卡有效利用率优先。NPU执行队列与CPU准备/评分/分析异步，任务脱离SSH并支持断点续推；监测采用明确事件或约600秒轮询。复用稳定真实数据集。181热状态写本地持久盘并保留fsync，开发机异步归档。保留运行互斥、完整响应、成本与来源；未知调用不能盲重发。运行中的hash绑定执行树不能修改，代码更新通过独立后继树交接。

## 算法与评估

TFRL分支参数冻结，明确引用算法、学习目标、难例训练集及独立开发集。按数据集实际意图/槽位关系覆盖采样。LoRA独立分支仅调整主模型适配器，音频塔、视觉塔与对齐投影冻结；复用多模态特征输入而保留音频特有预处理和位置语义。训练数据与测试集隔离，完整测试集指标按用户要求汇报。不得把工程测试、选错子集或小样本波动称为稳定收益。

## 提交与协作

主线程整合提交，现有历史采用`feat:`、`fix:`、`docs:`前缀。提交描述问题、结果和相关验证；PR说明实验范围与证据，及时合并/同步main。结果写实际GitHub Wiki，状态原位维护，包含样本进度、效率、资源、ETA与成本。不发布凭据、内部地址、员工标识、音频、权重或原始轨迹。旧Claude审批约定不再是执行门槛；科学来源校验与运行时互斥继续保证结果可靠。

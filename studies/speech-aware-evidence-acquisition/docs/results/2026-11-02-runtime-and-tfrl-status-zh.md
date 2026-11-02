# 2026-11-02 运行态与 TFRL 进展

> 历史运行快照，保留结果用于追踪；当前状态与执行优先级以[持续更新的任务状态](Research-Status-and-Plan.md)为准。本文旧owner/审批措辞不恢复已撤销的Claude审批限制；科学来源与运行时互斥仍保留。

## 当前运行态

181（A800I-03）可达，两个 TP4 vLLM 服务仍驻留：`saea-real500-tp4-i0` 和 `saea-real500-tp4-i1`，合计覆盖 8 卡。当前没有运行中的学习、TFRL 或 REAL500 进程；驻留服务占卡不等于任务满载，因此不能报告持续 8 卡有效利用率。

已核验的真实数据任务均为终态，不能重复启动：

| 任务 | 进度 | 资源 | 墙钟/条 | 调用 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| REAL500 TP4-S96 | 500/500 | 8 卡、2 实例 | 2.905 秒 | 1000 | COMPLETED |
| REAL500 TP4-S128 | 500/500 | 8 卡、2 实例 | 2.766 秒 | 1000 | COMPLETED |
| REAL500 TP8-S256 | 500/500 | 8 卡、1 实例 | 4.661 秒 | 1000 | COMPLETED |
| current-seed H1 context-repair | 313/313（310 新生成 + 3 保留） | 8 卡、2 TP4 | 310 行 11.141 秒 | 2170 新调用 | COMPLETED |
| current-seed H2 | 312/312 | 8 卡、2 TP4 | 11.383 秒 | 2184 | COMPLETED |

所有这些任务的行 checkpoint、run manifest、输出和评分文件均存在；未知请求没有被重放。H1/H2 当前完整 625 条合并结果已经进入 CPU 读阶段。

## TFRL 结果与停止结论

current-seed 家族在固定 625 条开发集合上的官方合并结果为：`vote - lock = -0.4040pp SLU-F1`，预登记门槛为至少 `+1.0pp`，因此状态为 `stopped-family`，不允许继续复用同一候选或在同一 H1/H2 上调参。H1 半折差为 `-1.2455pp`，H2 半折差为 `+0.4074pp`；两半方向不一致，不能宣称稳定收益。

这不是最终 test 结果，也不污染冻结的完整 test。`SLURP-TEST-FULL-13078-V1` 仍只在 D/A/best-T 三套方法身份、runtime/session、预算、owner allowance、exposure rows 和 detached queue identity 完整后运行。

## 下一候选的准入边界

已登记的 TFRL-v2 证据条件化消费 rail 仍处于 CPU 设计和 smoke 准备阶段。外部 data-root 中 H1、H2、union 三份 bank receipt 已通过 schema、canonical hash、bank 文件 hash 和 `vllm-tokenize` token 计量核验（readiness=`ready=true`）。这只证明 artifact 可消费，不等于实验已获准入；在完成 4 行 detached smoke、来源/owner/checkpoint 注册前，不能启动 Fold B 的 1,080 条任务。

下一候选只改槽位证据条件化消费，保持 P、知识构建、奖励、模型和解码固定；TRAIN 困难集按原始真实意图—槽位关系冻结，不构造笛卡尔积，不读取 test 标签调优。必须先完成 bank 的真实 vLLM tokenize、来源隔离、4 行 smoke 和 detached checkpoint 注册，之后才可安排 NPU。

## 工程验证边界

公开仓 `main` 已包含 `31935f49`：ready queue 原子 checkpoint 写入/恢复、完整 test fail-closed gate 和 TFRL 单组件协议。当前主机 Python 环境没有 pytest，且直接导入 rails 会触发系统 Python 3.11 的既有 dataclass `mappingproxy` 默认值错误；因此本轮没有把本地解释器异常冒充为 rail 通过。正式 TFRL rail 测试必须通过固定 `saea-base` 镜像执行；镜像当前不在本地，拉取受 Docker Hub 代理认证阻塞。该阻塞只影响本地测试复核，不改变已封存的历史结果或准入结论。

## 效率改进状态

本轮已落地一项可跨实验复用的 CPU 准备优化（`195d5940`）：`data/runtime_manifest_cache.py` 将已经验证过的真实 runtime roster 固化为仅含媒体身份、SHA-256、时长和采样率的 manifest。manifest 绑定 roster/split 身份，采用规范化 JSON、临时文件、`fsync` 和原子替换；读取时拒绝篡改、截断、非规范 JSON、重复样本和路径逃逸。它不读取或保存 transcript、annotation、answer，也不改变模型请求语义。独立 smoke 已验证排序、写入/读取、来源身份校验和篡改拒绝；pytest 需在固定镜像内复核。

仍未实施、且应按优先级推进的项目：

1. 将 manifest 接入实际 queue 输入阶段，并在一个真实 500 样本任务上比较 cold preparation 与 warm preparation；实测前不报告加速倍数。
2. 在固定镜像内复核已绑定的 TFRL-v2 bank receipt，并完成 4 行 detached smoke；当前仍未完成 smoke 准入，不能启动 1,080 行 NPU 任务。
3. 将 CPU 评分、bank distillation、成本归因作为 ready queue 的独立后继 job，保持 NPU 生成和 CPU 分析并发，避免 NPU 空档。
4. 只有方法冻结、准入凭证齐全后，才启动完整 13,078 条 test；此前 test 子样本只作历史汇报，不能用于调优。

容量探测、batch-size 扩展和旧 current-seed 家族均已终态或未晋升，不列入下一轮重复实验。监控保持 600 秒轮巡或明确终态事件触发；下一次 NPU 任务必须报告逐卡利用率、完成量/总量、秒/条、资源和 ETA。

队列监控也已完成一项工程修复（`016388cf`）：`scripts/npu/card_queue.py` 将进程存活/日志终态检查与昂贵的 `npu-smi`/Docker 卡探测分离。等待 gated job 时，卡探测默认 600 秒缓存；每个任务首次进入卡门禁时强制探测，避免前一任务释放卡后盲启动或产生无谓空档。断点 checkpoint、SSH 断链恢复、未知启动拒绝和终态判断保持原语义。公开 CPU 回归现为 **50 项通过**；本地无 pytest，因此未将该结果扩大为完整 study 测试通过。

TFRL-v2 bank 的 CPU 准入诊断由 `scripts/checks/tfrl_v2_bank_readiness.py` 实现。当前命令使用显式 `--data-root /mnt/cephfs/xingchao/omni` 审计返回 `ready=true`：三份 receipt 的 schema、canonical hash、bank 文件 hash 和 `vllm-tokenize` token counts 均通过。该工具仍 fail-closed，未创建或修复 bank；下一步是完成外部 artifact 身份、owner、checkpoint 绑定及 4 行 detached smoke，之前不安排大规模 NPU。

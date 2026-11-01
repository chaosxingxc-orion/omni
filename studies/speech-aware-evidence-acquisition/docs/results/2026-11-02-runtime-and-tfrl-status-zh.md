# 2026-11-02 运行态与 TFRL 进展

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

已登记的 TFRL-v2 证据条件化消费 rail 仍是 CPU 设计状态。其 bank receipt、exemplar bank 和 rule bank 都是零 hash pending；rail 会在 flight start 拒绝 pending bank、估算 token 数、来源折叠或消费样本重叠。因此不能直接启动 Fold B smoke 或 1,080 条任务。

下一候选只改槽位证据条件化消费，保持 P、知识构建、奖励、模型和解码固定；TRAIN 困难集按原始真实意图—槽位关系冻结，不构造笛卡尔积，不读取 test 标签调优。必须先完成 bank 的真实 vLLM tokenize、来源隔离、4 行 smoke 和 detached checkpoint 注册，之后才可安排 NPU。

## 工程验证边界

公开仓 `main` 已包含 `31935f49`：ready queue 原子 checkpoint 写入/恢复、完整 test fail-closed gate 和 TFRL 单组件协议。当前主机 Python 环境没有 pytest，且直接导入 rails 会触发系统 Python 3.11 的既有 dataclass `mappingproxy` 默认值错误；因此本轮没有把本地解释器异常冒充为 rail 通过。正式 TFRL rail 测试必须通过固定 `saea-base` 镜像执行；镜像当前不在本地，拉取受 Docker Hub 代理认证阻塞。该阻塞只影响本地测试复核，不改变已封存的历史结果或准入结论。

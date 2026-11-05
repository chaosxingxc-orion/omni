# 两个 TP4 与 TFRL 实验执行

状态：2026-11-05 已完成两个四卡实例部署。第二实例已通过实际健康、模型身份、runtime与session注册；不做布局研究或吞吐扫参。

## 资源与任务

- 卡0–3：保留现有 TP4/8200，48K、max-num-seqs128。立即推进已授权 ACE candidate0 固定80条诊断，客户端C16；不等待另一实例迁移。候选测试实际消费来源约束，不仅凭ACE算法名声称知识构建改善。
- 卡4–7：两个TP2已合为TP4，64K、max-num-seqs128、token batch2048。`/health`返回200，模型列表报告max_model_len=65536；运行/等待请求均为0。固定原镜像与模型，旧两容器停止后保留。新profile为`qwen3-omni-explore-1xtp4-s128-b2048-ctx64k-v1`，model为`bf16-thinking-explore-tp4-ctx64k-v1`，session为`serving-explore-tp4-ctx64k-v1`。
- 两条持续研究线为知识构建与知识消费。TF-GRPO、ACE、Reflexion作为构建/经验方法对照；GEPA主要优化消费节点，方法跨线时显式声明学习对象。算法不各占一个实例，依赖任务用DAG表达。

## 接续与效率

A0已经13,078/13,078完成且正式评分封存，不再安排重跑。Reflexion候选2的现有双臂分析由CPU核验原预测、请求来源、评分与哈希，不重新生成80条。节点供给和空轨迹审计修复由其他负责人执行，主线程不重复开发。

切换前核对旧生产者终态、未知请求、真实租约及其他负责人占用，保留旧配置、历史响应和失败成本。部署与实验用既有脚本及外部配置，新服务需新profile、runtime receipt、session和队列绑定，不篡改旧身份。新服务健康不等于有实验负载，真实请求/完成轨迹才算有效产出。

每线提前准备独立后继；CPU评分、外部教师与NPU异步。教师只在本机运行，已完成ACE/Reflexion响应共享复用。监控按600秒或终态事件，报告完整样本/总量、秒/条、卡数/实例、ETA和tokens。当前无已证实的稳定TFRL收益，不能把迁移或启动写成精度提升。

## 本轮交付入口

独立执行任务为 ace-tp4-activation、tp4-lane-merge、reflexion-pair-evidence。状态以各自落盘报告和主线程验收为准，委派消息不是完成证据。完整公开基线见 A0-Full-Test-Baseline；实验结果继续原位更新Wiki。

## 已完成合并的运行证据

合并前A/B生产者分别退出0/1；240/80次生成响应与对应FIT封存和端点日志均验证，未知请求0。B旧租约在核验原PID消失、生产者身份及请求完整性后保留并清算。旧guard保留，新lane复用原A guard inode；旧lane保持退役标记，后继须使用新lane和新预检绑定。第一TP4的容器、PID及启动时间保持原值。

通用配置入口`configured_engine_merge.py`复用已有deployment配置、卡占用校验、实际进程身份和resident注册流程，单独支持多lane合并；context-only入口限制保持。固定镜像83项测试通过、3项跳过；旧全目录profile测试将partition误作profile的既有失败单独记录。部署0生成、0教师调用、0benchmark；4卡/1新实例，整体2个TP4，样本0/0、秒/条不适用、剩余0，精确启动成本未知。服务健康不证明实验满载或精度收益。

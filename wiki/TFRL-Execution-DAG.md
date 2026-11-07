# CPU与NPU异步任务图

更新：2026-11-07 07:10 UTC。研究图不是可执行launcher；当前运行证据和指标见[当前任务](TFRL-Current-Status-and-Next-Plan)。

```mermaid
flowchart LR
  A["NPU A: Decision运行"] -->|生成释放| C96["NPU A: OFF C96已绑定"]
  C96 --> PON["NPU A: P thinking ON已绑定"]
  K2["NPU B: K002运行"] -->|生成释放| K1["NPU B: K001恢复已排队"]
  K1 -->|生成释放| R["NPU B: 教师候选队列注册中"]
  Prep["CPU: 教师候选准备/注册/准入"] --> R
  Fix["CPU: 第三实例启动配置修复"] --> C["NPU C: 旧P1080待启动"]
  K2 -.轨迹.-> Score["CPU: 组内评分"]
  K1 -.轨迹.-> Score
  Score --> GRPO["本机: TF-GRPO教师"]
  GRPO --> Next["CPU: 编译下一批候选"]
  R -.结果.-> Eval["CPU: 官方评分/节点归因"]
  Eval --> GEPA["本机: GEPA节点教师"]
  GEPA --> Next
  Eff["CPU: 600秒效率窗口"] --> Config["CPU: 后继效率配置"]
```

四个教师策略分别为Reflexion Decision operation/object、Slot spoken-surface、Slot binding-rescan、ACE锁定卡片说明。共同completeON科学父；只改对应节点，不叠加。候选先在固定Fold A1080对比，冻结后Fold B1080确认；完整TEST13078不调优。

CPU准备不得占NPU资源后才开始；评分与生成异步。TF-GRPO的组内奖励是必要科学依赖，其余独立教师候选不等待它。已有固定镜像和配置脚本够用就执行，不扩展泛化加固。12卡持续有效负载是目标，当前第三实例未产出模型请求，不能宣称满载。

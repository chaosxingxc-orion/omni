# CPU与NPU异步任务图

更新：2026-11-07 09:06 UTC。图为研究/资源依赖摘要；运行由已存在的配置队列执行，实际进度见[当前任务](TFRL-Current-Status-and-Next-Plan)。

```mermaid
flowchart LR
 A["NPU A: OFF C96 842/1080"] -->|生成释放| PON["P thinking ON: binder存活"]
 PON --> HIST["P排序: binder存活"]
 K2["NPU B: K002 467/625"] -->|生成释放| K1["K001断点恢复"]
 K1 --> D["Reflexion Decision: waiting-container"]
 D --> S["Surface: waiting-source"]
 S --> R["Rescan: waiting-source"]
 R --> ACE["ACE: waiting-source"]
 FIX["CPU: 缓存已验证，安全接续"] --> C["NPU C: P任务"]
 C --> B175["B175 625: 准入完成，绑定待交付"]
 K1 -.轨迹.-> REWARD["CPU: 同策略K2奖励"]
 K2 -.轨迹.-> REWARD
 REWARD --> TF["本机远程教师: TF-GRPO独立模板"]
 TF --> BUILD["CPU: 编译新候选"]
 D -.结果.-> EVAL["CPU: 官方配对与节点归因"]
 S -.结果.-> EVAL
 EVAL --> GEPA["本机远程教师: GEPA独立模板"]
 GEPA --> BUILD
 OLD["Decision人工候选: FINALIZED"] --> PAIR["CPU: 配对分析已派发"]
```

教师四个候选各自继承同一completeON父，不能沿图累加科学改动。所有未知前驱由现有binder读取真实队列状态后绑定。175零请求原因是tiktoken词表缓存未挂载，离线get_encoding0.212秒、create_context2.086秒；正在做旧worker结算与新挂载接续，尚无业务首请求。FIT8/16/32固定镜像39项通过，真实吞吐对照待后继配置，不能宣称提速。

候选先固定Fold A1080比较，冻结后Fold B1080确认，再完整TEST13078、不调优。CPU分析异步，只有必要科学依赖阻挡对应实验。当前尚未实现12卡持续满载。

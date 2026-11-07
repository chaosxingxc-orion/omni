# CPU与NPU异步任务图

2026-11-07 20:46按最新用户顺序更新；完整定义见[逐节点阶段方案](Sequential-Node-Optimization)。当前只优化P，同阶段的候选跨实例并行，后续阶段严格依赖冻结。

```mermaid
flowchart LR
 HIST[历史P输入与轨迹] --> BASE[CPU：P独立基线和失败归因]
 BASE --> TEACH[本机远程教师：各算法P专属模板]
 TEACH --> PREP[CPU：候选审核与配置准备]
 PREP --> A[NPU实例A：P候选]
 PREP --> B[NPU实例B：P候选]
 PREP --> C[NPU实例C：P候选]
 A --> EVAL[CPU：独立P评价与回退检查]
 B --> EVAL
 C --> EVAL
 EVAL -->|下一轮| TEACH
 EVAL -->|多轮收益确认| FP[冻结P]
 FP --> S[固定P输出，只优化S]
 S --> FS[冻结P和S]
 FS --> D[固定P/S输出，只优化D]
 D --> FD[冻结P/S/D]
 FD --> SYS[handoff与系统组合优化]
```

这是目标执行图，不是已派发清单。P-only入口与历史P评价正在开发；三实例新任务只承接P阶段。现有正在运行的历史实验保留，尚未启动的旧全链后继已HOLD。CPU分析和NPU采样异步，监测600秒或终态事件。

# H1 关闭 thinking 档等待拆解

核验对象：H1 关闭 thinking，313/313 完成，8 卡、2×TP4、每实例 C96；读取已封存 debug timing，不产生模型调用。计时文件和 rows checkpoint 均通过 SHA 校验，样本边界计时覆盖 313/313。

## 观测结果

| 环节 | 次数 | 平均 | P95 | 累计（可重叠） |
| --- | ---: | ---: | ---: | ---: |
| trajectory slot wait | 313 | 119.60 s | 415.78 s | 37,433.72 s |
| native FIT capacity wait | 626 | 3.13 s | 13.39 s | 1,958.33 s |
| native FIT 执行 | 626 | 4.06 s | 14.34 s | 2,540.62 s |
| executor wait | 626 | 1.72 s | 3.46 s | 1,078.06 s |
| context build | 626 | 0.258 s | 0.801 s | 161.38 s |
| audio file read | 1,878 | 0.005 s | 0.021 s | 9.41 s |
| contact lock wait | 626 | 0.000003 s | 0.000004 s | 0.0019 s |

累计值因样本并发而重叠，不能相加作为墙钟。关闭 thinking 的成功容器墙钟为 951.75 秒；准入到首个 P 请求平均 138.68 秒。现有证据最明确地指向**轨迹槽位长期占用/等待**，而不是音频读取或 contact lock。FIT capacity wait 也存在，但量级远小于 trajectory slot wait。

## 工程解释与边界

`reproduction/ojw/runner.py` 在 `primary_slot()` 中等待 trajectory slot；`run_contact()` 随后把同步调用送入 executor。`core/model/adapter.py` 的 `request_total` 只包住模型请求，不能解释请求前的 slot 等待。`node_pipeline.py` 同时记录 prepare/model 的 queued、started、finished，但当前 H1 配置未启用 node pipeline 诊断，因此不能把等待继续拆成准备、媒体、FIT 或 prefix 锁的因果占比。

下一步应先在不改变科学配置的独立后继树启用 debug 级 `node_pipeline_timing` 和 queue bridge，记录每个样本的 prepare queued/started、model queued/started、slot id、FIT capacity 和实际请求开始；保持 normal 模式 600 秒监测。确认 slot 是否串行占住 P→S 全链路后，才考虑缩短 slot 生命周期、拆分 P/S 槽位或提高受验在途数。不能仅凭这次累计等待直接改 C96 或宣称十倍加速。

工程优先级：

1. 先完成三档 Agent 配对，避免在科学对照中途改变调度。
2. 用独立 debug 后继测量 trajectory slot 的占用区间和每实例有效在途数。
3. 若确认 slot 被完整 P→S 轨迹独占，再做小批容量后继；每次改动必须保留断点、原生 FIT 和同一 H1 名单。
4. 只有端到端墙钟和 NPU 利用率同时改善，才采纳工程改动。

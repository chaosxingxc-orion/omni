# 独立真实问句 CPU 召回：FoldA 全量

源码固定提交 `36022f619`，独立检索与评分完成；不依赖 P、native 注册或 `prepare_phase_supply`。未调用模型或 NPU。仅用于开发集检索研究，指标不是模型输出准确率，也不是泛化结论。

原 FoldA **1080/1080**，原声学 query SHA `e90fc17d4b76e3d70964bb54aa6ea26f06737bafecdf20408a05482c85ecff9a`；原 TRAIN pool 经四个保留 roster 排除后 **8554 问句、59 意图**。逐条核验 TRAIN 来源、原句、intent 和 utterance；query 身份与知识来源校验分开。无 pool 的 `cooking_query` 保持缺失。每个问句独立向量，正余弦按意图取最大值；没有合成、TEST 反馈或 gold 参与召回。

Recall@3 **957/1080 (88.611%)**，Recall@5 **994/1080 (92.037%)**，MRR **0.8325971008**，1 条 gold 不在正分结果中。按 intent 统计见 result.json。使用 Sol masked-output 全部 1080 行独立比对，前五排序和完整 gold 排名 **0 条差异**；@3 修复/退化 0/0，@5 修复/退化 0/0。参考文件 SHA `d7364eca83685621efcd253c822f82f328e7d90bb16115787c4e3f1df38cf6c4`，正式报告位于 `p-learning-overlap-audit/milestone2-full60-question/report.md`。参考比较是复现一致性，不代表两种方法的独立收益。

固定镜像 4 CPU、单线程 BLAS：来源 hash/加载/校验 **17.397 秒**，计算 **6.185 秒**，单独评分 **0.195 秒**；检索进程峰值 **1245204 KiB**，评分进程 **109748 KiB**。1080/1080，0 卡、0 实例模型调用、0 token 成本；CPU 用时为实测，金钱成本未知。8 项固定镜像测试通过。读取一次、多 cutoff 共用计算；完整缓存另可独立评分和配对，缓存及逐条参考留在执行机本地私有目录，不公开原句或轨迹。

当前实现配置化支持 max 与不同 k；kNN 留待需要时按 Sol 代码扩展。原生生产 adapter 是后续工作，不是本独立交付前置。截断缓存的 MRR 是截断 MRR，完整排名应使用 k=60 缓存。

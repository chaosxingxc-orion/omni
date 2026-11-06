# PSD construction rollout 入口：未完成

本分支增加 NodeWorkload 的显式 construction population 入口：外部配置必须提供 roster、运行视图及基线的摘要绑定，并要求 roster 为 construction/discovery、视图为 discovery；沿用原有顺序、split 摘要、音频身份、查询源摘要及其样本身份检查。原有 Fold A/B 与默认入口未改变。原生媒体字节和历史祖先曝光检查仍由原有执行门禁负责，人口审计不能代替这些检查。

该变更只是入口的一部分，尚不能用于声明完整 PSD 多轨迹采集：没有交付 K 次完整 SDK 轨迹的执行配置及来源绑定、同题同 policy 的组评分 reader、FIT/恢复总预算的组合验证，也没有完成 stage/prepare/register 的真实端到端验证。H1/H2 需要各自严格匹配的运行视图及查询向量，未在此生成或验证。现有历史诊断分组器不证明同一采集策略，不能作为 GRPO 组。采集和部署策略、temperature/seed 必须明确绑定；seed 不构成多样性证明，重复结果须保留权重。既有难例规则应单独固定引用，不能重采样评估集。

验证：指定 sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026 镜像，/opt/saea-venv/bin/python，CPU、network none、源码只读，20 项通过（实际 625 roster 及查询源篡改、原 NodeWorkload 与 Fold 测试）。测试查询向量为离线 fixture，非业务调用；未启动 NPU、外部教师或模型。人口审计的 625 与 Fold A/B 样本、话语、清单音频摘要交集为零，H1/H2 是完整划分；该审计不验证原始音频或祖先曝光。

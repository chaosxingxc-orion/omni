# GEPA Perception 节点优化

GEPA 的 Solver prompt 合同冻结 Perception，不能用于声称改善 P 候选覆盖。Perception 使用独立 schema `saea-perception-prompt-contract-v1`，只改模板首段 `perception_instruction`，冻结 Solver、知识、供给策略、解码和两节点输出接口。共享校验引擎保留旧 Solver schema 与字节身份。

## 配置与执行

1. 固定真实 TRAIN 轨迹为 source；`source.json.document.groups` 每项包含完整 P request/response、实际检索内容、下游 S request/response、开发参考和反馈。所有输入教师的 query 必须记为 learned。
2. 配置 `gepa_perception_material.py seed --repo ... --data ... --spec ... --out ...`。使用固定镜像及 `SAEA_TOKENIZER_PYTHON` 指定该镜像原生 tokenizer 解释器，不安装临时依赖。
3. 外部探索使用 `configs/learners/tfrl/gepa-perception.json`，对应独立算法/节点模板。`external_exploration.py` prepare/run/collect 保留一次请求及已知终态；明确 HTTP 拒绝后新请求另建身份，未知状态不重发。
4. 教师完成并经语义审查后，`gepa_perception_material.py pair` 从独立 pinned seed 与真实外部 collected 回放候选。禁止人工冒充教师提案。
5. `gepa_perception_workload.py stage|prepare|register` 复用共享 Flight，配置 schema 为 `saea-gepa-perception-workload-v1`，workload kind 为 `gepa-perception-template-pair`。完成固定模板、真实 FIT、登记与主线程批准后，加入既有队列。

## 比较与成本

四臂为 parent、child、AA、lock；每臂独立执行 P→S，禁止共享原 P。AA 是原生 P 模板机制要求的重复控制。每样本正常 8 contacts、fallback cap 16、FIT cap 8；80 条分别为 640/1280/640。这不是旧 Solver-only 4 contacts 方案。

S 指令/模型/解码固定，但根据各自 P 候选重新获取的 S 知识可以不同，这是 P 干预的因果后果。`gepa_perception_read.py` 调用原生 P 分支官方评分并核验各 S 消费自己的 P、P 实际新模板、相同语音/供给、固定 S 模板/解码，报告 child-parent 与 AA-lock。无完整真实候选测试时不宣称精度提升。

当前 workload 支持既有 H1 feedback/selection；H2 扩展需额外 query/utterance/audio 排除审查，不能把已曝光 feedback80 当独立验证。输入容量以供应商实际准入为准；大批轨迹超长应选择有机制依据的稳定子集并保留完整链路，不能仅增大输出预算掩盖输入上限。

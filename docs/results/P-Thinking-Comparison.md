# A 历史 P thinking ON：1080 条完整 CPU 评分与配对分析

全量 **1080/1080** 已完成离线官方评分；原生 checkpoint 为1080 keepable、0 torn。系统 Intent 84.0741% → 85.0926%，官方 SLU-F1 70.9703% → 72.3849%，Strict 587 → 583。空答和非法回答保留在完整分母中。

这是既有 P thinking ON 历史实验的最终分析，不归因于新 TFRL。当前默认仍为 **P OFF / S ON / D ON**。原始生成后的 finalization 失败记录保留；本交付是通过未修改原生门禁的独立官方评分，不将原 attempt 改写为 FINALIZED。

## 实际 CPU 故障与处理

`score-batch` 的单个 finalize job 因 `delivery config differs from recorded run` 失败，发生在官方指标计算之前。launcher 通过参数指定了 runtime overlay，并向环境传入其 SHA，却没有向评分回放传入 `SAEA_RUNTIME_CONFIG` 路径。四片段基础配置 hash 因而与记录的带 overlay 配置不一致。

隔离容器仅补入原始、只读挂载的 runtime 配置路径；重新回放得到的 hash 与原 run 完全一致。随后原生 `read_checkpoint`、完整 `verify_bundle`、未修改的 `score_bundle` 门禁及官方 scorer 均通过。无需修改源码、调整评分标准或重新生成。原始 runs、contacts、attempts、source 和 guard 全部只读；仅新的分析目录和临时目录可写。

## P 独立结果

| 指标（完整1080分母） | P OFF | 历史 P ON | 变化 |
|---|---:|---:|---:|
| Top-1 / Recall@1 | 867/1080 (80.2778%) | 912/1080 (84.4444%) | +4.1667 pp |
| Recall@2 | 951/1080 (88.0556%) | 928/1080 (85.9259%) | -2.1296 pp |
| Recall@3 | 977/1080 (90.4630%) | 932/1080 (86.2963%) | -4.1667 pp |
| MRR | 0.849691 | 0.853086 | +0.003395 |
| P 合法覆盖 | 1079/1080 | 1080/1080 | +1 |

候选顺序按原响应中首次出现的 intent 做 group_v2 稳定归并；重复候选不当作非法，不采用最终 Decision 的选择来替代 P 排名。P 非法时完整分母的排名与召回贡献为0。下游失败行的 P 仍按原始已知响应单独验证。

| P 配对指标 | 改对 | 改错 | 净变化 | McNemar精确双侧p |
|---|---:|---:|---:|---:|
| recall_at_1 | 71 | 26 | +45 | 5.45855e-06 |
| recall_at_2 | 31 | 54 | -23 | 0.0165088 |
| recall_at_3 | 25 | 70 | -45 | 4.27101e-06 |

P ON 的首个 P 响应 Recall@3 为 929/1080，部署恢复后为 932/1080；实际 P 恢复涉及 4 条。恢复响应、初始响应和费用均保留，恢复不被视为同策略独立采样。

P ON的候选数分布为1004条单候选、55条双候选、21条三候选；OFF为174/124/781条及1条非法。Top-1提高伴随候选召回下降，不能据此声称P整体更好。MRR只小幅变化。

## 系统结果与有效回答

| 指标（完整1080分母） | P OFF | 历史 P ON | 变化 |
|---|---:|---:|---:|
| Intent | 84.0741% | 85.0926% | +1.0185 pp |
| 官方SLU-F1 | 70.9703% | 72.3849% | +1.4146 pp |
| Exact span-F1 | 61.4446% | 62.6192% | +1.1746 pp |
| Strict | 54.3519% | 53.9815% | -0.3704 pp |

Strict 要求原始 scenario_action 完全正确且原始实体集合无 FP/FN。官方 SLU-F1 是完整1080记录的词/字符距离计数汇总后的 pooled micro F1，其数值分母来自实体TP/FP/FN而非固定1080；不能与 Strict 或 exact span-F1互换。未追加标签投影或 filler normalization。

| 系统配对指标 | 改对 | 改错 | 净变化 | 差值95%区间（pp） | 精确双侧p |
|---|---:|---:|---:|---:|---:|
| intent | 38 | 27 | +11 | [-0.444, +2.481] | 0.214539 |
| strict | 60 | 64 | -4 | [-2.392, +1.651] | 0.787747 |
| slot_exact | 66 | 67 | -1 | [-2.186, +2.001] | 1 |

二元指标使用配对差值的正态近似区间和精确 McNemar 检验；未进行 bootstrap 或数据重采样。多个诊断检验未校正，不将单次历史点估计解释为稳定收益。官方 SLU 差值在此仅作描述性完整分母比较。

| 回答覆盖 | P OFF | 历史 P ON |
|---|---:|---:|
| legal_entities | 716 | 723 |
| legal_empty_entities | 361 | 355 |
| blank | 3 | 2 |
| invalid | 0 | 0 |

合法覆盖按可解析输出、TRAIN 原始合法 scenario_action 组合、全局原始实体类型与非空字段核验；合法空实体输出也是有效回答，不凭空增加 intent×slot 组合约束。历史 P ON 的两条空答来自 Slot 初始输出及长度恢复后仍未完成，保留在 Intent/Strict/SLU分母及成本中；对应P响应仍独立计分。

| 原始金标分组 | 数量 | OFF Strict | ON Strict | Strict改对/改错 | OFF SLU | ON SLU |
|---|---:|---:|---:|---:|---:|---:|
| gold_has_entities | 666 | 323 | 318 | 36/41 | 74.8745% | 76.1395% |
| gold_no_entities | 414 | 264 | 265 | 24/23 | 0.0000% | 0.0000% |

金标无实体组的SLU-F1按官方计数约定为0，因为该组没有实体TP，不能将0解释为该组回答全部错误；该组应结合Strict及FP考察。按原始60个 intent 的完整聚合与P指标在 result.json中；逐行金标、配对分数、响应来源、正常恢复和原始失败原因仅保留在 private 中。

## 历史节点对照位置

既有compact分析中的原始P OFF控制也是本次同一完整1080控制：Intent84.0741%、SLU70.9703%、Strict587。历史compact P为84.5370%/71.5145%/595，compact Slot为Strict600；这些是分别登记的不同干预，不能与本次P thinking ON混称同一候选或相加。P排名分析的OFF基准867/951/977、MRR0.849691已按原样本逐行复用并核对。

## 同配置比较边界

实际原始请求：P ON temperature=0、seed=20260913、thinking预算3072、answer reserve4096、max_tokens7168；P OFF不思考、answer reserve/max_tokens4096。S/D均thinking预算3072、answer reserve1024、max_tokens4096。P ON因此同时增加了P总输出上限，是该历史注册的预算配置，不应描述成等tokens预算对照。

两臂使用同一 Fold A1080、同一 TRAIN 金标、同一官方 evaluator 和评分开关。phase material一致：True；knowledge snapshot一致：True。初始P供给顺序匹配 1080/1080，正文hash顺序匹配 1080/1080。

目标变化为P thinking OFF→ON；S/D保持各自 thinking ON 和原合同。实际P解码、原始基础配置与runtime overlay均逐项封存在private。原始供给不重新抽样，下游请求随P候选变化；因此系统结果是历史全链响应比较，不能拆成独立 Decision 因果收益。运行时期、source提交与调度差异会限制效率归因。

下游观察性转换：P ON召回gold的932条中919条最终Intent正确；P OFF对应977条中908条。两臂未召回gold时最终Intent正确均为0。该条件统计受P候选及样本难度影响，不是独立D优化收益。

系统Intent净+11条的95%区间跨0，Strict净−4条且区间跨0；当前结果不足以支持默认替换。保留现默认P OFF，并把P ON视为历史排名/候选覆盖取舍的证据，不归因于新TFRL。

## 历史成本与本轮耗时

| 历史已用成本（含正常恢复） | P OFF | 历史 P ON |
|---|---:|---:|
| 模型contacts | 3,265 | 3,269 |
| 输入tokens | 35,005,713 | 31,674,690 |
| 输出tokens | 4,973,031 | 6,613,752 |
| 累计请求秒 | 679,522.756 | 788,215.103 |
| 重复消费音频秒 | 9,327.679 | 9,328.155 |
| generation wall秒 | 11151.070 | 12835.195 |
| wall秒/条 | 10.3251 | 11.8844 |

| P ON节点（含恢复） | contacts | 输入tokens | 输出tokens | 请求累计秒 |
|---|---:|---:|---:|---:|
| solver | 1078 | 10,030,145 | 1,722,245 | 203673.120 |
| slot | 1107 | 15,876,173 | 2,698,571 | 323706.932 |
| perception | 1084 | 5,768,372 | 2,192,936 | 260835.051 |

P ON恢复子集共 31 次额外contacts、输入 338,949 / 输出 27,046 tokens；包含4次P恢复与27次Slot额外contacts，其中3次是带编号的Slot覆盖恢复访问、28次带compact repair后缀。已包含在上述节点和总费用中，不再次相加。请求累计秒因并发不能当作wall。历史A路为4卡/1实例，P ON分配卡时代理 14.2613 card-h（4×generation wall），不是实测设备活跃卡时。

本轮新增模型/teacher请求、tokens、NPU卡秒及推理API支出均为0；CPU仅1实例、2核配额。CPU分析wall 18.318秒、进程CPU 18.249秒；完整评分与容器耗时见 result.json/runtime。CPU硬件单价、历史API费率与真实device-hours未知。

本轮容器累计wall **806.104秒**（包括首次失败分析和两次独立校验）；官方评分及原始bundle验证 760.399秒，其中score_bundle 612.774秒。该累计包含容器启动开销，不等于主线程端到端时间。当前完成1080/1080、剩余0、ETA0。

相对原控制，历史P ON输入tokens -9.52%，输出tokens +32.99%，generation wall +15.10%。

## 原生资源交接与可复核交付

原生 generation-released receipt 在评分前已记录1080封存、generation_lease_released=true。原容器已终态退出；不需要再次清理锁或重置请求。A生成资源已由该原生流程释放，下一候选仍须自行取得租约并通过readiness。本轮不碰服务、共享guard或其他任务租约，也不派发下一候选。

固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`，主解释器 `/opt/saea-venv/bin/python`，`PYTHONDONTWRITEBYTECODE=1`、network none、无NPU设备挂载、cwd=/tmp。实际源码和数据挂载来自原容器inspect；原官方scorer依赖环境也按原挂载复用。

public/report.md 与 public/result.json只含脱敏聚合。private保存原故障、source pins、完整配对行、每次已知响应的request/response/wire哈希、恢复行、原生checkpoint/bundle验证、原环境与隔离评分入口；SHA256SUMS提供交付完整性。

配对分析首次执行因误用不存在的合并SLU计数字段退出；修正为现有word+char计数汇总后重新分析，原失败日志保留。官方scorer只成功执行一次。原始失败状态和早期启动失败全部保留；既有attempt仍OPENED，独立评分完成不冒充原finalization成功。未修改AGENTS或活动source，未启动子agent或额外Codex CLI。最终确认性TEST仍需原完整13078条，本任务没有读取或调优TEST。

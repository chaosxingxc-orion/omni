# PSD Slot 覆盖恢复实现与验证记录

## 目标与状态

目标是消除候选 Slot exact-set 拒绝造成的有效答案损失，最终验收目标 100% 可评分回答；预算或服务失败不是有效回答。本实现尚未满足完整交付：已知无效补问形成终态后的跨 attempt 新一轮接续尚未打通，且尚无真实 v2 轨迹业务验证，不得投入生产。

## 根因与方法

原始 batched Slot 结果少一候选时，严格 exact-set 校验拒绝整份结果。恢复配置对完整 JSON 的各行按原请求 schema、实体类型和引用分别校验，保留合法 V；仅对 M 按原候选次序逐个通过原 SDK 节点补问。补问保留原始音频、原 Slot 知识消息及独立 contact/FIT 身份；Decision 仍选择原集合。重复或越界 ID、不可解析 JSON 直接保留拒绝，未推测语义。多源 receipt 保存各 contact、有效分支及哈希，reader 使用原响应、原请求 schema、冻结知识映射重新校验和组装。没有伪造 CoreCompletion。

## 路径与配置

- `src/speech_aware_evidence_acquisition/reproduction/ojw/native_intent_slot.py`：分支调度与待恢复状态。
- `src/speech_aware_evidence_acquisition/rails/runner.py`：SDK before/after、原始响应托管、补问 handoff 和接续状态。
- `src/speech_aware_evidence_acquisition/rails/adapters/slurp_intent_slot.py`：分支校验；`rails/intent_slot_pipeline.py`、`rails/candidate_handoff.py`：多源 receipt 和 trace 重放。
- `rails/native_episode_config.py`、`rails/native_episode_receipt.py`、`core/flight/native_recovery.py`：接触上限、终态路径和 contact journal 接续；`scripts/analysis/intent_slot_workload.py`、`prepare_intent_slot_candidate.py`：独立配置和预算派生。
- 配置样例 `configs/examples/psd-slot-coverage-recovery-v1.json`；程序身份 `configs/policy/phase-native-intent-slot-batched-recovery-v1.json`。旧 v2 未配置恢复时不改。

固定最多三候选、主 P/Slot/Decision 3 次，Slot 缺失每候选最多一次，共最多 3 次额外请求：每样本 generation 上限 6；1080 样本上限 6480。raw-audio FIT 每 contact 1 个计数端点，预留 6480 FIT 端点；feature FIT 端点数按既有 `native_contact_required_cap` 计算。此上限依赖新配置显式关闭旧 answer-only fallback；不同策略不得沿用此数字。预留不是实际消耗，token、费用与墙钟尚未知。

## 验证

在 npu2 自有隔离 CPU 容器，镜像 `7e60208f`，网络关闭、源码只读、工作目录 `/tmp`，使用 `/opt/saea-venv/bin/python`：

```
python -m pytest /src/tests/unit/test_candidate_protocol_v2.py /src/tests/unit/test_intent_slot_pipeline.py /src/tests/unit/test_candidate_handoff.py /src/tests/unit/test_intent_slot_runner.py /src/tests/unit/test_intent_slot_workload.py /src/tests/unit/test_native_contact_fit.py /src/tests/unit/test_native_request_regressions.py -q --tb=short -p no:cacheprovider
```

结果 123 passed，5 warnings，28.46s。测试使用脚本化传输，不调用业务模型。实际 SDK callback、独立 FIT、runner、输出 seal/reader 覆盖单缺项成功、两个缺项、来源篡改、无效补问、预算不足；崩溃后已知补问响应复用、UNKNOWN 零重发。没有证明真实数据覆盖率或 100% 有效答案；实验 0/1080，秒/条、卡数、ETA 和实际成本均未知，测试使用 CPU、业务模型调用 0。

## 缺口、风险与复验

1. 已知无效补问形成终态后，普通 checkpoint 将弃答作为完整行保存；现有 native contact replay 固定同一请求和结果，不会分配下一轮新身份。要满足「待恢复」跨 attempt 接续，需实现来源绑定的后继恢复协议、已用预算继承和 checkpoint 行选择，并验证不重发已知请求。当前失败行保留 `slot_recovery_pending` 及原始轨迹，但不等于可自动接续。
2. 恢复仅覆盖 stop 且完整 JSON 的缺失/非法候选；坏 JSON、空 stop、length、重复冲突未自动语义重问，保留失败。旧 fallback 在恢复配置关闭；不保证 100%。
3. 需要新 v2 真实响应的多源回放、独立 release 和同名单科学复验；历史 v1 fixture 不能冒充 v2 证据。测试仅模拟响应，不能推断收益。
4. 预算上限以当前实现和 raw-audio FIT 为准；须正式注册全新实验身份、计划、FIT、配置 hash、供给和 guard 后由主线程启动。尚未发布或触碰运行树。

## 主线程预审后的补充核验

SDK 组合测试新增断言：Decision 候选为完整原集合；第二次 Slot 的 schema 不包含已有效候选；首个和补问来源的原响应分别篡改均被 `validate_receipt` 拒绝。固定镜像单测通过 1 项。新增 workload derive 测试从样例读入 pipeline，核实 80 行 generation 和 FIT 各 480，固定镜像通过 1 项；仅覆盖 derive，未实际完成外部 stage/prepare/register/launcher。

预审第一项仍未实现：`parse_config` 仍要求恢复配置关闭 answer-only fallback；原 fallback 使用同一 agent contact，且其第二次请求未经过独立 FIT。不能仅放宽配置，否则会违反新身份与共享预算要求。应继续实现 SDK 层的来源绑定新 contact、独立 FIT 和共享预算后，实测先 fallback 再补问及预算耗尽。新配置仍不可发布。

stage/register 预算元数据已改为调用 `contact_budgets()`：恢复臂以 3 次基础调用和 6 次总上限报告；旧 GEPA/Node 臂沿用原公式。固定镜像 `test_intent_slot_workload.py`：18 passed，2.91s。该变更没有实现 answer-only 共用预算，也不构成部署许可。


## SDK 共享预算阶段更新

coverage 和 `budgeted-v1` 现可共存，空 thinking 原响应及 answer-only 子 contact 保留独立请求、FIT、响应、成本与父子 trace；Slot 补问与子 contact 共用 `max_extra_contacts`。SDK receipt 保留父子两次响应并以子响应作为节点结果，reader 核对请求、响应、父子关系和解码；workload 由配置计算每行 `3 + max_extra_contacts`。固定镜像的脚本化 SDK 测试覆盖空答→answer-only→缺项补问→Decision、预算耗尽停止、来源篡改、无效答案费用和 UNKNOWN 不重发。未运行真实业务。

前文关于 fallback 必须关闭及尚未实现的段落是历史记录，不再描述当前代码。已知无效终态后的跨 attempt 接续、完整 stage/prepare/register 及真实数据科学复验仍未完成，不得投入生产。

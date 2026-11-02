# FIT 效率恢复审计（2026-11-02）

审计对象：`src/speech_aware_evidence_acquisition/core/preparation_fit.py`、`core/direct_fit_recovery.py`、`core/contact_journal.py`、`core/tfgrpo_fit.py`及相关单元测试。没有执行 SSH、Git 或源码修改。

## 当前行为与效率边界

1. Direct 全量 FIT 走 `build_direct_only_raw_fit_gate`（`preparation_fit.py:1626`）。它先渲染完整 roster 的所有 body，再按完整 wave 调用 `fit_requests`；runner 的 `whole_roster_wave_barrier` 在发送前要求整批准备完成（`rails/runner.py:7800`）。因此 FIT 是“整轮成功后才有 receipt”的原子门槛。
2. `GuardedTokenizer.request_json`（`preparation_fit.py:41`）在发 HTTP 前先增加计数并调用 `contact_sink`，当前 sink 只写 `{contact, method, route, run/attempt/session}` 到 `.contacts.jsonl`，随后才执行 `client.request_json`。日志行表示“已获准并尝试发送”，不表示服务端收到或返回。
3. `prepare_artifact` 只有 producer 正常返回才发布 `.request-fit.json`。如果进程在第 k 个 endpoint 后退出，会留下 journal 但没有 artifact；同一 FIT identity 看到 journal 或 artifact 即拒绝重试。
4. Direct recovery（`core/direct_fit_recovery.py:11-90`）只允许两种安全情况：祖先 generation 零 contact 且 FIT 三件套完全不存在时首次执行；或 `.request-fit.json + .contacts.jsonl + .custody.json` 全部存在、hash/roster/session/endpoint 序列均匹配时零 contact 复用。任何 partial/missing/changed evidence 都是 UNKNOWN，抛出错误，绝不 resend。
5. `ContactJournal`（`core/contact_journal.py:48-142`）已有逐 contact intent/response 原语：intent 无 response 明确为 `unknown`；response 必须匹配 intent hash；`NativeContactReplay.assert_all_contacts_observed`（约 324 行）对未知状态硬拒绝。该能力目前用于 native generation recovery，没有接入 FIT。
6. `tfgrpo_fit.py` 的动态文本 FIT 也只写单个 endpoint journal；存在 journal 而没有完整 artifact 时拒绝重发。因此当前设计牺牲可恢复的部分耗时，换取不重复计费/不重复请求。

## 已知/未知边界

- 已知且可复用：完整 artifact、完整 endpoint journal、custody 文件均存在，且所有输入、服务 identity、请求 roster、receipt hash 一致；复用不产生 tokenizer contact。
- 已知未开始：generation ancestry 明确为零 contact，目标 FIT identity 下没有任何 artifact/journal/custody；可开始一次全新 FIT。
- 未知：存在任何 journal 行但缺少对应响应/完整 artifact；artifact、journal、custody 任一缺失或 hash 不一致；HTTP 超时/断链发生在响应边界；服务重启或 session identity 改变则属于身份不兼容。未知结果与身份不兼容均禁止自动补发。
- 目前 Direct FIT journal 不保存 request body hash 或 response，所以无法从现有文件证明某一行已被服务端观察，也无法安全地只重做剩余 rows。Direct 活动实验的现有 FIT journal 因此只能按完整波次成功/未知处理。

## 最小可行补丁（后继执行树，不修改当前运行树）

将 FIT 从“整轮 artifact 原子提交”扩展为“每 row 独立保存请求与响应”，保持现有 guard、cap、lease 和 exposure identity：

1. 在 `GuardedTokenizer` 增加两个可选钩子：发送前写 immutable intent（request/body/session/fit identity hash），返回后写 immutable observed response（规范化响应 bytes/hash、HTTP route、latency）。发送异常只留下 intent，状态为 UNKNOWN。
2. Direct FIT 为每个 sample 使用 `ContactJournal` 目录或等价 `<contact-id>.intent/.response` 文件；所有 intent 先完成 roster/slot/audio 校验。每个 response 持久化并 fsync 后才允许生成该 row 的 fit result。
3. 新增 `resume_fit`：逐 row 校验 intent、payload hash、session identity、fit plan 和 roster。已有 observed response 只在本地重放 `measure_vllm_multimodal_request`；只有没有 intent 的 row 才能发送。存在 intent 无 response、孤立 response、hash/identity 不符时立即拒绝；任何 UNKNOWN 必须先核验终态并完成来源/成本对账，换 attempt 不解除阻塞，不退还已预留 cap。
4. 完整 wave 仍需在 generation 前汇总所有 rows，重新计算 boundary hash、每 row fit verdict、总 contacts 和 custody；发布新的完整 FIT receipt。旧 `.contacts.jsonl`/旧 schema 保持只读兼容，不能就地转换成 observed。
5. 动态 native FIT 可复用同一原语，但先只实现 Direct raw one-contact-per-row，避免同时改变 feature packet 多 endpoint 的连续序列校验。

该补丁只减少“已确认 response 的前缀”重做；对 timeout/断链留下的 UNKNOWN 不会提速，也不能把现有 partial journal 追认成成功。

## 必须补的测试

- 中断第 k 行：前 k 行各有 intent+response，k+1 只有 intent；resume 复用前缀、拒绝发送 k+1，并报告 UNKNOWN。
- 中断第 k 行且前缀 response 缺失：resume 永不调用 client，cap 不退款，错误明确为 unknown。
- response/request/session/fit-plan 任一 hash 被改写，或孤立 response 出现：拒绝恢复。
- 完整 observed prefix 且全局无 UNKNOWN，完成租约交接后只发送未有 intent 的后缀；最终 roster、boundary 和科学测量与冷跑一致，分项保留历史成本/本次新成本，总 FIT 调用数不重复累计。
- response 重放路径不产生 HTTP contact，仍通过 `measure_vllm_multimodal_request` 的 token 数、margin 和 fit verdict 校验。
- 并发恢复时同一 contact 只能有一个 intent/response；重复 producer 通过原子发布失败。
- 保留现有 Direct tests：`test_direct_native_recovery.py`、`test_direct_repeated_recovery.py` 中的 partial/missing custody 永不 recontact 断言；增加旧 schema journal 只能走 UNKNOWN 的兼容测试。
- GuardedTokenizer 计数边界：intent 计数在请求前、response 计数在返回后；请求异常计入 reserved contact，不能 refund。

## 结论

当前实现对正在运行的 Direct FIT 是可信的：已有 partial journal 不能安全恢复，只能完整成功后零 contact 复用，或在明确 zero-generation 且无 FIT evidence 时新做一次。效率补丁应放到独立后继树，先落逐 contact intent/response custody 和 UNKNOWN barrier，再做中断恢复测试；不要修改当前活动实验或把 journal 行数当成已完成 FIT。


## 补充：实现时必须收紧的恢复与并发条件

- 上文“要求新注册 attempt”不是 UNKNOWN 的解除手段。换 attempt 仍需核验原进程终态、旧租约交接、所有祖先 contact 结果和历史成本；任何未解决 UNKNOWN 必须全局阻止新 FIT endpoint，包括其他 sample 的未发送后缀。只有纯 observed 前缀且后缀无 intent 时才可续跑。对应的“前缀加后缀”测试必须不包含任何 UNKNOWN。
- 推荐的最小实现不直接把现有 generation `ContactJournal` 当作兼容接口：它的 `intent` 绑定的是 generation PreparedCoreRequest、计费预留与 native contact 身份。应复用 `atomic_file.publish_new` 的 fsync/原子发布机制，增加专门 FIT schema，固定 `fit_plan`、generation ancestry、完整 wave 输入、实际 endpoint request hash、sample ID、body hash 和完整 endpoint response。只有派生 fit row 持久化而没有响应也可做到保守恢复，但会丢失 response 已收到而 row 尚未提交的窗口。
- `GuardedTokenizer.contact_sink` 只有 `(method, path)`，目前拿不到 payload/response。增添可选钩子时保留原 callback 签名；发送前 intent 必须包括 payload，不能把旧 journal 行补写成 observed。`_LoopbackPromptAuditClient.request_json` 在 HTTP 非 200、JSON 解码失败时直接抛出，返回值只有解析后的 mapping；最小版本可保守将这些情况保留 UNKNOWN。若需要保存失败 HTTP 的已知终态，则须扩展 audit client 的 response observer，不能从异常文本推断请求未发送。
- 现有 Direct reader 对 endpoint 行做字典完全相等校验（`direct_fit_recovery.py:80-86`），并固定 `/tokenize`；不要向旧行追加字段。推荐独立版本化 sidecar 和 reader 分支，维持旧完整 receipt 的零调用复用，旧 partial 继续拒绝。当前 reader 同时要求同一 `session_id`；新设计不得默默放宽到“模型相同即可”。
- 完整 raw response durable、fit row 尚未提交的中断：恢复必须通过保存的 response 在本地重复 count/tokens 验证和 margin 计算，不能调用实际 HTTP；坏响应应标记“观察到但无效”，不能算成功，也不自动重发。逐 row 的 `fits=False` 是已知容量不足，整轮仍禁止 generation。
- 全 wave 的结果与 custody 两阶段发布之间也有崩溃窗口；新 sidecar 应支持仅从已校验的完整 observed rows 重新汇总并完成发布，不调用服务。对现存相同文件走字节一致的幂等确认，不覆盖冲突内容。

### 并发现状与最小范围

`GuardedTokenizer` 自身 `counts.total() -> guard -> counts += 1` 没有线程锁，不适合直接共享给 thread pool；GIL 不能保证整个准入序列原子。Direct 当前串行调用因此没有这一并发问题。`_LoopbackPromptAuditClient` 虽有 thread-local connection 和 request_counts 锁，也不替代全局预算锁。

已有 `build_native_contact_fit_gate`（`preparation_fit.py:1204-1222,1242-1294,1346-1352,1493-1496`）提供可参考的安全结构：raw vLLM 默认最多 8 个 FIT 并发，feature 默认 1 且明确禁止多 endpoint feature 并发；每 contact 独立 GuardedTokenizer/client/journal；共享锁原子预留完整 endpoint 配额、检查 contact ID 唯一性及 per-sample cap；失败不退款；共享全局 endpoint 序号在锁内分配。其预留在 semaphore 保护的执行区域中，确实限制活动请求数。不要把“已有 native concurrency=8”当成 Direct 也已并发，或直接把该 gate 接入 Direct 改变 whole-wave 合同。

最小补丁先做串行可恢复 FIT，并将并发单独后继实现。若随后并发：固定同一 FIT plan 的全局预算、每 contact 独立连接和证据、有限 in-flight 队列，出现未知后停止新增准入并回收已在途结果；保留所有 intent 的预留成本。故障后已在途请求不能通过取消 future 被当作没发生。完成顺序可乱序，最终输出按原 roster 顺序汇总。补充 cap 边缘竞争、重复 sample 双提交、guard 撤销、磁盘写失败、乱序响应、第一请求失败后 drain、全局 UNKNOWN 阻止其他后缀等测试。

### 调用点与本次验证范围

`fit_phase_runtime`、普通 `build_live_phase_fit_gate`、Direct wave、native per-contact 和 `tfgrpo_fit.build_tfgrpo_fit_gate` 是核心调用点；另有 `analysis/system_text_capacity.py:86`、`analysis/native_graph_capacity.py:271` 的 capacity probe。新可选 API 必须保持这些旧调用行为。`rails/tokenizer_runtime.SubprocessTokenCounter` 是独立本地 CPU 分词桥，有自身串行锁、超时及 fail-closed 逻辑，无服务端 FIT 授权，不应混用为实际 multimodal FIT。

只完成静态审计，没有运行正式测试，也没有性能测量；上述测试为建议，不能表述为通过。实际全量完成数、耗时、8 卡利用率和成本须由主线程当前真实日志给出，不能从本报告推算。审计源码 SHA256：

- preparation_fit.py: `8300bf46fe0e36664dc982570a41944955baf33f7956db4ce2cdd7265903eddd`
- direct_fit_recovery.py: `7cf4d05fa157d593dc208e732ce772c5c1873fcddec5977ca60c76c0d4b2bbe2`
- tfgrpo_fit.py: `fc64731dbd4d7f50792698916cdebb841e278e893e2d18b4a3ba505c0ad4942b`

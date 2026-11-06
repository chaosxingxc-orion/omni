# Checkpoint successor 注册链与输入闭包 v1

通用 compiler 的科学根注册与运行直接前驱承担不同职责。所有配置和源文件保持 hash pin；新输出保持 held，不能替代 terminal custody、外部 admission 或 Flight preflight。

## 注册契约

| 字段 | 含义 |
| --- | --- |
| `predecessor_registration` | 根注册的 `{relpath, sha256}`；所有后继保持相同 pin |
| `immediate_predecessor_registration` | 直接前驱 candidate 原始字节的 pin |
| `immediate_predecessor_plan` | 直接前驱 registered generation plan 的 pin；显式 immediate edge 必须提供 |
| compiler `candidate` / `plan` | 直接前驱的 candidate/plan，供 terminal custody 和新 attempt/FIT 身份派生 |

旧一跳注册没有 explicit immediate edge 时，`predecessor_registration` 同时表示直接前驱。没有通用“guard”名字规则；GEPA 业务原有 `generation-v1` 根身份约束仍保留。

根 `docs/readiness/<cell>/generation-plan-registered-v1.json` 必须原样保留。每一跳的 registration/plan、完整 resume declarations 与 bridge evidence 均必须可追溯。注册遍历迭代执行，最多 128 个祖先；拒绝循环、不同 run、根 pin 冲突。业务校验认证每个 bridge 的源配置、checkpoint pin、预算与科学 baseline，并要求整张祖先图从直接前驱可达，只有注册根为无前驱节点。新 FIT 身份不能复用任一祖先；已存在 root pin 不是豁免证据。

## 可选 catalog 与搬迁

Compiler v1 可额外提供：

```json
{
  "lineage_assets_manifest": {
    "path": "/mounted/catalog.json",
    "sha256": "<catalog bytes SHA256>"
  }
}
```

文件使用既有 `saea-experiment-assets-v1` schema，`files` 是 logical path 到绝对 `{path, sha256}` 的映射。没有该字段时既有行为保持。catalog 只补充被注册链引用的 registration/plan 和根 generation plan；与主 assets 重叠的已消费文件必须相同。缺失、篡改、路径冲突均拒绝，不从 repo 猜测或伪造 generation-v1。

`saea-terminal-package-prepare-v1` 可额外提供 `lineage_paths`，结构同 catalog 的 `files`，必须精确覆盖 catalog 全部条目；目标文件 SHA256 必须等于原 pin，不能以重新计算 pin 掩盖内容变化。旧路径可以离线；原 catalog 本身必须可读取并匹配 compiler 中的 pin。若无需搬迁，省略 `lineage_paths`。没有 compiler catalog 时提供该映射会被拒绝。

Prepare 输出：

- `lineage-source-manifest.json`：原 catalog 字节副本。
- `lineage-assets.json`：搬迁后的 catalog；compiler template 自动引用其新 pin。
- `lineage-relocation.json`：源 compiler pin、源 catalog pin及副本 pin、新 catalog pin、完整映射。

Prepare 检查所消费闭包后发布；package build 的 template 校验接受可选 catalog，并在 custody audit 前后重新检查。`complete` 委托同一 build。Compiler prepare/finalize 在实际消费时重新读取 catalog 和引用文件，不信任前一阶段已验证的结论。Compiler pin-index 记录 `source_lineage_manifest`，输出 assets 包含已消费闭包的字节和 pin。

Prepare 同时接受单桥与 multi bridge；multi `bridge_paths` 以 ancestor attempt 为键，逐个搬迁 evidence，完整集合必须匹配，不修改 checkpoint/费用/unknown 处理。

## 配置迁移

主 assets 保留当前 candidate、直接前驱 plan、活动 fragments、authority、exposure ledger、runtime/gate/preparation receipts。将缺失的根 registration、原 generation plan 及更早的 immediate registration/plan 放入可选 catalog，无需手工给 candidate 增补 root 字段。

直接前驱 plan 应以独立 logical path 输入，例如 `docs/readiness/<cell>/predecessor-<attempt>-generation-plan.json`，字节保持原样，不能占用新的 `generation-plan-successor-v1.json`。若已发布 registration 引用了历史 successor plan 的固定路径，必须保留其精确 pin；与新输出冲突时当前 compiler 会拒绝，不会改写历史引用。

不要把旧 `terminal-custody-result.json`、`external-successor-admission.json`、`successor-admission-operation.json`、`protocol-successor-v1.json`、旧 successor generation/fit plan 整包加入主 assets；它们不是新的 admission/custody 输出。放在 catalog 中的未引用文件不会导入，但显式搬迁映射仍需覆盖全部 catalog 条目。更简洁的 catalog 可以仅列需要的闭包。

## 验证边界

`tests/test_repeated_successor_lineage.py` 包含 package prepare/build → compiler execute/finalize → registered_workload.validate_workload → PairWorkload.validate → validate_checkpoint_successor 的两跳测试；业务 validator 均执行真实函数体。fixture 替换 producer `old_source`、terminal custody/inventory、gate、derive、prepared output/native custody 判定和 runtime 构建；它证明接口组合及业务验证路径，不证明真实 producer 认证或 1080 条运行成功。真实挂载输入测试 `test_successor_real_integration.py` 的 skip 只表示缺少挂载，不能作为固定镜像执行证据。

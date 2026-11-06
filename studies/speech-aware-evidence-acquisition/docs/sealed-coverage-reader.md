# 封存结果合同覆盖读取

`sealed_coverage_reader.py --config /absolute/config.json` 只读 generation manifest、其 outputs、绑定 roster 和训练库存，验证 SHA、完整1080 唯一名单、split identity 和 dataset lock；输出必须是独立新文件。模板见 `configs/examples/sealed-coverage-reader-v1.json`。数据和源码只读挂载，仅私有分析目录可写，不对 live checkpoint 使用。合同合法不代表音频支持、官方准确率或完整 bundle 验收。

`recovery_contacts` 含已从所有 role_visits/interrupted_role 去重并可直接识别的补问，记录样本、request ID 和可推断的类型；中断 Slot 补问若无来源标记可能漏记，因此是可观察下界。候选级 valid_ids 与完整阶段接受分别记录；未知中断 contact 单列，不推断其为补问。父子按 contact ID 后缀只能标识候选关系，不能替代 journal 验证。成本、最终正确性缺少可绑定证据时为 unknown，不能用全运行成本差代替。journal 文件名由原生 `ContactJournal._path` 的 sha256(request_id UTF-8) 派生；输出只是候选路径，须按 request ID 和 receipt hash 在只读 journal 定点核对，不能因此重发 UNKNOWN。`scores` 必须为 null；非 null 明确拒绝，最终正确性由既有配对分析处理；官方分数与配对比较继续用既有工具。

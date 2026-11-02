# 学习候选到完整 H2 的共享执行流程

本页适用于 ACE 与 Reflexion 的冻结候选。两个方法共用 `tfgrpo_h2_job.py`、注册工作流、配置化启动器和 `tfgrpo_h2_read.py`，不新增算法专属 launcher。没有真实候选时不启动评估，不生成虚假审查通过记录。

## 配置与前置条件

`configs/evaluation/examples/` 提供两个方法的独立任务身份模板。`candidate_release.sha256=null` 是有意不可执行的占位，必须替换为真实 release 文件哈希；不能用全零哈希冒充锁定。示例中的运行配置名称来自私有执行树，相关模型、部署和receipt文件未随公开仓发布；它不是公开仓可直接启动的配置。运行前在实际独立执行树重新核验模型、部署、service session、receipt 和输出预算。

候选须有完整 FINALIZED manifest、原生请求/响应/FIT、来源及费用。生成恢复支持不等于发布恢复支持：当前跨 attempt 候选发布和恢复代 H2 评分仍不支持，遇到这种情况先修共享适配器，不能删除恢复声明或重发已知请求。空记忆当前也不能通过消费路径。

## 复用证据导出器

候选真实FINALIZED后，运行共享CPU导出入口（也放入配置化launcher的argv）：

```bash
python scripts/analysis/learning_candidate_review.py \
  --repo "$study_root" --data "$data_root" --candidate "$candidate_path" \
  --manifest "$manifest_path" --out "$review_dir"
```

导出器复用现有seal与native来源验证，输出完整TRAIN证据、父版本、逐组更新、冲突和成本，以及`review.pending.json`、`release.pending.json`。不调用模型、不裁剪规则、不自动判定supported。release的review pin为空，审查表decision=pending，不能直接准入。真实审查通过后另存不可变review/release并填实际哈希；不覆盖pending原件。TRAIN原文材料只保存私有目录，不进入GitHub。

## 独立语义审查

审查只读取 TRAIN 证据，不能读取 H2/test 逐样本标签来修规则。逐项覆盖候选全部条目：

1. 核对引用组和轨迹，确认意图、槽位、事实与规则一致；部分奖励不代表正确。
2. 说明规则的适用条件、行为、例外；查找 TRAIN 反例，防止跨意图泛化和把样本答案当通用规则。
3. 对照全部未解决冲突。影响保留条目的冲突不能标为无影响；不能悄悄删条目后沿用原候选哈希。
4. 区分“不受证据支持”与“已被证伪”，保留拒绝理由和真实成本。拒绝候选不进入昂贵 H2。

`rails/learning_candidate_release.py` 的 `verify_review` 定义精确字段。CPU可以检查哈希、完整覆盖、来源成员资格、parent、预算和训练/评估隔离；规则语义与反例判断由独立审查者完成，结构验证不能自动产出 `supported`。逐项 `content_sha256` 使用项目规范化 `sha(content)`，不要用普通字符串文件哈希替代。只有完成审查才填写 `eligible-for-development-evaluation`；这不代表部署晋升。

release 精确绑定 strategy、candidate、source_manifest、consumer_roster、learning_dataset、review。数据文件用 data-root pin，H2 roster 和 TRAIN625 view 用 repo-root pin。实际消费再次核验，ACE/Reflexion声明与候选方法不符即拒绝。

## 固定执行入口

以下是已有命令的参数模板。将参数和挂载写入版本化 `configured_job_launch.py` JSON 的 argv；不要复制脚本。所有路径变量均须来自该实验的固定配置。

```bash
python scripts/analysis/tfgrpo_h2_job.py stage \
  --repo "$study_root" --data "$data_root" --candidate "$candidate_path" \
  --manifest "$manifest_path" --evaluation-config "$evaluation_config" --out "$stage_dir"
python scripts/analysis/tfgrpo_h2_job.py prepare-supply \
  --repo "$study_root" --data "$data_root" --stage-record "$stage_dir/complete.json" --out "$supply_dir"
python scripts/analysis/tfgrpo_h2_job.py register \
  --repo "$study_root" --data "$data_root" --stage-record "$stage_dir/complete.json" --prepared "$supply_dir/complete.json"
python scripts/analysis/tfgrpo_h2_read.py \
  --repo "$study_root" --data "$data_root" --name "$evaluation_name" --out "$analysis_dir"
```

stage生成配置后提交独立执行树，再prepare-supply/register；登记后用既有准入和运行流程，重新绑定干净commit及预检。最后一条仅在实际生成及来源满足完整评分要求后执行。原语义审查、生产来源校验、生成调度和CPU评分分别完成，不能跳过中间步骤。

## 验收和决策

完整固定312条，共享一次P、learned/lock各一次S，正常936调用，1872 fallback硬上限、936 FIT。报告两臂官方pooled SLU、Intent、修复/破坏、原始意图/槽位关系切片、配对区间，以及学习与评估全部成本。现有筛查门槛SLU至少+1pp且Intent不降；区间跨零不称稳定收益，多轮开发选择需披露。采用后必须用该版本重新采集，才构成跨轮RSI。最终完整13078条test不参与学习。

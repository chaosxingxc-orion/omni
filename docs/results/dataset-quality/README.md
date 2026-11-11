# SLURP数据质量清洗清单

当前版本：clean-v7。已确认并排除训练54条文本（240条录音）、测试27条文本（87条录音）；相同注释的合成训练数据同步排除59条文本（151条录音）。训练保留11,460条文本/50,388条录音；测试保留2,947条文本/12,991条录音。原始数据及gold不覆盖。

这是已确认首批，不是已证明全量干净。训练11,514条和测试2,974条均已程序筛查；首批子进程语义复核训练407条、测试109条，另有主线程补充审查。未复核与争议项不自动排除。原始devel本批未排除。筛查同词面类型差异、短句、空实体只是线索，不是删除规则。

[exclusions.jsonl](exclusions.jsonl)逐条公开split、slurp_id、原句、原始标注/实体、意图、关联录音名、删除原因、反证和听审状态。包含意图与句义不符、明确槽位类型错误、独立输入缺少任务信息、原句与标注跨业务错配。没有根据模型答错、低相似度或少数标签删数据；未做音频听审，不把标注问题称为音频损坏。

训练15667/15668的sentence是#NAME?，而annotation/tokens是完整客服请求。最终按字段冲突排除，不能说其完整注释不存在。审查中的中间恢复判断已被逐字段复核纠正。

## 重建与复现

准备官方SLURP原始train.jsonl、devel.jsonl、test.jsonl、train_synthetic.jsonl后运行：

```bash
python apply_release.py --source /path/to/slurp/metadata \
  --exclusions exclusions.jsonl --output /path/to/new-clean-metadata
```

脚本验证原记录哈希、句子和标注，生成新目录并输出校验哈希；不覆盖源文件。主工程已用本公开清单重建四份metadata，逐字节SHA一致。排除传播仅限同split+slurp_id及完全相同的合成注释，不按标签/相似度传播。

后续实验必须固定清洗版本和SHA。旧结果保持原分母；清洗前后比较须同时公开分母、排除数及原因。测试质量审查是人工介入的数据修订，不能隐瞒为完全未经检查的独立评测。测试原句不作为教师提示或训练示例。

## 来源与许可

原数据：SLURP，Bastianelli、Vanzo、Swietojanski、Rieser，EMNLP 2020；https://github.com/pswietojanski/slurp 。本页清洗决定是本项目的派生审查，不代表原作者裁决。文本遵循CC BY 4.0，见SLURP-LICENSE.txt；本发布不分发音频。report.json保存原始数据SHA、原始/排除/保留计数和范围限制。

clean-v6新增测试4183、7295、3204、319共4句/16录音：分别为普通time误标time_zone、clear动词误标time、音乐语境disco误标place_name、相对日期误标time。删除依据与反证见逐条清单。

clean-v7新增TRAIN2228/8595及TEST17119/16178/1832，分别涉及时区误标地点、星期误标事件、公司误标人物及专辑请求中的歌手误标歌单。完整上下文分析见[S-Slot-Context-Analysis.md](../S-Slot-Context-Analysis.md)。

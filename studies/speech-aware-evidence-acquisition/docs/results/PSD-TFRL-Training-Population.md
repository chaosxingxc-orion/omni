# PSD TFRL 既有学习池与评估集核验

## 结果与下一步

现有构建池625条可以继续作为TFRL学习素材的来源，不需要重新抽样。Fold A1080和Fold B1080保持原名单。按样本ID、话语ID与清单中的音频SHA256比较，构建池与两个Fold均无交集，两个Fold之间也无交集。H1 313条和H2 312条是构建池的完整不相交划分；每个集合内部一条样本对应一个话语。

|比较|样本ID交集|话语ID交集|音频摘要交集|
|---|---:|---:|---:|
|Fold A / Fold B|0|0|0|
|构建625 / Fold A|0|0|0|
|构建625 / Fold B|0|0|0|
|H1 / H2|0|0|0|

本次核验了既有运行视图的内容摘要、所有名单的split摘要，以及625条构建音频到原800条父媒体清单的逐项身份、时长、采样率和路径绑定。未重新读取原始音频文件，音频隔离结论基于已锁定清单；未证明整个历史学习祖先链无曝光，也未验证这些样本在当前PSD上的难度、意图/槽位覆盖质量或收益。

## 当前采集入口的实际缺口

目前NodeWorkload的population只支持完整Fold A/B；无population时仅接受历史80条。它尚不能直接用当前PSD配置在既有构建625或H1/H2上完成新的分组采集。不能为了迁就入口而另建DEV80，也不能将OFF/ON两种策略各一条轨迹当成同策略K4。

后继采集应复用既有构建名单、预载音频和节点配置，显式记录采集策略、模型、知识版本、三节点模板及解码参数。先以当前系统轨迹识别持续错误或不稳定样本，再按数据集原生意图/槽位关系检查覆盖；不以历史失败直接断言当前难度。困难选择的规则和选中ID固定成可复用引用，不能每个算法重新抽样。

TF-GRPO需要真实同题同采集策略多轨迹，重复结果保留权重；温度0重复调用不保证提供相对奖励信号，探索采样策略须与部署策略分开记录。四算法可以共享合格轨迹源，但各自教师输入模板、反馈与预算独立。稳定PSD基线尚未锁定，本报告不授权绕过既有实验顺序。

## 可回溯输入

所有路径相对 `studies/speech-aware-evidence-acquisition/`。脚本只读这些文件，不改名单、不调用模型。

|输入|样本数|话语数|文件SHA256|
|---|---:|---:|---|
|fold_a|1080|1080|4ba55688e80d8d768ec6cfc3ff4bacc9802fb34e4dab54913934015184497215|
|fold_b|1080|1080|1d30c501612686004936b5dd295885d9452175e6449b4a66187c08227a322cea|
|construction|625|625|36949a683739676616d4e4845fb302cad59721c8032ce4d65d18b35823a56cba|
|construction_roster|625|625|e8abc65ba4626cccb4aae9d2eb1bd73f077a6e9a407b3dba9c18bbe4c03eeddd|
|h1|313|313|7f540a76dec8eed29cc2295f6090c4551e2910bfd61cc0d8e98f09ba044a51c0|
|h2|312|312|87b3fa7032e4b51650bce2bb68f9f336a807a55b9ebc94f50661521711fccca7|

对应路径为：Fold A/B的 `configs/dataset/slurp-speech-native-evo-fold-{a,b}-1080.json`；构建视图 `configs/dataset/slurp-speech-native-stage3-construction-r01c.json`；构建与H1/H2名单 `docs/receipts/rosters/slurp-stage3-construction-r01c{,-h1,-h2}.json`。

可复算脚本与结构化结果位于主仓同目录的 `PSD-TFRL-Training-Population-audit.py`、`PSD-TFRL-Training-Population.json`。从研究目录运行 `python docs/results/PSD-TFRL-Training-Population-audit.py . /tmp/psd-population-audit.json` 可重算。本次新增模型调用0，不产生新的精度、吞吐或NPU利用率结论。

## 知识库来源补充核验

三库来源进一步核验显示：构建625全部已存在于锁定TRAIN示例库中，且与意图卡、槽位卡直接来源无交集；Fold A/B与三库来源的样本和话语交集均为0。库内存在同题示例不等于实际召回，后继困难选择和轨迹分析必须记录实际供给来源，不能把构建池成功率直接作为迁移收益。详见[知识关联与来源核验](PSD-Knowledge-Binding-Audit)。不重新抽样、不改测试集资格。

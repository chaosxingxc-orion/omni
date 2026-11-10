# S咖啡槽位知识卡对照结果

已完成固定20条训练机制样本的新 matched parent 配对分析；每臂保留原3个候选分支，共120个既有响应。新增模型调用0，正式CPU评分通过。

唯一语义变化是 coffee_type 格式句从 “One spoken word as uttered.” 改为 “Spoken words as uttered.”。20条按原候选集合与非保留条件选取，其中实际coffee意图gold仅4条，其他16条按各自真实gold分支报告；这不是20条coffee正例，也不是端到端意图识别结果。

|指标|新matched parent|候选|
|---|---:|---:|
|coffee gold4 strict / typed F1|50.00%|54.55%|
|coffee gold4整条全对|1/4|1/4|
|coffee_type TP / FP / FN|1 / 2 / 2|1 / 1 / 2|
|coffee gold4 official conditional micro SLU F1|71.30%|77.01%|
|全20实际gold typed F1|43.08%|45.45%|
|全20实际gold typed整条全对|3/20|3/20|
|全20实际gold strict F1 / 整条全对|40.00% / 2/20|42.42% / 2/20|
|全20实际gold official conditional micro SLU F1|64.14%|66.26%|
|全部60分支invalid|2/60|0/60|
|实际gold20分支invalid|0/20|0/20|

唯一多词coffee_type gold仍未恢复：父输出两个错误碎片，候选变为空，因此减少2个FP；另一个coffee样本输出跨度过宽，新增1个FP。coffee_type净减1FP，正确提取数和整条全对均未增加。另有非coffee gold新增一个过提取，以及一个peer gold补回漏槽。没有证据确认新增声学转写错误；专名词面波动须独立核验，不按大小写调模型。

两条协议无效均来自非gold的其他候选分支，原响应未闭合并持续输出空白，不能解释为coffee_type长度限制得到修复。全部60对中15个分支、11条样本有实体或合法性分歧；其他候选分支也有身份标识变化。两臂未随机交错，不能将所有变化归因于这句改卡。

已有推理每臂使用4卡/1实例、并发64。父315.43秒、5.26秒/分支；候选274.48秒、4.57秒/分支。合计709575 tokens、约0.6555卡时，含全部无效请求成本，费用未知。新增正式CPU评分6.46秒，NPU使用0，剩余工作量0。

结论是保留父方案，不扩展候选。下一步先核完整饮品描述与配料的标注边界、冗余中心词及业务类型的提取条件；协议尾空白单独诊断。当前只有4条coffee gold、1条多词gold，不能声称稳定收益。本次未执行复测或新推理。

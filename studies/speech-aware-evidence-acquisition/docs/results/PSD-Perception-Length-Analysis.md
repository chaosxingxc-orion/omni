# Perception长空白输出：已验证机制与候选

## 结论

两个已保存的P失败响应均在`distinction`达到schema规定的128字符并闭合引号之后，开始重复字段间空白，未输出`intent_id`等其余必需字段，最终耗尽4096 tokens。主线程独立复算确认。不是trim就能恢复的完整答案，也不能从解释文字猜标签。

|证据|样本一|样本二|
|---|---:|---:|
|已闭合distinction字符数|128|128|
|字段外尾部空白字符数|6756|14879|
|重复字符周期长度|5|11|
|输出tokens / thinking tokens|4096 / 0|4096 / 0|

两例都是选择出的失败样本，不能据此推算全部length失败的根因比例。其JSON结构不完整，原响应和费用继续保留。

## 为什么解释字段先输出

请求规范化会递归排序对象键，使schema properties顺序成为distinction、evidence_refs、intent_id、unresolved_fields。required数组顺序不强制输出顺序。当前安装的XGrammar默认按properties顺序约束生成，能够解释观测，但历史请求实际backend未保存，仍不能确证该请求走了XGrammar。

## 可配置优化及边界

安装的vLLM提供服务级`--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`，可限制JSON字段间可变空白；字符串内正常空格不因此删除。当前auto配置未启用此限制；显式指定backend会取消自动回退，需验证P/Slot/Decision全部schema，不能当纯空白开关已实测生效。当前未改服务或运行实验。

旧guided参数在该版本已移除；当前适配器也不接受相关请求参数，不能把未生效参数写入配置当作优化。增大输出上限只会容许更多空白，尚无证据支持它作为修复。

## 一个独立提示候选

已保存`configs/probes/slurp_psd_perception_compact_v1.txt`与`configs/evaluation/psd-reference-prompts-v1/compact-workload.json`。它保留reference A全部文本，追加单行紧凑JSON、distinction为1–96字符、闭合值后立即继续分隔符和必填字段三项要求。96是待验证选择，schema仍为128，可能损失描述信息，不能先称收益。

候选相对reference A仅name、run_id和P模板绑定不同，原Fold A1080顺序、KB、标签、Slot/Decision及解码不变。它是可审阅配置，尚未stage/prepare/register/admit或入队。提示、服务解码配置与有限纠错是三个独立干预，分别评价；小样本机制核验不替代全Fold覆盖与官方精度。

## 验证范围

2份已有响应的CPU分析、schema/提示/实际镜像源码只读核对、候选逐字段比较已完成；主线程核24份交付文件摘要一致。没有新模型、教师或NPU调用。最终目标仍是完整1080有效回答与历史P→Solver效果对齐，尚未达成。

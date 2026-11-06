两份候选分别仅在真实正常 Perception / Slot instruction 后追加语义指导，是系统基线提示对照，不是 TFRL。未增加 reference A/B 或 compact 正常节点前缀。

具体父本是主线程已准入的 `complete-coverage-on-workload-v1.json`，SHA256 `1c502d1d8d3db034f6dbf1bbde02630d95ae654d14a220551f08891f95c4971c`。其正常P模板为 `slurp_native_candidate_perception_off_v1.txt`（62b62d50…），正常Slot模板为 `slurp_native_candidate_slot_shared_v1.txt`（14003571…）。完整pins及父路径见 binding.json。ON指Slot thinking ON；P保持父本 thinking OFF。

- `p-coverage-on-workload.json`、`slot-coverage-on-workload.json` 是可由标准 `scripts/analysis/intent_slot_workload.py` 的 NodeWorkload 直接加载的完整派生配置。
- 每份与父本的差异严格限于目标节点 instruction_template、独立 name/run_id。没有更换共同基线提示。
- 两份 `*-instruction.patch.json` 各先测试真实正常父模板pin，再替换目标 instruction；仅模板科学差异允许变化。
- binding.json是来源元数据，不是运行参数或准入回执；其状态为 BOUND_STATIC_DERIVED_NOT_ADMITTED。不继承父本准入，不声称已测收益或100%覆盖。

全部sample_ids顺序、material/population/audio、KB/检索k、3候选、其他节点、v4g、恢复策略/共享池、纠错子请求模板paths/hash、thinking/decoding/runtime/execution均原样继承。父ON/OFF选择无需先决定赢家：对另一个已pin的完整父本，要求正常模板pin相同，再独立应用同一instruction patch并分配新身份；保留那个父本自己的thinking，不能拷贝本ON派生配置的thinking覆盖OFF。

静态检查验证真实父SHA、标准NodeWorkload构造、完整1080同名单、精确差异、恢复模板pins及ON/OFF thinking独立保留；没有重跑prepare/FIT/注册/评分。正常source/material/renderer/FIT/注册准入须由部署者另行完成，不以静态加载代替。

两项独立实验先使用既有同名单机制样例及有/无实体正确负控，再全1080配对，保持v4g和k不变；看有效回答、intent/SLU/strict、得失/终态/费用，Slot另看normalizer前后实体TP/FP/FN。不得按gold或case ID在线过滤，不同时改两个节点，不自动安排部署。

两项各自基于完整覆盖 ON 同一 workload（SHA256 `1c502d1d8d3db034f6dbf1bbde02630d95ae654d14a220551f08891f95c4971c`），只替换目标 instruction_template，name/run_id 是独立运行身份。各 patch 先检查父 pin；parent-diff 记录全部变化。两项不可合并当作单因素比较。

P 使用共同父模板原文，加上旧 compact 模板的紧凑输出段；不引入 reference-A 来源说明或先前 operation 候选。Slot 在父原文后追加紧凑输出和最小充分引用说明，不含先前 role-grounding 候选。均不改 schema、候选范围、知识、thinking、恢复和共享预算。

这些文件仅静态派生，尚未 stage、register 或 admit，也不继承父准入。正式评审完整覆盖结果后，再对每项分别走现有来源 pin、material、renderer、FIT、注册和准入流程，使用同一 FoldA1080 对比合同合法率、Intent/SLU/Strict 与原始及额外成本。

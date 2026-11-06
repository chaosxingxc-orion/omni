# PSD length recovery 与 main 集成

独立worktree的受测集成提交为 `e18569ebb2f142d0f2ea3b3d0353f356bd730f1f`；合入main `b3fa1eba2317e6b851c3d966181ece95540e44ce`（包含用户指定 `9752617b` 与 fresh custody `1c8d6c7c`）。没有改main工作树、运行release、队列或lease。最终报告独立提交在该受测源码提交之后，报告自身不改变受测代码；具体源码SHA见同目录 result.json。

合并冲突保留主线 `allow_fresh_normalized`：fresh注册资格、正常journal和FIT可用于reference normalizer和Perception length修复，包括两者同时启用；不带fresh标记的跨attempt判定对这两项均明确拒绝。主线 `gepa_pair_workload.validate` 对非checkpoint设置fresh标记；FlightRuntime对journal及FIT入口设置fresh标记，真实跨attempt入口保持默认拒绝。旧已知无效Slot后继在无新策略时保留。模板仍为 `perception_repair_template` 的confined relpath/SHA pin，默认pin `e9baa301ffa49bfc067d9571e517e7228d2ea48171a9c18448184c4f79a74507`；父length和子请求、结果、费用仍按前次实现保留。

固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`，仅ssh npu2，CPU/断网/源码RO/tmpfs。日志摘要：

```text
pytest test_intent_slot_runner.py -k 'known_invalid_repair_has_successor or perception_length_repair'
6 passed, 64 deselected, 5 warnings in 12.08s
pytest test_intent_slot_workload.py -k 'recovery_workload_derive or perception_length_recovery'
2 passed, 94 deselected, 2 warnings in 8.53s
```

使用 `/opt/saea-venv/bin/python -m pytest -q --tb=short --show-capture=no -p no:cacheprovider`，`PYTHONPATH=/src/src:/src/scripts/analysis`；两条命令在固定镜像使用只读 `/src` 和临时 `/tmp`。第一条覆盖fresh资格和默认跨attempt拒绝、旧Slot后继、P length成功/失败/UNKNOWN；第二条覆盖workload配置和注册派生边界。真实SDK、mock transport与tokenizer，非业务模型/真实FIT。远端快照与本地关键代码五项SHA核对一致。此前模板及失败链的139项结果见 length报告，本次未重复该组。

尚未做真实workload注册、独立release运行或完整Fold A评分；不能将资格单测称作真实注册通过，不能承诺100%覆盖。P schema错误、空非thinking length、Slot length、语义错误、UNKNOWN及新增策略跨attempt仍需后续处理。无教师、业务NPU或新模型调用；当前运行样本进度、卡数、ETA和成本不在本次范围，未知。

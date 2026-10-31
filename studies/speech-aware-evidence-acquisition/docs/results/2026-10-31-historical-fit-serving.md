# 历史 FIT 服务身份与报告衔接

## 已实现

换引擎后，旧 FIT 记录必须使用旧 runtime 和旧服务身份，不能借用新 attempt 的回执。原 A0 与 rec1 缺少完整 flight receipt，因此保留独立的历史快照校验路径。

`scripts/analysis/historical_fit_serving.py` 校验父 session、全部 member session/receipt、runtime 的路径、SHA256、模型、实例清单、端点与进程身份。FIT 报告现已接入该读取器：每个祖先单独绑定快照，新 attempt 仍使用实际 attachment 回执。报告逐条审计全部 FIT artifact 和 journal，零完整样本 attempt 的成本仍保留。

新增 `--historical-map` 与 `--historical-map-sha256` 参数。映射 schema 为 `saea-historical-fit-serving-map-v1`，固定 READ、generation run、新 attempt，以及所有祖先的 commit、FIT run/attempt 和 snapshot 相对路径/SHA256。映射必须完整匹配祖先清单；越界路径、缺项、额外项或错配均拒绝。哈希须来自独立审查的 READ 记录，不能在运行时从待验证文件临时计算后充当审查证据。

相同服务且不提供映射时保留原路径；变更服务却不提供映射时拒绝分析。旧、新模型 fragment、模型文件和 binary 必须相同；runtime receipt 可分别绑定。此处允许分析不同 runtime，不代表其生成行为已经证明等价。

## 验证与限制

开发机标准库测试通过：3个测试、29个场景，涵盖快照篡改、父子身份、映射范围、缺失/额外祖先、旧新 runtime 分离、模型/权重/binary 变化与缺少 FIT。命令：`PYTHONDONTWRITEBYTECODE=1 python3 tests/unit/test_historical_fit_serving.py`。

报告集成测试已补充四种组合（保留样本/成本-only祖先 × 相同/不同服务），**尚未执行**。npu2 SSH核查超时；开发机没有pytest，且项目既有模块在Python 3.11导入时触发mappingproxy dataclass错误。四个变更Python文件语法检查通过，不等同于集成测试通过。此前36项固定镜像测试只验证旧版本。

真实历史快照和独立哈希登记尚未取得；未执行真实A0全量审计。快照只能证明历史FIT服务身份，不能代替历史generation attachment证明。运行时checkpoint兼容检查未放宽，仍拒绝变更service_session。后续须完成正式镜像测试、真实证据核验和明确登记的部署兼容衔接，才可接续653条缺失样本。

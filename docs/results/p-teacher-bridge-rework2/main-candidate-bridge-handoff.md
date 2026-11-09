# Main 消费桥接交接

实际 parent patch 已通过：/home/xingchao/workspace/omni/repro/codex-subprocess/p-teacher-candidate-bridge/rework2-output-20261107T141205-3507136。三个候选子目录各包含 `configs/probes/gepa/`、`parent.baseline.json`、`baseline.json`、`patch.json`、`result.json`。复制模板目录至 runtime 配置根的同名位置，使用已应用 baseline 或在校验原父 SHA-256 后应用 patch；模板 hash 与来源在 result 单独记录。路径是 `probes/...`。原父与非 P 字节一致性、原生 parser/read 均通过。FIT 和 supply 由 runtime 处理，收据不代表运行准入。详见 rework2-report.md。

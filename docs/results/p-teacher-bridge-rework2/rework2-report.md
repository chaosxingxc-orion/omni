# P 消费桥接 rework2

三份候选已分别从原 wire 重放、编译并将 JSON Patch 实际应用于精确父 baseline；父 baseline SHA-256 `3933c579ec1610744a8202b267aae0f096df534d51284ebdb057afa274cff89a`，科学父 candidate `205ce07adaf5c2fc5fb7a1c1569a5b264859318b31199b1e31e5eade1da1156c`。父 baseline 是 p-retrieval-supply-analysis/inputs/parent.baseline.json 原 context 提取件，原生 source 提交 2586419b119d7077a405a4fb951e00ee59c231eb。实际父模板 `probes/slurp_native_candidate_perception_off_v1.txt` 文件 SHA-256 `62b62d50c9eeea020cec004d54e2ec573887fb9ef61a8a1dded37565ad99eb27`，并与科学 parent 的 executable template 字节相同。

每个输出包含原父 `parent.baseline.json`、已应用 `baseline.json`、RFC 6902 `patch.json`、模板 `configs/probes/gepa/` 和包含原请求、响应、usage、候选与独立模板 hash 的 `result.json`。patch 精确 test/replace `/phase_material/perception/instruction_template` 字符串，路径为 `probes/...`；字节替换证明其余原父字节不变。用原生 `default_phase_program`、`load_phase_material_specs`、`render_static_instruction` 读取父和每个子模板，SHA-256 匹配。结果目录 `/home/xingchao/workspace/omni/repro/codex-subprocess/p-teacher-candidate-bridge/rework2-output-20261107T141205-3507136`。每个候选目录作为配置 overlay 根，复制其 `configs/probes/gepa/` 到 runtime 的 `configs/probes/gepa/`，使用该目录 baseline 或对同一 pin 的父 baseline 应用 patch；收据单独核对 hash。

固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`、`/opt/saea-venv`、network none，实际 tokenizer；4 项集成测试通过，覆盖三候选实际 apply、原生 resolver、错误路径与旧值、父 baseline 漂移、模板 hash、非 P 字节保持。只读文件系统导致 pytest cache 提示，不影响测试。GEPA 223 tokens，Reflexion c1/c2 262/279 tokens；原请求 usage 按 request 去重，历史 GEPA 192078、Reflexion 231672、ACE 223753 tokens，本次新增 API/NPU 0。完成 3/3，NPU 0 卡，秒/条与 ETA 不适用。

该收据证明消费路径，不证明运行准入、FIT、supply 或精度；这些留 runtime。ACE 原有单轴产物与历史来源未改。保留 804f7535/86fd1af8/1dda8bb8；未调整服务、core/flight/bundle、AGENTS，未 push/merge。

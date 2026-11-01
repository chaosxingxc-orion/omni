# R01C CPU 准备登记进展

2026-11-02 在 181 的固定 `saea-base` 容器中，对 R01C 检索查询准备做了 create-only 预检和发布。执行树为 `/mnt/cephfs/xingchao/omni/work/dev/umbrella`，数据根为 `/mnt/cephfs/xingchao/omni`，使用脚本 `scripts/flights/nx/r01c_encoding_write_registration.py`。

结果：625 条完整 TRAIN R01C roster 通过身份、音频 pin、Fold A/Fold B utterance disjointness、source inventory 和协议 hash 校验；`new_contacts=0`，`registered=false`，没有访问模型、tokenizer、encoder、网络或 NPU。create-only 发布生成/确认了：

- `docs/readiness/2026-10-31-slurp-tfrl-r01c-encoding-npu-port-v1/registration-v1.json`，SHA-256 `48b210ef0998…d2d77b`；
- `configs/preparation/slurp-tfrl-r01c-speech-query-encoding-protocol-npu-port-v1.json`，SHA-256 `ff8f8f4b0dd2…a93d59`；
- `encoding-ledger-row-draft-v1.md`，SHA-256 `5e91bfd8cb01…761ef`。

181 外部根中还存在 R01C 的 625 行 speech-vector segments/content、content provenance、agreement 和 alias 文件。它们是 root-owned 受控产物，只在运行时按 receipt 使用，不复制到公开仓库，也不能仅凭文件存在就视为已完成 encoder registration。

## 当前边界

这一步只完成了 CPU 准备登记草案，未生成新的向量、未消费 NPU、未写 exposure ledger、未翻转任何 TFRL cell。TFRL-v2 五个 cell 的 `PORT_DEPLOYMENT_PROFILE`、root admission、owner countersign、cell repin/flip 和 detached 4-row smoke identity 仍未齐全；因此不能启动 smoke 或 Fold B 1,080 条实验。

下一步应在不接触模型的前提下，将已存在的 R01C receipt 与 registration 绑定进 TFRL cell 的 preparation pins，再重跑固定镜像 derive/readiness；只有所有 zero sentinel 消失、deployment profile 实测完成且 detached checkpoint 凭证成立后，才可把 4-row smoke 放入队列。

## 身份核验补充

现有 sealed receipt `SAEA-SLURP-R01C-SPEECH-QUERY-CPU-ISOLATED-V1/encode-v1` 记录的是另一条 CPU preparation run，protocol hash 为 `0e9b6341…ae0a0`；新 port registration 的 protocol hash 为 `ff8f8f4b…a93d59`，run identity 也不同。因此旧 receipt 不能直接 repin 到 `...CONSTRUCTION-R01C-SPEECH-QUERY-ENCODING-NPU-PORT-V1`，必须按新 registration 生成同身份的 preparation receipt。现有向量、agreement 和 alias 文件的存在与内容 hash 只能作为待绑定输入，不能替代该 receipt。

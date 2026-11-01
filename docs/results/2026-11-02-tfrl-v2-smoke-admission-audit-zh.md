# TFRL-v2 四行 Smoke 准入审计

当前审计结果为 **不准入**。TFRL-v2 bank 使用固定 `/mnt/cephfs/xingchao/omni` data-root、`/usr/bin/python3` verifier 检查为 `ready=true`，三份 canonical receipt hash 均通过；这不等于 smoke 已获得运行授权。

固定 `saea-base` 镜像中的 derive 预检已通过，construction H1/H2、construction smoke、Fold-B 和 Fold-B smoke 五个 cell 的 module self-check 均为 `ok`，程序分别为 `phase-native-module-v1-c7` 和 `phase-native-module-v1-c10`。

当前 smoke audit 仍拒绝，原因是：

- root admission 仍为 `DRAFT-NOT-MINTED`，没有真实 `DECIDED` receipt；
- `PORT_DEPLOYMENT_PROFILE` 仍是 pending，尚未绑定真实部署测量；
- 八个 rails 仍为 `rails_pending`，包括 bank store、rule variant、vote policy 和 stage-3 label；
- 当前 model calls 为 0，尚无 detached smoke checkpoint 或请求证据。

TP4-128 的真实 500 条结果（8 卡、2 个 TP4 实例、2.766 秒/条）只能作为工程配置候选证据，不能替代 port cell 所需的 profile binding、owner admission、checkpoint identity 和 smoke 证明。审计未启动任何 NPU 请求，也没有重复 CPU preparation。

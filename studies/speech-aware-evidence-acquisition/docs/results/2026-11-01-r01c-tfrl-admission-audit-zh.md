# R01C/TFRL 准入核验记录

## 核验口径

公开仓库只保存注册、配置和审计规则；真实 TFRL-v2 bank 位于 181 的外部数据根，不能把公开 checkout 中不存在的运行时文件当成缺失或伪造补齐。使用固定镜像执行：

```bash
python scripts/checks/tfrl_v2_bank_readiness.py \
  studies/speech-aware-evidence-acquisition/docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/bank-release-v1.json \
  --data-root /mnt/cephfs/xingchao/omni
```

2026-11-01 在 181 实际结果为 `ready=true`。H1、H2、union 的 canonical receipt hash、bank 文件 hash、schema、内容 hash 和 `vllm-tokenize` token 计量均通过；三份 receipt hash 分别为 `98959ae9…cc57`、`d6eecb62…ccef`、`c62a0b44…f9c3`。该结论只代表外部 artifact 可用，不代表整条 flight 已经准入。

## 仍需补齐的运行准入

- R01C agreement、phase-supply、vectors/segments/content/alias/provenance 的完整 pin 和可复核 receipt；
- 非 zero sentinel 的 `root-admission-v1.json` 与 owner countersign；
- `PORT_DEPLOYMENT_PROFILE` 的实测速率、资源和 provisional card-hour bound；
- detached smoke 的 run identity、checkpoint identity、队列登记和恢复凭证。

在这些凭证齐全前，不启动 Fold-B 大规模 NPU 实验。181 当前只有两个 TP4 常驻服务，不能计作 8 卡任务满载；smoke 必须进入可恢复队列并报告逐卡利用率。

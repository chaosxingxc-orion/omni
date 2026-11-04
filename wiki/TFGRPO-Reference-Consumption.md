# 多阶段 TF-GRPO reference 的开发评测入口

## 算法与来源

`external-tfgrpo-reference-v1` 对应原 summary → advantage → group_update → batch_update 多阶段流程，strategy 为 `tfgrpo-reference-multistage-external-dense-offline-v1`。连续意图/槽位奖励和离线轨迹复用是显式改编；它不属于融合式 `external-tfgrpo-v1`，也不签发 native learning seal。

## 完成条件与编译

仅使用已完成的 `prepared.json`、`collected.json`、`learner/candidate.json` 及完整 contacts/provider 原始响应。编译器逐个核对 provider wire，在临时目录从所有阶段重放候选，保留原输出，不修改源目录、不发起请求。缺失、UNKNOWN、截断响应、重算哈希后的伪造候选均拒绝。

编译配置是 `saea-tfgrpo-reference-compile-v1`，仅含 `schema`、`prepared`、`collected`；两个 pin 使用绝对 `path` 和文件 `sha256`。固定镜像内执行：

```bash
python scripts/analysis/tfgrpo_reference_release.py compile --config compile.json --out preview.json
python scripts/analysis/tfgrpo_reference_release.py prepare --config consumer.json --out params.json
```

## 语义审查与消费

编译成功只证明来源。每条原文经验仍需独立审查，拒绝案例答案泄漏、过度泛化或缺少依据的规则；不悄悄重写 teacher 输出。release/review schema 分别是 `saea-tfgrpo-reference-release-v1` 和 `saea-tfgrpo-reference-review-v1`。消费者、selection、全部学习曝光以及 parent 已学身份必须保持 query/utterance/audio 隔离，并使用固定 TRAIN625/H2 来源。

`prepare` 参数与共享 external release 一致。runtime 模型别名需要绑定冻结 actor 的等价配置。共享 Flight 与 `external_experience_workload.py` 运行同一 P、learned/lock 两个 Solver 的配对评测，`external_experience_read.py` 生成官方结果。

## 成本与科学边界

完整 `learning_usage` 和 accounting 保留，多个 release/evaluation 按 `usage_identity` 去重一次。结果中的 `shared_learning_cost` 不计入本次 evaluation `total_cost`。工程测试使用模拟 HTTP，不构成模型效果证据；稳定收益仍需真实独立开发集评测。

# 复现命令与配置

代码源提交 `36022f619`；配置为本目录 `sourcepins.json`（`query_limit` 和 `expected_queries` 均为 1080，`expected_members=8554`，`expected_intents=59`），公开配置使用 /data 和 /state 可移植挂载路径，运行前按本地布局配置；内容 SHA 不变。原实际配置及其 SHA 保存在私有运行收据中。gold pin 独立于检索配置：`/var/tmp/p-recall-depth-diagnostic-v1/labels.json` SHA256 `5bd8cd157cfaaf91805124d0a84134405028ca5adab22c6f79f5b02a2f579d9c`；只给评分进程，绝不输入检索。

实际在固定 CPU 容器，固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`，容器 `p-real-question-full-v3`，network none、4 CPU、12 GiB，源和模型数据只读挂载、结果写本机持久盘 `/var/tmp/real-question-full-v2`：

```sh
python -m pytest test_real_question_recall.py -q -p no:cacheprovider
python real_question_cpu_prepare.py sourcepins.json candidates-v3.json
python real_question_cpu_score.py candidates-v3.json gold.json score-v3.json --reference reference.json --expected 1080
python real_question_cpu_score.py candidates-v3.json gold.json paired-v3.json --baseline candidates.json --expected 1080
```

设置 `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/src`。`gold.json` 格式为 `{"path":"/var/tmp/p-recall-depth-diagnostic-v1/labels.json","sha256":"5bd8cd157cfaaf91805124d0a84134405028ca5adab22c6f79f5b02a2f579d9c"}`；`reference.json` 为 Sol masked-output 的 private-retrieval-rows.json。修改 config 的 `k`、`additional_k`、`batch_size` 可产生所需 cutoff，多个 cutoff 一次共享输入及计算；每个输出要求不存在。`--baseline` 指向另一个同样 query 身份的候选缓存，按 gold 独立计算逐样本 repair/regression。输出和 gold 均应存私有位置。

正式缓存 SHA256 `1767830927f1ddd32b450cfa00c803c0bf15cd66cc53f1543549b867869ce010`；评分文件 SHA256 `ea56f0e1c0a98ec160f921755b560f3cc506aa673d350d9d9ecba25caf710267`；配置 SHA256 `a9d7a46ca2914e8f484aad00b7e176bd56cbbac3d64696a0ca7a81654070821b`。前一 v2 首次完整运行与全部文件保留；v3 基于完成代码重跑。原历史 8 条诊断未删除。

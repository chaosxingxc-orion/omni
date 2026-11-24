连续训练候选的验收人口为每步 32 条录音、每条 4 次独立采样。三个相邻 step 必须都不超过 600 秒，计时从本批首个业务请求派发到下批首个业务请求派发，包含参数更新及服务策略切换。当前版本只验证评分调度和 CPU 对照；尚无 NPU 耗时结论。

reference_batch 接收已封闭的动作，在冻结的模型和处理器范围内按完整条件、提示与动作去重。按配置限制每个 rank 的行数和补齐后 token 数；各 rank 同步前向，尾批填充仅用于参与通信。结果回填给原请求，持久缓存命中时不再调用模型。运行 check_continuous_reference.py 需要原工程的 release（含 resident、forward 等依赖），两个 CPU rank 用同一台主机本地存储和 Gloo。真实 roster 检查与 CPU 小模型数值验证分开。

report_step_latency.py --run RUN --config TRAINING-CONTINUOUS-v1.candidate.json --after INITIAL_STEP 从原派发、响应和不可变提交记录计算墙钟，输出只表示耗时。lock_precision_baseline.py 单独封存既有数值证据的来源。接受训练版本还需真实 NPU 验证，确认服务和训练进程不重启、实际加载正确的 LoRA，且没有逐步数值审查。

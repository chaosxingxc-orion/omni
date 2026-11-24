连续训练候选的验收人口为每步 32 条录音、每条 4 次独立采样。三个相邻 step 必须都不超过 600 秒，计时从本批首个业务请求派发到下批首个业务请求派发，包含参数更新及服务策略切换。当前版本只验证评分调度和 CPU 对照；尚无 NPU 耗时结论。

reference_batch 接收已封闭的动作，在冻结的模型和处理器范围内按完整条件、提示与动作去重。按配置限制每个 rank 的行数和补齐后 token 数；各 rank 同步前向，尾批填充仅用于参与通信。结果回填给原请求，持久缓存命中时不再调用模型。运行 check_continuous_reference.py 需要原工程的 release（含 resident、forward 等依赖），两个 CPU rank 用同一台主机本地存储和 Gloo。真实 roster 检查与 CPU 小模型数值验证分开。

report_step_latency.py --run RUN --config TRAINING-CONTINUOUS-v1.candidate.json --after INITIAL_STEP 从原派发、响应和不可变提交记录计算墙钟，输出只表示耗时。lock_precision_baseline.py 单独封存既有数值证据的来源。接受训练版本还需真实 NPU 验证，确认服务和训练进程不重启、实际加载正确的 LoRA，且没有逐步数值审查。

训练常驻候选沿用原正式更新与 checkpoint：模型、Adam 和调度器跨批保留，仅在进程启动时恢复。每个采样窗口单独提交，可追溯到对应的采样版本与返回的行为概率。服务通过本机控制接口切换适配器，排空旧请求并清理缓存。精度证据单独锁版本。该目录供审阅；完整运行仍依赖原工程、配置与部署，尚未发布为达标的训练版本。

固定镜像中，check_formal_resident_windows.py 使用连续三个真实训练窗口测试 checkpoint、采样清单和 Clock 的衔接，拒绝八种错误输入组合；check_formal_resident_loop.py 用八个 CPU 进程、小模型与 AdamW 验证三次连续更新，恢复和发布用测试替身。check_resident_projection.py 验证三批输入共用只读挂载并拒绝覆盖。以上不代表整步 NPU 耗时或训推数值验收。

正式控制器的首步启动和后续命令使用同一个训练进程，命令按运行身份、序号和前一步 checkpoint 绑定；本地 checkpoint 保留在训练机，返回控制端的封闭产物含 LoRA、loss、采样身份和恢复清单。package/ 保存本次源码快照所列的完整成员，部署仍需原封闭工程依赖。旧版三个整步实测 80.6–81.4 分钟，其中 rollout 159–175 秒、带梯度前后向 233–241 秒。新版通过真实历史 checkpoint 的启动合同与控制端接收验证，尚无新 NPU 整步实测。

The sampled worker publishes the accepted consumer fields once per window. Readers verify the generation, dataset, report and tensor file. To compare with the previous implementation, run `replay_packet_container.py` against a completed training container, its sealed config and cohort, with separate local scratch. It mounts original inputs read-only, requests no accelerator and compares the complete packet on every reader.

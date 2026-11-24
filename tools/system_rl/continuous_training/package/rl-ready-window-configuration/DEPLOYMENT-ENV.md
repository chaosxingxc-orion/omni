P 节点训练、推理和控制器共同使用 `P-EXPERIMENT.template.json`。实验人员在这里调整 LoRA、microbatch、累积次数、组大小、推理并发和图宽。在站点配置或 `OMNI_DEPLOY_*` 环境变量中设置模型路径、设备和训练内存。每个实验只解析一次。

```sh
python3 paired_experiment.py --profile EXPERIMENT-DEPLOYMENT.example.json \
  --release "$OMNI_TRAIN_RELEASE" --output "$experiment_dir"
export OMNI_DEPLOY_EXPERIMENT_PACKAGE="$experiment_dir/PACKAGE.json"
```

在固定镜像中运行此入口。编译器还需要能读取当前源码树里的父策略文件及四个 LoRA 读取器。以输出的 `TRAINING-POLICY.json` 执行现有候选锁定和构建流程。`rank-profile.json` 可直接供 `expand_published_lora.py --profile` 使用。分支准备与训练打包模板都读 `OMNI_DEPLOY_EXPERIMENT_PACKAGE`。恢复旧 checkpoint 时使用其封存的配置。

把同一次生成的 `EXPERIMENT.json` 放到推理机本地，通过 `deploy_resident_graph_lanes.py "$service_root" --experiment "$local_snapshot" --prepare-only` 组包。部署时使用同一入口去掉 `--prepare-only`。推理机需要同包的 `paired_experiment.py` 和 `rank_profile.py`。训练与服务各自生成所需的运行参数。正式控制器启动时核对服务 READY 中的实验摘要。若已有初始权重，复用它绑定的原始 rank-profile 字节，准备阶段会核对其内容。变更 LoRA 或 GPU 拓扑时，已有的权重与精度回执仍需满足新实验的要求。

部署配置在启动时解析一次。主机、端口、设备列表及本地根目录可以随环境更换；算法、LoRA 参数、精度、批大小和评测周期留在实验版本里。启动参数只定位配置或输出目录，不覆盖这些实验参数。

使用已有 config_profile.py 的 compile_profile(template, profile, output)，或调用其命令行。v1 配置仍只从文件读取。v2 在本地 profile 的 environment 中声明宏及类型：path 为绝对路径，host 为主机名或 SSH 别名，port 为 1–65535 的端口，int 为整数，devices 为非空且不重复的非负整数 JSON 数组。每个宏优先读 OMNI_DEPLOY_<宏名>，否则读 values；缺失或类型错误会在派发前失败。派生目录可以在 values 中引用其他宏。此配置只放可记录的部署信息，凭据沿用服务现有的认证方式。

例如分支准备：将 OMNI_DEPLOY_PARENT_CURSOR、OMNI_DEPLOY_PARENT_CHECKPOINT、OMNI_DEPLOY_EXPERIMENT_PACKAGE、OMNI_DEPLOY_INITIAL_ADAPTER 设为本机的文件或目录，然后运行：

```sh
python3 prepare_initial_branch.py --release "$OMNI_TRAIN_RELEASE" \
  --config BRANCH-INPUT.template.json --profile BRANCH-DEPLOYMENT.example.json \
  --out "$OMNI_BRANCH_OUTPUT"
```

同一输出目录只采用第一次解析后的快照；换环境后恢复仍使用它。更换部署应使用新的输出目录。RESOLUTION.json 记录模板、原始配置、生效配置与结果的摘要；ENVIRONMENT.json 只记录声明项的来源及生效配置。以上文件留在本地热目录。

分支准备还支持 OMNI_BRANCH_CONFIG、OMNI_DEPLOYMENT_PROFILE。训练组装使用 OMNI_TRAIN_MANIFEST 和 OMNI_TRAIN_OUTPUT；如果 manifest 是模板，还要提供部署 profile 和 OMNI_TRAIN_RELEASE。显式 CLI 路径优先于对应变量。Actor 使用 OMNI_ACTOR_CONFIG 指定已封闭的训练配置；TP 部署使用 OMNI_ROLLOUT_DEPLOYMENT_ROOT 指定部署根，可用 OMNI_ROLLOUT_CONFIG 指向预解析配置，与已有部署根不一致时拒绝切换。运行时不反复读取环境。

正式训练分支可用 `configure_branch_training.py` 读取 `BRANCH-TRAINING.template.json` 和 `BRANCH-TRAINING-DEPLOYMENT.example.json`。后者声明八个可从环境传入的路径；例如 `OMNI_DEPLOY_ACTOR_SOURCE` 选择训练机上已验收的算子目录，`OMNI_DEPLOY_PROVIDERS` 选择推理实例清单。启动后根据各实例回执核对实际设备，不从实例名推断卡号。已封闭的训练构建与精度证据须匹配。

```sh
python3 configure_branch_training.py --config BRANCH-TRAINING.template.json \
  --profile BRANCH-TRAINING-DEPLOYMENT.example.json \
  --release "$OMNI_TRAIN_RELEASE" --output "$OMNI_TRAIN_OUTPUT"
```

训练部署选择权重初始化或 checkpoint 恢复。前者复用冻结基座，使用新 LoRA 和空优化器；后者恢复原优化器和 Clock。实例观察记录、训练及精度诊断分别封存。运行目录只保存本次选择和引用。

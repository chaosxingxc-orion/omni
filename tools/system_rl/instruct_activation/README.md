# Instruct TP4 activation

This is the parameterized successor to the sealed single-instance Instruct activation. It is a base checkpoint vLLM service, used to validate host compatibility and reproduce the numerical discrepancies. The RL controller's precision overlay, adapter activation and exact policy admission remain in the original RL pipeline.

1. Resolve `activation.template.json` with `prepare.py --profile PROFILE.local.json --output CODE_DIR`. Put the existing RL release's `client/source` on `PYTHONPATH` to use its `config_profile` resolver. Compile a separate profile per instance. Supply the current boot ID, immutable image, model and artifact roots, physical cards, NIC, and separate API, vLLM and HCCL port ranges. Both profiles on one host use the same local hot parent and launch guard; the hot parent and guard must exist on local persistent storage. Keep the profiles local.
2. Place each immutable code directory on its host. Run `python3 CODE_DIR/launch.py check --attempt check01` on each profile. This uses the same image and mounts without NPU devices or network.
3. Run `python3 CODE_DIR/launch.py start --attempt run01 --preflight-attempt check01` for each profile. Startup checks the receipt, ports, existing device mappings, and live device processes. Docker maps only the four physical device nodes; the CANN process sees four logical IDs, 0–3.
4. Check `launch.py ready --attempt run01`, then submit the actual audio request and collect token IDs, log-probabilities and cache metrics. For a precision verdict, compare the same token IDs at the same positions and use the frozen training reference and existing parity criterion.

## 更新态 LoRA 校准

实验约束放在 `parity-window.template.json`：`stages` 的顺序决定何时从初始 adapter 切换到后继 adapter；每项选择训练清单中的 group、replicate。精度门槛、请求超时可配置。部署 profile 采用原 `omni-local-profile-v1` 格式；提供 RELEASE、TRAIN_MANIFEST、EVIDENCE、CHECKPOINT、LANES 和 TRAINING_WORLD_SIZE。RELEASE 指 `prepare_policy.py` 输出的双版本部署包，前后权重分别由其 `checkpoint_bundle`、`successor_checkpoint_bundle` 指定。LANES 中填该包的逻辑实例名。每实例端口、卡组、容器与 guard 从包中读取。当前首次切换验收支持初始版本到一个后继版本；超过一次 FSDP 前向可容纳的请求数须分窗口。

将原 config_profile 和 rl-ready-window-configuration 放入 PYTHONPATH。执行 `parity_window.py compile --template parity-window.template.json --profile PROFILE.local.json --output WINDOW`，在本地持久盘保留模板、部署快照和解析后的配置。在已有 guard 和启动器下，将生成的 wire 与阶段 JSON 放入对应的远端运行目录，由 `parity_window.py run --window WINDOW` 发请求。部署时继续复用原 io_utils 和 measure_existing_services。若某个阶段已有请求记录，固定入口拒绝重发，先对账；不会覆盖或删除前次运行。

受控回收每个阶段的 activation-probe01（含原始响应和 DRAINED），每条 lane 的 ACTIVATION、PRECISION、CACHE-POLICY、SWITCHED 和 full-logits。控制端以 EVIDENCE/<阶段>/<lane> 定位该 lane 的源证据，以 EVIDENCE/states/<lane> 定位切换回执。`parity_window.py pack --window WINDOW` 逐请求验证原始证据并生成 PAIRED.json；将其作为原八卡评分入口的 updated_http_parity，restore_run 指定同一份 checkpoint。收回 HF 完整退出回执及各 rank 的 RESULT 至 EVIDENCE/hf，执行 `parity_window.py review --window WINDOW`。复核不占用 NPU。

可复制 `parity-window.profile.example.json` 到本地填写；只有实验设计变化时才修改 window 模板。严格四/八卡首步对照在原训练配置使用 JSON 布尔值 `"deterministic_backward": true`，仅在建立新谱系时指定。配置化窗口的 HF 评分直接取 PAIRED 中的 limits；仅复核旧无 limits 封闭包时需要提供 `parity_criterion`。部署包的 checkpoint 子目录由封闭 publication 引用定位。prepare_policy 的本地配置可用 `runtime_sources` 指向运行修复源码目录、`implementation_sources` 指向受控发布的 implementation 目录；现行验收拓扑仍为 TP4。

常驻实例使用本地 inbox 时，原 check_policy.py 通过已封闭的前后 policy 引用和各 lane 的切换回执链查找当前版本。配置沿用 inbox_switch 的 previous、successor、terminal、inbox。切换仍要求原 guard、已对账旧请求和连续 checkpoint。首次静态窗口继续通过 parity_window.py 执行。

正式部署在 `prepare_policy.py` 的本地配置显式设置 `release_kind: formal`，同一入口生成独立包，不携带逐层或全词表采集模块。可用 `formal_engine_limits` 指定 `max_num_seqs` 和 `max_num_batched_tokens`（正整数）；未配置时沿用原服务设置。更改上限需要新包，以同批真实语音及相同 adapter 核对实际吞吐和同动作概率。第一版正式包沿用了已有算术配置中的 `preserve_original`；清理不参与输出的原算子对照需独立验证。校准包继续用 `release_kind: precision`，不接受正式容量覆盖。

使用 `formal_decode_candidate` 在同一入口生成独立性能候选。可配置 `without_original_controls` 去掉原算子对照，或设置 `skip_empty_experts` 跳过四个专家均无 token 的块，并在 `empty_expert_sources` 提供已封闭的效率差量 JSON。在原 guard 下先预检并排空请求。单实例切换时在 cutover 配置中用 `retain_lanes` 引用继续运行的对照实例。候选的训练准入由更新态精度与缓存复验决定。

用同一个 `measure_existing_services.py` 运行配对压测，复用发布包中提供 `io_utils` 的 Python 路径。配置 `grouped_blocks` 为两条真实输入及各自的请求引用，`round_order` 用 `["paired"]` 或 `["paired", "paired_warm"]`，`grouped_concurrency` 指定每侧并发。两侧提交同一批请求，每个并发档有独立 cache salt。结果保存引擎 ITL 的总时长及间隔数，以及首 token、排队和每侧输出 token 吞吐。

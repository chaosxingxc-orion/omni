# PSD B / ON K2 successor 准入状态

## 结论

当前不能启动或接续 ON K2 的 K001/K002。8202 的 B TP4 服务健康，但 K2 资产仍是 CPU `stage/prepare` 完成后的 held 状态，旧 ON queue `psd-on-e8e11567-v9` 已失败，不能复用或重发。

## 已核实事实

- 181/A800I-03：固定镜像 `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`。
- B 服务：`saea-psd-aligned-b-tp4-v1`，PID `1397553`，端口 `8202`，TP4，`max_num_seqs=128`，`max_num_batched_tokens=2048`，上下文 65536，runtime receipt `62898fa4...`，session `serving-psd-aligned-b-v1`。
- 8202 的 live profile canonical SHA 为 `b4cff78e...`，其 profile 文件声明的 `model_fragment` 是 `bf16-thinking-a-tp4-ctx64k-v1`。
- ON K2 候选和 B runtime receipt 声明的 model fragment 是 `bf16-thinking-psd-b-v1`。因此候选与 live deployment 身份没有闭合。
- K001/K002 各 625 条；两个 checkpoint 均 `generation_authorized=false`、`new_model_calls=0`。K001 supply SHA `15e8b239...`，K002 supply SHA `72914a90...`。
- 8200 当前继续处理请求；本次没有停止或重启任何服务，没有发起新的模型请求。

## 解锁条件

1. 通过现有 `admit_real_engine_candidate.py` 配置入口生成新的 successor admission，使当前 B profile、runtime receipt、session、candidate、source、checkpoint 和 PID 启动身份一致。
2. 在共享 guard 下完成失败旧 queue 的终态/lease 交接；不删除锁，不盲重发未知请求。
3. 对 K001/K002 逐一完成 resident preflight、candidate/source/policy/checkpoint hash 核对。
4. 通过后按 `monitor_interval_seconds=600` 入队，从 held 资产接续，不重复 CPU stage/prepare/register。

本页是当前结果记录，不代表 K2 已准入或已产生轨迹。

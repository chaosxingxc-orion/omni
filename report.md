# Raw vLLM FIT 并发运行配置接线

## 变更

源码提交：`ca2110be6a01d84fd9de2b5c6c2f3c412e7ef42d`，分支：`feat/raw-vllm-fit-concurrency`。

本次只扩展现有 `saea-native-efficiency-v1` 外部运行配置：可选键 `native_fit.max_concurrency`。loader 先校验它必须是整数 `8`、`16` 或 `32`，overlay 再要求它只用于 raw-speech native contact FIT、vLLM 和 dispatch。`FlightRuntime` 普通路径和 `RecoverableNativeFitGate` 的惰性 factory 都通过同一个 `_build_native_contact_fit_gate` helper 将值传给既有 `build_native_contact_fit_gate`。没有改请求正文、回答预算、知识、数据、recovery/journal/lease 或计费逻辑；未做 render 复用。

未提供该键时，helper 不传 `max_concurrency`，既有 raw-vLLM factory 默认仍为 8；feature-packet 路径继续使用原来的默认/校验行为。

## 配置示例

在已有 `saea-native-efficiency-v1` 文件中加入：

```json
{
  "schema": "saea-native-efficiency-v1",
  "dispatch": {
    "schema": "saea-dispatch-slot-fill-v1",
    "role_input_offload": true,
    "ready_request": true,
    "prepare_workers": 2,
    "validation_workers": 2,
    "model_inflight": 1,
    "admitted_trajectories": 4
  },
  "native_fit": {"max_concurrency": 16},
  "audio_cache": {
    "enabled": false,
    "max_entries": 0,
    "max_bytes": 0,
    "preload_workers": 1
  }
}
```

将 `16` 替换为 `8` 或 `32` 即可进行对应实验。现有 launcher 的 `runtime_config_path` 和 SHA pin 不变。

## 测试结果

工作目录：`studies/speech-aware-evidence-acquisition`。

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:$PWD/src \
python -m pytest tests/unit/test_flight_factory.py \
  -k 'native_efficiency_runtime_config or native_fit_external_config_through_flight_factory_to_gate' -q
14 passed, 35 deselected

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:$PWD/src \
python -m pytest tests/unit/test_runtime_overlay.py \
  tests/unit/test_native_contact_fit.py tests/unit/test_native_fit_recovery.py \
  -k 'native_fit or only_raw_vllm or bounds_concurrent or recovery or custody' -q
25 passed, 46 deselected

git diff --check
python -m py_compile <factory.py> <runtime_overlay.py> <runtime.py>
passed
```

测试真实调用了 `build_flight_runtime`、`FlightRuntime._prepare` 的 composition/overlay 链路、`FlightRuntime._build_native_contact_fit_gate` 和既有 gate factory。真实 gate 并发测试分别以 8/16/32 槽位同时提交 `limit+1` 个 raw vLLM FIT contact，确认活动 tokenizer endpoint 不超过配置上限，额外 contact 等待，并核对全部 fsynced contact journal 的累计编号。

## 未测边界

本机未使用固定 saea 镜像，也未连接服务，因此本报告不宣称固定镜像验收、真实 vLLM 吞吐或 NPU 结果。未运行的完整集合中，仓库已有 3 项测试因环境缺少 `openjiuwen` 失败；另有脚本保护测试需要其规定的完整环境变量。它们不涉及本次改动的配置和 FIT factory 路径。没有 SSH、模型调用、teacher、spawn、远端实验或部署操作。

## 部署后 8 vs 16 对照方案

在同一已登记 raw-vLLM 数据、模型、请求正文、回答预算、知识、实例布局和服务镜像上分别生成只改变 `native_fit.max_concurrency` 的 8 与 16 配置，并各自使用新的 attempt、独立 runtime-config SHA 和现有 recovery/journal/lease/计费链路。先完成固定镜像的 model-free/preflight 核验，再按既有正式 launcher 脱离 SSH 运行；不复用未知请求，不重发已有 intent。

对照时记录完整样本完成数/总量、秒/条、FIT wait 与 FIT elapsed 的 p50/p95、生成请求吞吐、HTTP 尾延迟、卡数/实例、剩余时间和实际成本。两臂使用相同稳定数据和启动条件，保留 8 作为科学 baseline；只有在独立结果、请求数、响应和账本都闭合后，才解释 16 的效率差异。当前未部署、未启动该对照。

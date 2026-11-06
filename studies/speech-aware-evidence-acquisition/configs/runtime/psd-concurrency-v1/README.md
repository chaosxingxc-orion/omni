# PSD并发对照配置

状态：固定配置已备，未注册或启动，不代表效率提升。`c64.json`逐字节复用当前完整覆盖实验的外部runtime；C96/C128仅改变准入轨迹和模型在途上限。准备/校验各8线程、音频缓存1080条/4GiB/8预热线程不变。没有沿用通用example中的额外prefetch/GC开关，避免混入变量。

先选定同一科学父配置与固定Fold A1080，保持P/Slot/Decision提示、知识库、thinking、输出预算与恢复策略。用既有stage/register/admission和configured_job_launch入口，将所选配置只读挂载为`/config/runtime.json`，传`--runtime-config /config/runtime.json`，并将其SHA256写入`SAEA_RUNTIME_CONFIG_SHA256`及必要资产pin；准入和生成必须使用相同逻辑路径。运行后以实际dispatch receipt核对并发。容量profile和实际服务session必须支持所选上限。

推荐先C64→C96，若有效回答吞吐仍提高且覆盖、质量、尾延迟与KV未恶化，再到C128。保持同一引擎geometry，复用固定输入与prepared；运行中的release不热改。不能将C96配置直接覆盖C64活动attempt；恢复走既有断点兼容流程。

用已有600秒服务/硬件观测，分别报告样本完成/总量、合同合法回答、有效回答/秒、每卡Core、模型请求和token成本、排队/prefill/decode、KV/抢占、预计剩余时间。官方结果分母固定1080，误答与失败不剔除。瞬时Core高、配置加载成功或单窗口tokens/s不能单独验收加速。

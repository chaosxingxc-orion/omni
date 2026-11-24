此目录只用于独立验收。将 continuous_training/package/rl-ready-window-configuration 加入 PYTHONPATH，initial_action_review.py prepare 从自然响应封闭待复算动作。固定 HF 诊断镜像按原音频、提示、位置、动作重算；initial_action_review.py review 对完整动作逐 token 比对。训练与服务运行镜像不需要这两个探针。

在同一初始 rank-128 策略上，128 条自然响应的 1,312 个实际 token 全部与八卡 HF 相同（最大绝对 logprob 差 0）。此证据覆盖这一批和初始权重；不计为更新态验收或正式训练 step。正式推理只使用 source/audio_batch_geometry.py 中的按录音编码实现，保持语言部分批处理。

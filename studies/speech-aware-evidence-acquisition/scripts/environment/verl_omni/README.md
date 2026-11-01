# Omni verl LoRA 环境准备

这是离线准备包，**不是已经跑通的训练环境**。181 故障期间不运行 NPU、不修改 nga、不启动训练。构建和检查均独立于现有 SAEA 推理镜像。

## 固定来源

nga 镜像记录：`nga-verl-qwen35-v030:20261029`，镜像 ID `sha256:8190b3ec51929baca2388452039ccf05d681c01b7c91cc55dd1e7ac5570a76b7`。verl 上游提交 `e9405d70b77d56a23e72bb2e40a17b906856e913` **加本地补丁**；仅记录提交号不足以重建。包记录为 Torch 2.10.0+cpu、torch_npu 2.10.0.post7、Transformers 5.14.1、PEFT 0.21.2、vLLM 0.30.0、vLLM-Ascend 0.30.0rc2、NumPy 1.26.4、CANN 9.2.0。Torch 的 `+cpu` 后缀不表示不能由 torch_npu 使用 NPU。

`prepare.py` 复制 verl 源码、wheelhouse、原构建文件和证据到 omni 所属目录，逐文件记录 SHA256；复制前后检查 nga 源码未变化。不会复制权重，不会修改共享源文件。原上下文还有 CANN9.1 备用 Dockerfile，但没有同等的已核验镜像 ID；不自动切换。

## 使用

先准备（一次），后续 `verify` 复用，不反复制作：

```bash
python3 scripts/environment/verl_omni/prepare.py prepare \
  --nga-image-dir /mnt/cephfs/xingchao/docker/verl_v030_20261029/image \
  --context /mnt/cephfs/xingchao/omni/work/build/verl-omni-lora-v1/context
python3 scripts/environment/verl_omni/prepare.py verify \
  --context /mnt/cephfs/xingchao/omni/work/build/verl-omni-lora-v1/context
```

181 恢复后先核对原学习任务与资源，不自动停止它。若本机已有上述 NGA 镜像，用 `build.py --context ... --base-image sha256:... --receipt ...` 构建，仅增加探针，不更新基础依赖。若没有镜像，`--fallback` 从固定源码和 wheelhouse 在指定**本地不可变基础镜像 ID**上离线构建；输出是新镜像，不能宣称与 nga 原镜像相同。两条路径均禁用拉取及构建网络。不得为取得镜像访问或操作未授权的其他服务器。

`launch_probe.py --image sha256:... --model ... --out ...` 默认只打印命令；`--execute` 在恢复后的 npu2 上启动保留容器，写 `probe.json`、`probe.log`、`EXIT`。只挂管理设备和驱动，不挂计算卡，不做 forward/backward。探针校验导入、真实配置、Processor、meta Thinker 和精确文本注意力 PEFT 参数列表；任何检查通过均不能代表训练通过。启动结果不明时检查原容器，不自动重启。

## Omni 适配边界

- nga 的原始文本 completion 数据集不适用于原生音频，不沿用其 Qwen3.5 GDN 补丁和 LoRA 模块列表。
- 优先 Thinker（含音频塔/视觉塔、仅文本输出），冻结音频/视觉塔与 MoE 路由；保留系统求解阶段的真实音频输入。不可用转写文本代替音频来宣称原任务跑通。
- 初版 rank16、alpha32、dropout0；由精确匹配生成完整模块名单 `model.layers.<n>.self_attn.{q,k,v,o}_proj`，再显式排除 `audio_tower` / `visual`。实际meta探针证明，仅regex的初版会误注入音频塔；显式名单加排除后通过。匹配48层192个模块，预计13,369,344可训练参数；不使用仅按后缀的 q/k/v 或 all-linear，以免覆盖音频塔。
- 保留 BF16、SDPA、梯度重算；初次关闭 remove-padding、融合内核、sequence parallel（设1）和 torch.compile，待正确性与耗时实测后逐项优化。显式 `engine.model_dtype=bfloat16`，不能沿用 verl 默认 fp32。
- 必须补齐真实音频特征、音频占位 token 展开、Omni RoPE、回复 token loss mask、Thinker checkpoint 前缀映射及 FSDP 包裹。当前包只提供结构探针，未伪造可执行训练配置。

## 验证顺序

1. 镜像身份、包版本、导入与 meta/PEFT 结构。
2. 原权重加载无缺失/意外键，Thinker 参数与原基座逐项对应；单条真实 train 音频预处理和完整输入等价。
3. verl FSDP 单步前后 loss/非零有限梯度、LoRA 更新、冻结参数不变、显存/时间。
4. 连续与中断恢复对照，保存 optimizer/RNG/sampler/adapter；vLLM adapter 回灌、S-only 路由及概率一致性。
5. 固定 H1/H2 数据开展 SFT/GRPO。GRPO还需当前策略采样、奖励和版本同步；不把旧系统日志直接当在线轨迹。

首次资源安排按实测选择8卡阶段切换或4+4，不能照搬nga小模型microbatch24和吞吐。报告样本数、有效tokens、每卡吞吐、卡数、峰值显存、各阶段耗时及恢复成本。

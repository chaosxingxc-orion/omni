# 主模型 LoRA：冻结多模态塔与特征输入复用

## 结论与本次边界

按用户2026-11-01要求，LoRA只更新Thinker主模型适配器，音频塔、视觉塔及既有投影/对齐参数全部冻结；不是训练音频编码器。当前已验证的初始范围是48层文本注意力q/k/v/o共192个模块、rank16、13,369,344个可训练参数，embedding、lm_head、MoE与塔均保持基座参数。此前验证为完整结构meta检查，尚未启动真实权重训练。

**可以复用verl图像路径中的批次传递、FSDP和损失计算框架；需要增加薄的Omni音频输入适配。** 这与“把音频重新做一套训练引擎”无关。冻结塔不意味着主模型更新后必然保持原多模态能力，仍需固定音频/视觉回归集与基座输出对照。

## 已查实的代码路径

| 组件 | 可复用部分 | 必须适配部分 |
| --- | --- | --- |
| `verl/utils/model.py:extract_multi_modal_inputs` | 对多模态字典按dim0拼接，不限于pixel字段 | 先对齐不同音频的时间维和mask，不能直接cat变长特征 |
| `workers/engine/fsdp/transformer_impl.py` | 将多模态字典并入模型参数；NPU/FSDP与LoRA工程可以复用 | packed sequence、position_ids、loss_mask须与音频展开后的token一致；首次验证关闭remove-padding和序列并行 |
| `rl_dataset.py` / `multiturn_sft_dataset.py` | 消息、批次与训练接口 | 默认仅识别image/video；SFT还会删除维度不一致的多模态字段，不能直接复用成音频loader |
| NGA `raw_sft_dataset.py` | 简单response-only监督示例 | 当前为纯文本prompt+answer tokenization，未处理任何音频 |
| Omni Thinker | 原生接受`inputs_embeds`与显式Omni `position_ids`，可跳过冻结音频塔 | 音频需保留原Processor的16k采样、mel特征、feature mask、占位token扩展和RoPE；不能伪造image_grid_thw |
| Omni视觉路径 | 多模态外层框架 | 视觉有DeepStack中间层特征注入，缓存时不能只保存最后一层embedding；音频路径没有该视觉注入逻辑 |

Thinker原生音频路径是`get_audio_features → audio_tower → masked_scatter → model → lm_head`。音频塔输出维度已经对齐主模型hidden size，因此缓存冻结塔输出后，可以保留原音频token位置和Omni RoPE，将合成后的`inputs_embeds`交给同一Thinker。图像和音频可共享批次框架，但特征字段含义不同。

## 本次新增与验证

新增`audio_feature_boundary.py`：对变长音频特征补齐并保留有效mask；检查音频塔和token embedding冻结、音频塔eval模式、占位数量及response-only mask；调用原生音频编码和Omni RoPE，构造主模型输入。遇到image/video显式拒绝，防止遗漏DeepStack造成静默错误。该工具尚未注入verl actor，也未提供持久化缓存读取器。

固定独立verl镜像`sha256:3fc654a33d39341972afd4ceae482489ab26b96e73f10127f31cfe94de59c456`内，CPU随机tiny Omni进行了两次forward：原生音频输入与冻结特征边界输入的**最大logit绝对误差为0.0**。使用真实模型实现的缩小配置和随机参数，不加载生产权重，不反向、不优化、不使用NPU计算设备。这只证明该输入边界在此fixture上的等价性。

单元测试覆盖变长音频拼接、文本embedding保持、音频替换、冻结范围、eval模式、mask错配、长度错配和NaN拒绝。首次固定saea venv缺torch，收集退出2；改在上述固定verl镜像运行后10项通过。最终代码在`omni-verl-audio-tests-v3`中10项通过、5.19秒、退出0；与v2重复验证不累加计数。一般saea环境无torch时明确skip，不冒称通过。

## 最小接入顺序

1. 先走原Processor+冻结塔在线编码，复用FSDP主模型LoRA；保留原模型类与response-only损失。证明真实checkpoint加载、raw音频forward和LoRA梯度范围正确，再逐项开启packing优化。
2. 缓存原音频处理结果及冻结塔输出；缓存key包含音频内容哈希、预处理配置、塔/投影权重哈希、模型代码版本、dtype和分块配置。不同样本/音频按有序索引绑定，不能仅靠总token数。塔保持eval且在no_grad内运行，trainer调用整体train后必须重新固定塔模式。
3. 采样和训练共用音频特征缓存；每条新轨迹仍按实际prompt/response重建attention、位置和loss mask。音频特征可跨LoRA版本复用，**decoder KV缓存不可跨adapter更新复用**。
4. 真实NPU前向/反向验证后，接入verl LoRA checkpoint恢复及vLLM adapter同步，逐版本对齐rollout/train log-prob，再比较SFT与GRPO。参考/基座KL、主任务完整评测和多模态回归共同决定是否接受adapter。

当前没有新增模型精度收益、没有真实训练耗时基线、没有证明完整Ascend+Omni+verl在线RL已运行。以上实现是可复用边界验证，不把环境准备报告成训练完成。

## 来源与复查

verl来自本地NGA可运行环境源码快照，上游提交`e9405d70b77d56a23e72bb2e40a17b906856e913`及已记录本地补丁；Transformers为镜像内5.14.1。复查源码哈希：Omni modeling `a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340`；Processor `5ebde9431e03451e4d958afb1602704b1f184ebff8b6caf1d2875e3de6853172`；verl FSDP engine `41d832687cd7762bb9fb052f60092ce6d49f36a688ab77aff025d31882447b94`。原始测试日志位于npu2本地持久盘`/var/lib/omni-local-state/lora-audio-boundary-v1/`，未写入Ceph热路径。

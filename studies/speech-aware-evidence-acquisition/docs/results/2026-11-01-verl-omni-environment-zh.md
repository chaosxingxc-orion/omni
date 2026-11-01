# verl–Omni LoRA 工程环境：nga 复用与实际验证

更新：2026-11-01。**独立环境镜像已在 npu2 构建，导入与无权重 meta/PEFT 探针通过；尚未证明真实音频训练、分布式反向或在线RL可用。** 本轮没有启动参数训练，没有改变现有推理服务和科学基线。

## 已完成

| 项目 | 结果 |
| --- | --- |
| nga 来源 | `nga-verl-qwen35-v030:20261029`，记录镜像 ID `8190b3ec5192…`；nga 的保留 SFT proof 为 rc=0，但对象是 Qwen3.5 文本模型 |
| 固定源码 | verl `e9405d70b77d56a23e72bb2e40a17b906856e913` 加本地补丁；源码、wheelhouse、补丁和构建证据共1197文件、约85.77MB，逐文件SHA256锁定 |
| npu2 构建 | 本地已有 vLLM-Ascend v0.30.0rc2基础镜像；没有nga成品镜像，故用离线源码重新构建，非原镜像逐字节复现 |
| 最终镜像 | `omni-verl-lora-env:20261101-v3`，ID `sha256:3fc654a33d39341972afd4ceae482489ab26b96e73f10127f31cfe94de59c456` |
| 实际版本 | Torch2.10.0+cpu、torch_npu2.10.0.post7、Transformers5.14.1、PEFT0.21.2、verl0.8.0.dev0、vLLM0.30.0、vLLM-Ascend0.30.0rc2、NumPy1.26.4；基础栈未升级 |
| 导入 | torch_npu、verl、模型工具、SFT和PPO入口全部通过 |
| 真实配置meta检查 | Qwen3OmniMoeThinkerForConditionalGeneration、Qwen3OmniMoeProcessor可构建；保留音频塔，未读权重张量 |
| LoRA范围 | 48层、192文本注意力模块、384可训练张量、13,369,344参数；rank16/alpha32/dropout0，音频/视觉塔冻结 |
| 正式CPU测试 | 20项通过、4.34秒、退出0；覆盖上下文篡改、symlink、路径/镜像约束和无计算卡启动边界，含既有H2分析回归 |

最终镜像内探针实际退出0，7项检查通过，全部参数在meta设备；前向、反向、生成调用均0。`AutoModel`选择检查只记录默认行为，不代表默认verl能训练Omni。

## 找到并处理的问题

1. **离线构建不能直接把镜像ID当FROM名称。** 首轮BuildKit尝试远端解析后失败，没有构建成功。修复为先核验本地不可变ID，打本地专用tag，用legacy builder、`--pull=false --network=none`构建。保留首轮失败记录。
2. **LoRA误覆盖音频塔。** 初版meta探针检测到576个可训练张量，包含音频q/k/v，立即失败。显式192个完整模块路径加audio/visual排除后变为384个，且完整参数范围检查通过。未把错误配置用于真实训练；尚未将该行为归因为某个依赖的确定缺陷。
3. **默认模型入口不适配。** 原配置为`qwen3_omni_moe`，该版verl实际选择`AutoModel`；Omni外层封装与文本训练forward约定不等价。需要Thinker专用加载绑定，并核对原checkpoint的`thinker.`前缀，不能靠改architecture名称宣称解决。
4. **默认数据集没有音频路径。** nga的raw-completion和该版verl的图像/视频数据分支不能代替原生音频处理；音频展开、feature mask、Omni位置编码和回复loss mask还需适配。

nga目录与镜像未修改，没有访问其他NPU服务器。现有CANN9.1备用Dockerfile只作为证据保留，不因文件存在自动切换基础栈。Qwen3.5的GDN内核补丁与吞吐结论也不移植到Omni。

## 下一阶段

先完成原权重Thinker加载、音频数据适配和verl模型绑定，再依次验证真实train音频单步：有限非零LoRA梯度、冻结参数不变、峰值显存、分阶段耗时；随后验证checkpoint恢复与vLLM adapter回灌。首轮保留BF16、SDPA和梯度重算，关闭尚未验证的remove-padding、融合内核、sequence parallel和compile，正确性成立后按实测效率逐项优化。

训练仍保留固定H1/H2名单和完整test边界。系统轨迹只能按精确输入、token mask和策略版本转换；GRPO需新策略rollout及可信奖励。CPU准备与现有TF-GRPO实验并行，不把环境准备当作精度收益。

工具、构建/探针命令及验收顺序见[环境操作说明](../../scripts/environment/verl_omni/README.md)。本地构建证据位于`work/build/verl-omni-lora-v1`，模型权重、原始轨迹和完整环境日志不提交公开仓。

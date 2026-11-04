# 委派重启与外部探索入口交付

## 任务与实际结果

用户要求重启三个委派任务，明确交付约束；不能返回时降级为API式使用。已分别重新委派算法模板、外部入口、外部mutation适配，要求独立工作树、代码、测试、报告和commit。三者均只返回规则确认，没有创建指定工作树或实现。随后改为单次模板正文、证据展开函数、mutation选择函数请求，仍无指定产物。没有将这些确认计为完成，不再重复发送相同任务。

主线程按新授权改为直接驱动落盘和测试，实际完成四个独立算法模板/profile及`external_exploration.py`的prepare/run/collect入口。API式委派通道本轮也不可用；这些实现均由主线程完成，不归功于子智能体。

## 方法与产物

四法独立模板分别声明相对反馈、反思整理、记忆重试和提示搜索机制，均要求多个候选、根因、反例及开发/运行态信息边界。profile固定模板哈希与输出预算131072。入口配置注入固定模板与动态完整组，验证字典哈希、递归引用及canonical JSON还原，保留重复rollout权重。

prepare不调用模型，固化执行器与实际消息；run复用已有ExternalTeacher及持久化防重发；collect核对完整wire/身份/usage，拒绝不完整结束、算法错配、重复候选、未供给证据和GEPA错误父候选。结构成功仍标记语义与部署pending。

文件：`configs/prompts/learner/tfrl/`、`configs/learners/tfrl/`、`scripts/analysis/external_exploration.py`、`tests/unit/test_external_exploration.py`。使用方式见`docs/decisions/External-Exploration-Execution.md`。

## 验证与限制

固定NPU工程镜像，network none，执行`python -m pytest tests/unit/test_external_exploration.py tests/unit/test_external_teacher.py -q`。覆盖完整输入和重复次数保留、引用循环/缺键拒绝、TEST输入拒绝、模板篡改拒绝、真实执行器配合伪网络收集、wire篡改拒绝及算法/父候选/证据边界。最终21项通过，3.10秒；日志私有保存于`repro/external-entry-restart-v1/tests-final.log`。测试不调用真实API或NPU推理，不证明算法收益。

后续主线程补齐外部GEPA响应到既有配对编译器的来源分支，保留actor tokenization/FIT，不伪造本地教师身份。完整证据无损打包、保留rollout次数；庞大排除名单留本地校验，教师只接收名单承诺和允许边界。固定镜像相关35项测试通过，5.53秒。日志私有保存于 `repro/external-entry-restart-v1/gepa-packed-adapter-tests.log`。

已准备并启动一次真实GEPA候选生成：16个固定TRAIN组、64条既有轨迹，按原始scenario轮转和实际意图/槽位签名覆盖选择，输出上限131072。尚无已验收提案或新增NPU配对任务，ACE/TF-GRPO/Reflexion提案也未部署。不能称四法闭环完成或科学收益成立。

## 文件交付复核

用户指出通信不应是唯一通道。已盘点公开仓13个注册工作树、当前三个子智能体会话的实际工具调用，以及共享交付目录。发现历史工作树和报告确实存在；其中监控工作树有两处遗留修改，保留待核验，不把它算作这三个新任务产物。三个当前子智能体的记录仅有只读检查等调用，未见写入实现；新设三个任务目录只有主线程写入的TASK.md，没有receipt.json、REPORT.md或result.json。

这支持“本轮未发现可验收交付”，不支持“历史所有子智能体都没写文件”，也不足以断言失败根因是模型能力。任务消息接收/进入执行环节仍待核查。文件通道已建立并向三个原任务发出短路径通知；回执缺失时不继续发送相同长指令。私有核查证据位于 `repro/agent-mailbox/inventory.json` 和 `session-evidence.json`。

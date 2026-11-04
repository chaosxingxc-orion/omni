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

外部GEPA凭据到现有NPU配对编译器的适配尚未完成，ACE/TF-GRPO/Reflexion提案也未编译成已部署知识。无新增NPU任务，无新付费请求。该交付解决模板与调用入口，不能称四法完整实验闭环已上线。A0原运行任务未重启。

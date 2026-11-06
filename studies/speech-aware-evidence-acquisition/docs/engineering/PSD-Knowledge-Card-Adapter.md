# PSD v2 知识卡候选接入

## 已实现

`95f4b8cb`已合main：配置`slot_knowledge`将有来源记录的候选槽位知识库接入同一PSD v2 NodeWorkload。原始标签语义、候选来源及父库身份继续校验；只替换`slot-contracts`的release/snapshot，保留其他库、召回配置、节点模板、thinking和Fold名单。旧prepared供给失效后由原prepare流程重新制作，不复用错误身份。

`psd_knowledge_adapter.py`重放原编译配置、已记录教师响应和知识编辑，生成独立目录中的authored、snapshot、release、descriptor。已有目录不可覆盖。运行态加载固定版本；该适配器不调用模型。新增构建证据与消费样本按utterance检查交集，历史父库来源重叠另行报告，不冒称完全无历史曝光。

## 配置与验证

使用`configs/evaluation/psd-slot-knowledge-v1/publication.example.json`填写来源pin及新目录，再执行`python scripts/analysis/psd_knowledge_adapter.py --config CONFIG`；输出descriptor注入原Fold A/B workload的`slot_knowledge`字段。配置示例是占位模板，真实候选内容须依据原始定义和TRAIN证据审查。

开发分支固定镜像十组相关测试146项通过；主线程合并版针对知识供给、NodeWorkload、GEPA兼容等五组测试95项通过，15.88秒。验证候选快照重建、错误父库/篡改拒绝、版本不覆盖、源证据隔离和新内容进入实际候选供给与渲染。原两项fixture失败已修正，最终测试退出码0。

测试为离线fixture，未生成生产Fold prepared供给、未执行真实FIT或模型推理。知识内容变化可能改变token数量，须按候选版本做容量检查。此功能是知识构建接入，不是完整ACE Reflector→Curator学习循环，更不是已证明收益。

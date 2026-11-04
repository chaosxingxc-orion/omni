# 外部提案的研究者因素消融

`gepa_pair_material`接受与native/external互斥的`intervention`来源。文档绑定原始完整external receipt与父候选、contract、新mutation及明确的研究者review。原provider wire保持不变；新的文档SHA记录为`mutation_intervention_sha256`，不会伪装成模型响应SHA。实际actor tokenizer/FIT及模板编译不变，工作负载只允许feedback，不能直接进入selection/生产。

配置document schema为`saea-gepa-research-intervention-document-v1`，review声明`researcher-authored-factor-ablation`及`eligible-for-feedback-only`，钉住原mutation与新mutation SHA。子任务负责解释变量和证据，SHA并不证明语义合理。独立selection需要后续明确冻结与独立评估协议。

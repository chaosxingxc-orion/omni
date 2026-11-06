# 历史P→Solver与当前PSD的配置差异

## 已核实的差异

历史Fold A配置摘要已用原版本在固定镜像中复算匹配；此前不匹配来自数据根环境变量参与路径归一化。该复现不是原进程环境快照。

|项目|历史P→Solver|当前完整覆盖PSD|
|---|---|---|
|P thinking|定点请求开启，3072思考/4096总输出|关闭，4096输出|
|P预召回|k60，cosine与agreement权重各1|k8，cosine1/agreement0|
|P最终卡片上限|融合后最多8张|最多8张|
|实体抽取|Solver联合处理意图与实体|独立Slot；OFF或ON对照|
|最后决策|Solver能决定意图和实体|Decision选candidate_id，继承Slot实体|

三类知识库的release/snapshot摘要一致，但召回排序与节点分发不同，不意味着实际上下文相同。历史k60不是给模型60张卡；扩大提示卡数不能直接当作历史对齐。

## 请求证据及限制

仅定点核历史首样本两条请求：按角色与样本回执重算请求payload摘要，P实际收到8张意图边界卡，Solver收到32张（8边界、20示例、4槽位合同）。请求中P/Solver均为thinking开启、max_tokens4096，P有JSON schema，该Solver请求无response_format。

已重算响应文本及已存投影wire字节摘要；完整normalized response envelope摘要未能重建匹配，失败记录保留。这不是全1080请求审计，不据此宣称完整响应身份认证。历史serving_geometry.slots=4是服务槽位数，不能解读为卡数或实例数。

## 对下一步的影响

保持用户指定的P关闭thinking，不凭静态差异恢复旧设置。先完成现有单节点提示对照；若P候选遗漏仍是主因，再单独验证预召回与排序差异，并固定最终供给上限与其他节点。接口差异和配置差异提供研究方向，尚未证明各自导致多少精度差距。

补充核验固定镜像CPU容器exit0，主线程独核结果摘要`b6f892d91cfbefdd0c2c70c8a2076e7a1a46360325d83b1bcfafe57f2c432f11`；没有新推理或重新评分。历史成绩不改写。

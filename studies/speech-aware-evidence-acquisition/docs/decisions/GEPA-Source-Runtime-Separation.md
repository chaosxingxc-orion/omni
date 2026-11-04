# GEPA 历史来源与当前运行实例分离

## 问题与实现

配对材料的历史种子已支持 `historical_repository`，但 mutation 仍在当前运行树中重新 compose。切换推理实例时，端点和 runtime receipt 的变化导致历史 mutation 的 config hash 校验失败，阻塞新实例验证。

配对构建配置新增可选 `mutation_repository`，格式与历史种子目录一致：`{"root":"data","relpath":"work/dev/<immutable-source>/studies/speech-aware-evidence-acquisition","commit":"<40位commit>"}`。必须是数据根内的规范目录、干净且 commit 匹配；构建产物记录该配置，验证时重新解析。

三项职责分别固定：

- `historical_repository`：原始种子、冻结科学合同和模板。
- `mutation_repository`：生成 mutation 时的模型配置、注册、来源消息、候选与完整原生 bundle 校验。
- 当前 `repo` 与 `runtime_binding`：生成配对材料、绑定当前服务，保留原节点预算、科学模型和上下文约束。

未提供新字段时保留旧行为。没有删除 config hash、合同、父候选、响应或费用校验。当前 Solver 模板还必须与历史模板逐字节一致；挂载路径不能借此改变被评估的方法。

## 验证与边界

固定 saea 镜像、CPU、禁用网络：71项相关测试通过（13.79秒），覆盖来源分离、旧默认、合同/父候选/模板拒绝、历史目录篡改，以及原生 mutation/runtime 与配对科学约束。第一轮69通过2失败是新增测试缺少 fixture source_pins，修正测试后全部通过。

真实A实例配置的材料重建及80条stage验证均已完成，旧失败配置和运行树保留。该修复只解除来源耦合，不代表新实例已准入、推理已启动或候选产生稳定收益。selection80已负收益child不会仅为填满算力而重复验证；后继任务需明确科学问题与固定判据。

## 真实材料与当前实例验证结果

完整原生mutation材料重建容器退出0，合同、parent、child、compiled baseline、declaration五份文件SHA均与历史材料相同。此前发生source config hash拒绝的A实例配置，现可通过固定原生来源重建。

stage后已完成prepare和register。配置说明字段geometry_basis变更造成驻留fragment哈希不一致，恢复原登记文件后通过身份检查；随后组合历史B receipt只读视图与权威A实例/coordinator挂载，保持历史凭据字节不变。

最新准入已越过来源和服务身份校验，但在repin因注册包仍绑定旧配置哈希退出1。准入凭据已先写入，保留原件，不重复mint；该后继没有新模型调用。公共准入入口新增发送前配置/协议哈希核验，避免同类错误产生半完成凭据，固定镜像4项测试通过（1.30秒）。这不表示其余准入过程已具有事务性。

零模型调用核验确认编译parent seed与lock Solver模板字节相同，SHA256均为 `18bff5c557cac7ac14be6b08aa0a1a6ad54c1b9820199c3b6749b2271d756fbd`。selection80 parent相对lock的+2.4572pp不能解释成parent学习收益；同模板重复推理和执行轨迹差异仍需归因。child相对parent为负1.5301pp，不晋升、不为填卡重复。

优先调度已具备有效候选的新系统探索；当前80条A后继保留为工程验证，A0与驻留服务不重启。

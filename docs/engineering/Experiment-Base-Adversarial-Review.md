# 工程基座多轮对抗审查

审查基线为 `886bcf16c`。主线程进行三轮反例驱动检查，两路独立 Codex CLI 分别复核执行状态与部署。所有发现按原基线复现，修复后统一回归；未连接计算主机、未发模型请求、未改活动实验。

## 第一轮：执行结果是否可信

| 缺口 / Why | 复现 / What | 修复 / How |
| --- | --- | --- |
| EXIT 未绑定当前派发 | 放入其他 dispatch 的成功回执，后继可放行 | 比较 DISPATCH、START、EXIT 与配置身份；冲突 UNKNOWN |
| status 不核冻结配置 | 改模型字段后仍返回旧实验状态 | 用原 CONFIG-LOCK 约束查询身份 |
| 排队期间源码可漂移 | 前驱运行时改 pinned 脚本，后继执行新内容 | worker 执行业务前复核冻结输入 |
| 输出多生产者 | 两任务声明相同/嵌套输出路径 | 配置校验拒绝重叠产物 |
| 控制器换代忽略旧 worker | 旧 CPU worker 活着，新控制器超过并发上限 | 可观测 RUNNING 一并计入配额 |

## 第二轮：冻结和失败隔离能否被绕过

| 缺口 / Why | 复现 / What | 修复 / How |
| --- | --- | --- |
| PREPARE 可与原锁定摘要分叉 | 改输入并更新 PREPARE，原验收仍接受 | PREPARE pin 集合及 manifest 必须匹配 CONFIG-LOCK |
| 产物只检查存在 | 输出 symlink 指向外部旧文件被判成功 | 完成时再次核解析后的路径 |
| 单任务输出冲突杀整个 controller | 第一任务有旧输出，无关任务不执行 | 冲突在 worker 返回，按 DAG 隔离失败 |
| 上下层配置只有各自 hash | 外层 image/network 与下层冲突仍 prepare 成功 | 显式 pinned JSON 字段绑定；模板提供映射 |

## 第三轮：文档模板和部署行为是否一致

| 缺口 / Why | 复现 / What | 修复 / How |
| --- | --- | --- |
| CPU preparation 套用 generation admission | 3 个模板填值后仍被 loader 拒绝 | 有限离线 prepare 角色，与消费准入分离；模板进入回归 |
| 同名容器替代原实例 | 保存原 CID 后换同标签容器，status 接受 | 对比原 launch-result CID 及 hostname |
| 输入 pin 未覆盖实际投影 | 子 mount 覆盖父 mount 的 pinned JSON | 校验最深挂载映射后的实际文件摘要 |
| 动态库覆盖绕开源码清单 | 配置接受未 pin 的 LD_LIBRARY_PATH | 禁止任务覆盖，保留固定镜像继承路径 |
| 宿主共享缓存未传到容器 | argv 中仍为 /tmp tmpfs，无缓存/log策略 | 模板显式共享 tmp/cache/HOME 与文件日志；旧配置不自动变更 |

## 验收与保留边界

修复后 **109 项测试通过**。其中包含真实 CLI 子进程、排队中改源码、回执错配、失败隔离和旧 worker 配额反例；容器侧使用配置解析、真实文件投影及 Docker 接口模拟。独立审查另外验证原提交构建上下文可被 release verifier 接受，不能据此声称 Docker build 完成。

- 同一实验保持单 controller；CPU/NPU 并行使用 `run --phase all`。多个 phase controller 并行不受支持，不能通过删 RUN.lock 获得并发。
- 通用 argv 无法自动证明模型身份。未声明 bindings 时 model/image 字段仍只是元数据；生产模板必须绑定真实 workload 字段，并保留原生服务身份核验。
- worker 验证到执行间不是对抗性安全沙箱；生产源码和输入仍需不可变只读 release。大型资产缓存仍依赖明确 immutable 发布契约。
- 共享存储跨主机锁、实际容器用户权限、固定镜像内 Ascend 导入、NPU 接续尚未验收。旧恢复链及公开/权威源码分叉仍按迁移计划处理。
- 新 launcher 和回执协议应随新 release 使用；不能对活动任务原位替换源码。严格身份不接受缺少当前字段的旧 EXIT，应保留旧观察路径而非补造历史。

因此结论是：本轮找出了上一轮 91 项测试未覆盖的真实遗漏并完成本地修复，尚不能宣称所有部署风险已消除。后续以实际部署验收收口，不继续无边界增加框架层。

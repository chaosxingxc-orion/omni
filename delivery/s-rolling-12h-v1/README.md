# 四路异步执行源码快照

此目录及相邻 delivery 目录保存当前调度、CPU准备、结果收集和教师反馈源码，用于代码审阅。公开版仅把部署用户目录换为通用路径；运行源码与公开源码SHA见PUBLICATION-SOURCE-MANIFEST.json。运行中的原始路径未改变。

本快照不是完整可独立运行的镜像包。部署还依赖既有固定镜像、挂载路径、私有数据manifest、模型及前置delivery模块，均不得由公开快照自动下载或伪造。凭据从环境提供，私有轨迹和数据包不提交。

- scheduler.py：终态触发接续，提前stage，独立CPU评分/归档；600秒常规观测。
- prepare_plan.py --plan：从配置准备CPU任务，已发布任务复用，不重新生成输入或重发模型请求。
- group_collector.py：固定八seed按显式attempt映射收集，校验seed/config/pins及完整终态；失败历史保留。
- public_status.py：只读本地回执，按字段白名单生成公开汇总，不发起NPU轮询。

当前固定窗口以WINDOW.json为准；待执行容量与AI Core快照都不是连续12小时已完成的证明。

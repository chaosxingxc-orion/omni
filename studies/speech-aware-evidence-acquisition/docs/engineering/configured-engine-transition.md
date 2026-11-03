# 单实例引擎上下文切换

`configured_engine_transition.py plan|run|recover-stop --config FILE` 仅改变同一lane的原生context。模型权重、解码、TP、并发、端口、CPU、设备、镜像digest和其他server flags保持一致；目标不能超过已pin模型config的default-RoPE原生max_position_embeddings。新实例和新session必须使用独立名称。

配置严格声明 `source_settlement` pin、旧精确container ID、旧/新profile与model pins、native model config pin、`logical_model_fragment`、新session、逻辑data root、NIC、cards/port scope、独立本地state_dir，以及就绪探测timeout/interval。逻辑model名保留历史source身份；不能因deployment profile改名伪改历史数据。

运行前必须有同源sealed generation release或已完成terminal settlement，全部native contacts已观察，generation.guard空闲，flight.lock不存在，旧engine队列/连接为零。旧inspect/logs/session原样持久保存，随后只stop旧精确ID，不rm。新启动使用core.deployment生成的argv，保留intent/result与容器绑定。

半切换时 `locks/engine-transition.pending.json` 阻止统一GenerationLease再获取租约；新后继必须运行包含此检查的代码。异常后保留pending。`run`可依据已退出旧容器、已存在精确新容器和已发布阶段证据续接，不自动重复不确定操作。若stop intent已有而旧容器仍活跃，默认安全hold；操作方显式调用`recover-stop`，重新核验来源、live identity、guard及quiescence后允许一次可追踪的stop恢复，仍不循环盲重试。

新服务通过health/model就绪后，复制同镜像/权重的immutable receipt字段，重新验证runtime文件、权重hash、实际新PID/argv/设备/命名空间，再发布独立runtime receipt，注册新session。真实receipt随后导入新执行树并冻结；后继必须重新prepare/环境preflight，旧ready不迁移。旧session由stop证据封存，不删除历史。

就绪短轮询由固定脚本执行，有明确timeout，不消耗语言模型监测tokens；常态实验观察仍为终态事件或600秒。测试使用Docker double验证生命周期、不确定启动与中断恢复，并单独验证原始ContactJournal和真实GenerationLease；固定镜像测试不表示真实容量启动已完成。

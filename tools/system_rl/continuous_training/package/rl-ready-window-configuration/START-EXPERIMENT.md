在已安装站点启动正式 P 训练时，复制 P-RUN.example.json 为自己的实验文件。修改 run_id、baseline 和 experiment 内的参数。每个新 run 使用独立 Clock、优化器、目录和数据偏移；run_id 不能复用。当前站点已经安装；默认读取 ~/.config/omni-system-rl/SITE.local.json，也可用 OMNI_EXPERIMENT_SITE 选择其它已安装站点。站点登记的是已验收的初始 adapter、数据集、实际双机部署和精度证据。

    python3 -B start_experiment.py experiment.json --plan
    python3 -B start_experiment.py experiment.json

第一行是可选的离线预览。使用 --rehearse 可在独立目录用已封闭的同配置服务回执实际构建和编译，不启动第二个作业。第二行从同一个快照生成推理、训练和控制器配置，在原 guard 下部署两个 TP4、读取实际回执、构建训练包。正式控制器先预编译并部署 actor 静态输入，然后绑定在线服务并开始首批 rollout。提交后控制器脱离终端工作。训练进度见 run/CURSOR.json，首批轨迹、失败提交和恢复回执都保留在原目录。SUBMITTED 只证明派发；三步验收要求同一 run 的三个全局 loss、对应 checkpoint 和仍在运行的控制器。

本入口支持已登记的精度和权重配置。并发、传输、容量检查等参数从 experiment 读取。切换 rank、训练 batch 形状、图或设备拓扑，选择具有相应验收证据的 baseline；缺失时在占卡前报错。站点文件及服务登记表不进入公共仓。服务交接发生错误时检查原目录的 INTENT/READY 和实际进程；相同实验的未知派发不会重试。启动阶段耗时和异常写入 startup-events；正式控制器日志在 submit.log。

站点可在已登记 baseline 的 dispatch_inputs 下指定 static、lengths 两个不可变文件引用和 threads。两份索引属于指定数据集、预处理及逐样本召回版本：变更后重新生成。训练控制器从同一配置派发，每步仍用新策略生成真实动作。首次读取新 LoRA 时运行原校验；同一不可变文件可复用结果，完整恢复仍逐分片验证。上线中的实验由 handoff_dispatch_transition.py 在上一批已排空且训练命令已签收的边界接回原控制器；只有首次上线需要切换。生产耗时按 observe_transition_acceptance.py 的 checkpoint 返回至下批首派计量，同时报告首派至末回及相邻首派间隔。

生产者宏批的单位是录音。在已登记的 dispatch_inputs 下可以设置 producer 为 {"recordings_per_macro":128,"training_input_layout":"group_refs_v1"}。静态名单按数据集顺序准备，尾批保留；真实动作按每步策略新采。组文件保留原始动作与不变输入的内容引用，正式发布器和恢复读取器均须在同版本的 actor 构建内。宏批大小独立于训练端每卡 microbatch、累积次数及 rollout 并发。

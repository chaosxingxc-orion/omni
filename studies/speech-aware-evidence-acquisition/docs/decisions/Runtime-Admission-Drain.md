# 配置化停止准入与安全排空

## 配置和操作

新任务复用固定镜像与launcher，只挂载一个本地持久控制目录。将`configs/runtime/admission-control.example.json`复制为该目录的`config.json`，替换run_id/attempt_id，并使用容器内绝对路径。三个文件必须位于同一目录，不能为符号链接。

launcher的environment设置`SAEA_ADMISSION_CONTROL_CONFIG=/control/config.json`，将独占控制目录读写挂载为`/control`。默认不设置此变量，原执行不变。该配置是调度控制，不进入模型输入或科学配置。

在相同路径视图、相同固定镜像执行：

```bash
python scripts/environment/request_admission_stop.py \
  --config /control/config.json --request-id stop-for-engineering-fix
```

请求通过fsync和原子发布写入；相同请求可重复提交，冲突请求拒绝。不发信号、不杀容器、不调用模型。

## 运行语义

Flight仅对真实生成、连续原生OJW且具备完整行检查点的任务注入控制。调度器每次准入新样本前读请求；已准入样本继续所有角色、分支和检查点。停止状态在进程内锁存，删除请求不会使它继续准入。分批执行在cohort边界停止；单个完整node-pipeline面板作为原子准入单位，内部阶段全部排空。跨Flight共享面板明确拒绝此控制，需面板级协调器支持。

所有worker完成后，核验实际checkpoint的run/attempt、哈希链、无撕裂尾部、每个已准入样本均keepable，再原子发布`saea-admission-drained-v1`回执。出现未知响应、节点失败、写盘失败或不完整行时不发布成功排空回执。其他run/attempt或过期config的请求没有停止权限，当前运行继续；格式损坏则明确报错。不能产生错误对象的排空凭证。

`AdmissionDrained`终止完整生成路径，因此不会封存部分实验为成功。回执明确`generation_complete=false`、`generation_lease_released=false`；它不证明容器已退出或租约已清算。等待实际进程终态后，沿既有租约清算与checkpoint恢复入口建立新attempt，不重发未知调用。新attempt使用新的控制配置和路径；不要删除旧控制、回执或费用记录。

## 非功能原则与限制

控制必须持久、身份绑定、幂等、可审计；运行结清、数据完成、资源释放分别验收。已有工作优先结清，控制错误不扩大成盲重试。队列不能把已排空的非成功实验当作前序exit0，应使用既有明确交接机制。

固定镜像验证覆盖真实并发、真实Flight接线、分批边界、持久stop重启、完整检查点、未知请求及正常路径回归。没有向现有NPU实验部署或注入控制；旧冻结树仍无该能力。后继采用配置后才可使用。

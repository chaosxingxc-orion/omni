# Direct 后继启动与 CPU 预检配对示例

这是可填充的部署模板，不是已经准入或运行的实验。所有全零提交/摘要、REPLACE_WITH_REGISTERED_PROFILE 和 /srv 示例路径必须替换为实际冻结来源；不创建新模型任务，不改变预算或数据名单。

## 文件与业务入口

- direct-preflight-job.example.json：源 CPU 预检启动配置。
- direct-successor-job.example.json：已注册/准入 Direct 完整测试后继启动配置。
- direct-environment-preflight.example.json：上述两者共用的三个解释器检查，将其复制为私有 config/direct-environment-preflight.json。

final_report_workload.py 仅有 allowance / stage / prepare / register / validate，没有生成入口。Direct 完整13078的配置与注册先由它完成；prepare 验证 no-supplied-first-phase，不构造 A0 知识供应。实际生成复用 run_real_engine_candidate.py --candidate slurp-direct-fulltest-v1，沿用共享 campaign、FIT、锁、成本和封存。模板不绕过 registered_workload、readiness 或 admission。

## 三个解释器

| 角色 | 示例解释器 | 实际检查与边界 |
| --- | --- | --- |
| 业务 | /opt/saea-venv/bin/python | 导入 openjiuwen 与当前执行树 package；后继 argv[0] 使用此 Python |
| tokenizer 工具 | /usr/local/python3.12.13/bin/python3 | 导入 transformers/tokenizers/jinja2；只确认依赖可导入，不加载权重，不进行 tokenizer/FIT 调用 |
| 官方评分 | /opt/slurp-official/bin/python | SAEA_SLURP_OFFICIAL_PYTHON 显式绑定；实际导入并核对 progress 1.5、jiwer 2.0.0、tabulate 0.8.7、textdistance 4.1.5 |

当前 Direct/final-report 入口没有 tokenizer Python 选择变量，也不会因预检列出 tokenizer 解释器就切换业务 Python。需要独立 tokenizer 工具时，其自己的明确 argv 使用该已检查解释器；不能虚构 TOKENIZER_PYTHON 环境接口。普通 tokenizer 依赖版本由固定镜像绑定，模板不臆造版本号；正式运行仍需要原生 prompt-fit/模型身份校验。评分 venv 路径须替换为已有冻结环境，检查 symlink 指向的 Python 和动态库在容器内也存在。

## 必须维持的配对关系

1. stage、prepare、register、admit/repin及各阶段所需提交完成后，选最终干净执行树。然后填两份相同 expected_commit / repo_path，运行 CPU 预检。预检之后再提交会触发绑定拒绝；不能修改旧 intent 迁就新提交。
2. 两份配置的 image、environment、working_directory、repo_path、expected_commit 完全相同。即使是仅后继使用的 SAEA_DEPLOYMENT_PROFILE 和 SAEA_ADMITTED_LEVEL=96，也要提前放进源配置；完整Direct注册要求C96，2实例对应192总client宽度，不代表FIT并发或吞吐。GIT safe.directory 和 PYTHONPATH 都使用同一个容器内 study 路径。
3. CPU 源 network=none、pid_host=false；后继 network=host、pid_host=true，以复用已登记 resident 服务及真实进程身份核验。这些差异可被绑定接受，但不意味着服务核验已通过。
4. 除 /preflight 输出外，源挂载全只读，并在后继原样保留。业务源码布局为 /workspace/studies/speech-aware-evidence-acquisition，故 final_report_workload.py / run_real_engine_candidate.py 的 repo.parents[1]/docs/datasets.lock.json 对应 /workspace/docs/datasets.lock.json，不能误挂到 /study/docs。
5. 数据只读挂载拆为 /data/datasets、repos、models、knowledge、serving；后继另挂可写 /data/runs、contacts、coordination、work、prepared。Direct虽无supplied first phase，但FIT会写prepared/<FIT-run>/<attempt>/direct.request-fit.json、contacts journal和custody，所以prepared必须来自本地持久盘可写目录。serving必须指向现有resident会话凭据，否则找不到实例；不能拿空目录代替。不要源挂整个 /data 只读而后继再挂其可写子目录，现绑定会以遮蔽依赖拒绝。也不要源挂可写 /data，源依赖必须只读。
6. 模板不设置 SPEECHRL_DATA_DIR=/data：当前绑定把所有绝对环境值作为受保护路径，此设置会令后继的 /data/runs 等状态挂载被判冲突。这里显式 --data /data，现 run_real_engine_candidate.py 将其传给 build_flight_runtime；任何所选额外模块若确实要求此环境变量，必须先解决具体兼容性，不能声称本模板已覆盖该部署。SAEA_STAGE_TIMING_PATH 等可写绝对环境路径也有此问题；模板未启用它们。
7. /data/attempts 和 /data/locks 必须映射至原有唯一共享状态与 guard，不得新建空目录替代。模板中的 /srv/omni/shared-runtime 是逻辑示例，实际映射应保留原运行的唯一守护身份。项目工作区、runs、contacts、checkpoint和热日志来源必须在本地持久盘；readonly 数据来源可以是共享存储。create_directories 故意只列本次 launcher 控制目录，不自动创建共享锁或重置状态。
8. 服务身份凭据可能要求额外绝对路径、模型/解释器/动态库/proc可见性。按已登记 receipt 补充只读依赖挂载时，同时加入源与后继。模板不会重新映射 receipt 中的路径，也不代表模型可用性、全部文件可见性或设备利用率验收。

## 使用顺序

从已有固定启动器对 source 执行 layout / plan / launch，保留源容器用于 Docker inspect，等它 exit0。核对 result.json 与 ready.json，填 successor.required_inputs 中 ready、candidate、root admission 的真实摘要。后继新增数据状态目录需已按自有范围准备；最终执行树保持干净。

使用 configured_preflight_binding.py bind --source <source.json> --ready <host-ready.json> --successor <successor.json> --binding <binding.json>；再经同工具 plan / launch。layout、status、watch仍沿用原启动器。不要直接调用原 launcher.launch 跳过绑定检查。绑定文件需位于私有控制目录，配置中的路径也应保持私有。

新增示例只做本机 schema / compatibility 与漂移拒绝验证，不执行 Docker、模型、评分或tokenizer；模板不是现网已填配置。冻结外部依赖目录内容仍由部署负责，readonly 挂载不防止宿主机原位修改。

# 配置驱动的开发机异步归档

configured_remote_archive.py 在开发机运行，只通过 ssh npu2 读取固定容器状态与远端本地持久盘。远端操作只有 sudo -n docker inspect/wait、目录检查与 tar 读取；不停止进程、不改状态、不恢复、不覆盖旧归档。脚本不会主动部署或分配 NPU。

将 remote-archive.example.json 复制到私有配置目录，填写当前容器精确名称、其独占 /var/lib/omni-local-state/<job> 根、显式相对目录、原有共享 generation/FIT attempt JSON 绝对路径，以及开发机归档输出根。保留原始 shared attempts 路径，不创建替代目录。相对路径不允许重复、重叠或 ..；共享 attempt 也不得与归档目录重复覆盖。缺少的路径会明确写入 absence inventory，不以空文件冒充已归档。模板所有 example/REPLACE 值必须替换。

开发机需要 Python 3.10+、SSH；远端需要 Python 3.9+、GNU tar、timeout 和已配置的 sudo -n 读取权限。输出根必须私有（0700）；不存在时脚本按此权限创建。每个快照使用 UTC 时间和随机后缀的独立 0700 目录，包含 intent.json、state.tar、stderr.log、result.json。config/helper 摘要写入证据，tar SHA256 边接收边计算，日志和结果保持 fsync。所有内容保持私有，不直接进入公开仓。

单次归档：

    python3 scripts/environment/configured_remote_archive.py once --config /private/archive.json

异步监测（由操作者在开发机自己的控制目录脱离终端启动）：

    nohup python3 scripts/environment/configured_remote_archive.py watch --config /private/archive.json > /private/archive-watch.log 2>&1 < /dev/null &

watch 先取一份当前快照，然后等待 docker wait 明确终态或远端 timeout 600 秒，重新核验容器 ID 后归档。检测到容器被同名替换立即拒绝；每个输出根/容器由 flock 防止重复 watcher。容器已终态时完成最后归档便退出。仅状态查询，不产生模型调用；归档不会改变科学任务的终态。

result.status=complete 表示 tar 返回0且成员/attempt清单核验通过；changed-during-read 表示 tar 返回1，保留并核验已读取的内容，不能视为一致快照。任何失败都会保留已有 tar/错误日志并落盘失败结果。所有 included attempt 必须是正常大小的 JSON 对象；absent attempt 明确列出，不能把缺失记录说成已验证。即便退出0，tar也不是事务快照，consistent_snapshot 始终为false。

本工具只检查归档完整性与 attempt JSON 可解析性，不替代 scientific custody、checkpoint 哈希链或请求终态审计。它不解包文件，不提供自动恢复授权；内部链接作为归档成员保存，不跟随到外部。归档失败不会盲重试，操作者可先检查保留证据，再配置新的单次/监测任务。

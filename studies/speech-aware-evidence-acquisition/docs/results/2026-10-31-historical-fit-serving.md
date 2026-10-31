# 历史 FIT 服务身份校验准备

## 为什么需要

旧A0与rec1都没有完整flight receipt。换引擎后，不能把旧FIT记录与新服务的runtime receipt混为一谈，更不能为了通过审计修改历史session文件。

新增独立读取器 `scripts/analysis/historical_fit_serving.py`，接收由独立审查记录固定SHA256的历史快照，以及从历史FIT中取得的实际session和required identity。快照包含父session、全部member session与receipt、runtime receipt；每个文件都有相对路径和SHA256。校验模型、runtime、实例清单、父子关系、端点及进程身份一致，并拒绝越界路径、篡改或缺失记录。它不访问模型、不写历史文件。

## 已验证与边界

开发机Python标准库测试通过：1个测试含10个独立场景，覆盖正确历史身份、新runtime/session误用、快照哈希、文件篡改、端点/PID/父会话不一致、缺失receipt和路径越界。命令：`python3 tests/unit/test_historical_fit_serving.py`。

npu2固定镜像测试已请求启动，但主机故障导致启动/传输受阻，尚无通过结果。没有把开发机结果当作正式镜像验证。

本工具尚未接入全量FIT报告或运行时checkpoint兼容检查。历史快照仍需从受保护的真实记录生成并独立固定哈希；真实旧、新服务对应关系尚未验收。快照证明历史FIT服务身份，不等同于证明generation实际attachment，不授权A0接续。

下一步：主机恢复后保存并核验真实历史快照，完成固定镜像测试，再实现明确登记的新旧部署兼容证明和报告接线。原A0完整427条与全部已发生调用继续保留。

# 外部教师网络准入

批次首次运行前使用 `scripts/environment/configured_external_network.py run --config CONFIG --receipt FRESH_RECEIPT -- COMMAND`。检查使用无模型凭据的同源HEAD、验证TLS、禁止重定向，失败不执行批次命令。通过仅证明当时网络路径可用，不能证明模型能力、预算或后续请求已送达。

## 与实际请求保持一致

探针和 `configured_external_learner.py` 都支持 `proxy_mode`：默认 `inherit-environment`，使用urllib默认ProxyHandler及其HTTPS代理/no_proxy规则；显式 `direct` 使用空ProxyHandler。ExternalTeacher接受并保留该可选配置，参与原有配置身份绑定。网络配置与teacher配置应选择相同模式，在相同运行环境执行准入及批次；不要在开发机准入后声称远端容器已准入。

继承代理时，不先解析目标域名：HTTP CONNECT代理可以远程解析目标，本地目标DNS失败不代表真实路径不可用。连接、解析及TLS均由实际urllib路径完成。receipt记录模式和路径机制，不存代理URL、账号、密码或原始异常文本。`credential_supplied=false`专指API/源站凭据，代理本身可使用环境中的认证。更换代理环境会改变实际路径，receipt并非该环境永远不变的证明。

`ca_file`只用于探针指定额外信任根（例如本地TLS测试）；实际部署应与教师使用相同证书信任环境，不可用探针独有CA绕过教师证书失败。无凭据HEAD允许状态码需按实际端点配置；3xx即使误列为允许状态也拒绝。

保持旧intent/response及unknown状态不变。网络恢复不能证明历史unknown请求未发送，禁止自动重发；新配置/新批次必须按原有来源与成本规则审查。

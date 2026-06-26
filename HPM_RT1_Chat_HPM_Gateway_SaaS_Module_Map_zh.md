# HPM-RT1 Chat HPM Gateway SaaS 模块图谱

## 1. 当前本地版与 SaaS 版的区别

当前 `Chat HPM Gateway` 更接近：

```text
单机 / 私有部署 / 演示迁移 / 本地运维
```

如果要升级成真正的 SaaS 形态，核心变化不是把现有 FastAPI 放到公网就结束，而是必须补齐：

- 多租户
- 账户体系
- 远程配置
- 任务与配额
- 远程观测
- 安全边界
- 企业接入控制

---

## 2. SaaS 最少需要新增的模块

### A. 身份与权限层

需要新增：

- 用户登录
- 组织 / 工作区
- 角色权限
- API Key / Service Token
- Admin / Tenant Admin / Viewer / Operator 分级

为什么必须有：

因为本地版默认是“单操作者视角”，SaaS 场景下不同企业、不同项目、不同运维角色必须隔离。

---

### B. 多租户配置层

需要新增：

- tenant 配置隔离
- 每租户 policy profile
- 每租户 provider adapter 配置
- 每租户日志与报告目录隔离
- 每租户 connector / gateway 配置

为什么必须有：

同一套 HPM 内核服务多个客户时，不能共享配置、日志、策略和密钥。

---

### C. 密钥与 Secrets 管理层

需要新增：

- provider API key 托管
- key rotation
- secret encryption at rest
- tenant-scoped secret binding
- webhook secret / browser connector secret

为什么必须有：

现在本地版主要依赖本机环境变量；SaaS 里不能把客户密钥直接散落在进程环境中。

---

### D. 任务与队列层

需要新增：

- report job queue
- export job queue
- batch evaluation queue
- retry / backoff
- dead-letter queue

为什么必须有：

SaaS 里报告导出、批量评测、压缩归档不能全压在同步请求里，否则高并发时会拖垮主服务。

---

### E. 数据与存储层

需要新增：

- 结构化数据库
- 对象存储
- 日志归档存储
- 审计索引
- retention policy

建议拆法：

- `Postgres`：租户、用户、策略、任务、索引
- `Object Storage`：PDF、JSONL、导出包、压缩日志
- `Optional Search/Analytics`：事件检索与统计

---

### F. 配额与计费层

需要新增：

- 调用量统计
- 每租户速率限制
- 按 connector / provider / report job 计量
- plan / quota / overage 管理

为什么必须有：

没有配额层，SaaS 就没法稳定运营，也没法防止单客户压垮共享资源。

---

### G. 远程观测与告警层

需要新增：

- 集中化 health dashboard
- centralized diagnostics
- tenant-level usage analytics
- service alerting
- deploy / rollback visibility

为什么必须有：

本地版的健康检查已经不错，但 SaaS 需要的是“跨实例、跨租户、跨作业”的统一观测。

---

### H. 远程浏览器 / Connector 管理层

需要新增：

- connector 注册中心
- connector 版本管理
- connector capability compatibility matrix
- connector heartbeat
- connector remote disable / revoke

为什么必须有：

如果客户安装浏览器插件或网关连接器，SaaS 必须知道哪些 connector 在线、兼容哪个版本、是否需要禁用或升级。

---

### I. 企业接入与集成层

需要新增：

- Webhook
- REST API token access
- SSO / SAML / OIDC
- SIEM / SOC export
- ticketing / incident integration

为什么必须有：

企业最终不会只看一个前端页面，他们会要求对接现有 IAM、日志平台和安全系统。

---

## 3. SaaS 架构下的推荐模块分层

建议未来拆成：

```text
1. Public API Gateway
2. Auth / Tenant Service
3. Trust Runtime Core
4. Job Orchestrator
5. Report / Export Worker
6. Connector Control Plane
7. Config / Policy Service
8. Audit Event Store
9. Diagnostics & Observability Service
10. Billing / Quota Service
```

---

## 4. 当前代码基础里哪些能直接复用

当前本地版可直接复用的核心：

- `precheck / postaudit / gateway` trust runtime contract
- provider registry
- policy registry
- diagnostics / health / events / report jobs 语义层
- browser connector manifest / diagnostics 骨架
- packaging / deployment profile 思路

也就是说，当前工程并不是“推倒重来”，而是已经有 SaaS trust core 的雏形。

---

## 5. 当前还不能直接拿去做 SaaS 的部分

目前还缺：

- 真正的用户体系
- 真正的多租户隔离
- 持久化数据库模型
- 任务队列与 worker
- secret 管理
- 集中化告警
- 计费/配额
- 企业级鉴权与审计策略

---

## 6. 如果现在要进入 SaaS 路线，建议的最小增量

我建议不要一上来就做完整 SaaS，而是先做：

### SaaS Phase 1

- 加 `tenant` 概念
- 加 `API key`
- 加 `Postgres` 配置存储
- 加异步 `report/export worker`
- 加对象存储
- 加远程 health / diagnostics 汇总

这一步做完，产品会从：

```text
本地可迁移 runtime
```

升级为：

```text
可托管的单区域 SaaS alpha
```

---

## 7. 对外表述建议

可以这样说：

```text
HPM-RT 当前已完成 Chat HPM Gateway 的本地产品化闭环；
如果进入 SaaS 路线，下一步将不是重写内核，
而是在现有 trust runtime 之上补齐租户、队列、存储、观测、权限与计费等控制平面模块。
```

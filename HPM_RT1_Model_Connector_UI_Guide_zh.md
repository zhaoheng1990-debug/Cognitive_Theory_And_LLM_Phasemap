# HPM-RT1 模型接入设置页使用说明

适用版本：`v1.8.6+`

## 1. 入口位置

启动前端后，进入：

`安装与部署 -> 模型接入设置`

这里可以直接完成三类接入：

1. 本地 `Transformers` 小模型
2. 本地 `Ollama` 小模型
3. 在线 `OpenAI-compatible API`

---

## 2. 三种常用接法

### A. 本地 Qwen / DeepSeek + Transformers

适合：

- 已有本地模型目录
- 希望输入侧预检稳定启用
- 不想额外安装 Ollama

需要填写：

- `接入方案`：本地 Transformers
- `本地回答模型`
- `本地模型目录`
- `模型标识`

保存后会同时更新：

- `runtime.default_provider`
- `providers.<local provider>`
- `premise_auditor.transformers.model_path`

---

### B. 本地 Ollama

适合：

- 本机已安装 Ollama
- 已经能直接运行 `ollama list`
- 想统一走本地 HTTP 接口

需要填写：

- `接入方案`：本地 Ollama
- `本地回答模型`
- `Ollama 模型名`
- `Ollama Chat 接口`

如果输入侧也想走 Ollama，小面板会一起保存前置审计模型与接口。

---

### C. 在线 API + 本地前置审计

适合：

- 回答仍然走 GPT / Gemini / Kimi / Doubao / MiniMax / DeepSeek API
- 输入风险审计仍然需要本地小模型托底

推荐组合：

- `接入方案`：在线 API
- `在线回答模型`：选目标厂商
- `API Endpoint`
- `模型标识`
- `API Key 环境变量名`
- `API Key`
- `输入侧前置审计器`：优先选本地 Transformers

说明：

- API Key 不写入 `config.yaml`
- API Key 会保存到 `runtime.env`
- 启动器下次启动时会自动加载 `runtime.env`

---

## 3. runtime.env 是什么

这是新的运行时密钥文件，用来保存：

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- 其他在线模型凭据

它的作用是避免把敏感密钥直接写进 `config.yaml`。

当前版本中：

- 保存在线 API 配置时，如果填了 API Key，会自动生成或更新 `runtime.env`
- API 服务当前进程会立即载入它
- 启动器重启后也会继续自动载入它

---

## 4. 建议使用顺序

1. 先在“模型接入设置”里保存接法
2. 点击“刷新当前接入状态”
3. 看“连通性状态”是否变成“就绪”
4. 再运行“安装前预检”或直接做一次真实 `gateway` 调用

---

## 5. 当前边界

这次升级已经解决的是：

- 本地小模型接入入口
- 在线 API 接入入口
- 迁移后配置持久化
- API Key 持久化

还没完全做完的是：

- 前置审计本地小模型的风险校准
- 多在线厂商 endpoint 的更细粒度模板化
- 一键切换多套企业部署 profile

所以现在它已经是“可部署、可切换、可维护”的版本，但还不是最终形态。

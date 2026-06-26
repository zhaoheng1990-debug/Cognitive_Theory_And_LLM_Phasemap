# HPM-RT1 本地基模与在线基模接入指南

版本：`v1.8.6+`  
适用范围：`Chat HPM Gateway`

## 1. 为什么这一步是必须的

迁移后如果不接入本地小参数基模，HPM-RT1 的输入风险审计会退回到纯规则层。  
这并不是完全不可用，但在以下场景里能力会明显下降：

- 复杂虚假前提抽取
- 多重嵌套 presupposition 识别
- 细粒度 prompt claim 切分
- 高压缩表达里的风险对象抽取

因此，推荐至少接入一条“本地 premise auditor”链路；  
如果还需要让网关代调用一个在线/本地大模型生成答案，再额外配置 provider 即可。

---

## 2. HPM-RT1 里有两条不同接入面

### A. 输入风险审计基模

作用：

- 只服务于 `premise_integrity`
- 用来辅助抽取 prompt 中的 factual premises
- 不替代事实核查
- 不等于最终回答模型

配置块：

```yaml
premise_auditor:
  enabled: true
  mode: transformers | ollama
```

### B. Gateway 生成模型 provider

作用：

- 供 `/v1/gateway` 调用生成答案
- 支持本地模型或在线 API
- 由 `runtime.default_provider` 选择默认项

配置块：

```yaml
runtime:
  default_provider: openai_gpt_api

providers:
  openai_gpt_api:
    ...
```

---

## 3. 当前支持的接入方式

## 3.1 本地 premise auditor

当前支持两种：

1. `transformers`
2. `ollama`

### 推荐顺序

如果你本机已经有可直接读取的模型目录，优先：

```text
transformers
```

如果你已经把模型挂到 Ollama，优先：

```text
ollama
```

---

## 3.2 Gateway provider

当前支持：

1. `local_transformers`
2. `ollama`
3. `openai_compatible`

其中 `openai_compatible` 可用于对接：

- OpenAI GPT
- Gemini（若供应商提供 OpenAI-compatible chat 接口）
- DeepSeek
- Kimi / Moonshot
- Doubao
- MiniMax
- 其他兼容 `chat/completions` 协议的在线模型服务

注意：

不同厂商 endpoint 可能调整，因此本项目不硬编码所有厂商最新正式地址。  
请以对应厂商官方文档中的最新 chat-completions 接口为准，把 endpoint 填进配置即可。

---

## 4. 最推荐的三种落地方式

## 4.1 方式一：Qwen 本地目录 + Transformers

适合：

- 你已有本地 Qwen 模型目录
- 不想额外装 Ollama

配置示例：

```yaml
premise_auditor:
  enabled: true
  mode: transformers
  apply_to_gateway_audit: true
  transformers:
    model_path: "D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main"
    device_map: auto
    trust_remote_code: false

runtime:
  default_provider: qwen_local_transformers

providers:
  qwen_local_transformers:
    type: local_transformers
    model_provider: local_transformers
    model_family: qwen
    model_id: Qwen2.5-Instruct
    deployment_type: local
    parameter_scale: small
    model_path: "D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main"
    device_map: auto
    trust_remote_code: false
    max_new_tokens: 256
```

依赖：

```bash
pip install transformers torch
```

---

## 4.2 方式二：DeepSeek / Qwen 挂载到 Ollama

适合：

- 你已在 Ollama 中拉起模型
- 想用统一本地 API 方式管理

配置示例：

```yaml
premise_auditor:
  enabled: true
  mode: ollama
  apply_to_gateway_audit: true
  ollama:
    model: qwen2.5:3b-instruct
    endpoint: http://127.0.0.1:11434/api/generate
    timeout_seconds: 6

runtime:
  default_provider: local_ollama_qwen

providers:
  local_ollama_qwen:
    type: ollama
    model_provider: local_ollama
    model_family: qwen
    model_id: qwen2.5:3b-instruct
    deployment_type: local
    parameter_scale: small
    model: qwen2.5:3b-instruct
    endpoint: http://127.0.0.1:11434/api/chat
    timeout_seconds: 60
```

DeepSeek 只需把模型名改成你在 Ollama 里实际可用的 tag，例如：

```yaml
model: deepseek-r1:7b
```

---

## 4.3 方式三：输入审计走本地小模型，生成走在线 API

这是当前最推荐的工程组合。

原因：

- 输入审计敏感，放本地更稳定
- 回答生成可继续用 GPT / Gemini / Kimi / Doubao / MiniMax
- 兼顾成本、隐私和效果

配置示例：

```yaml
premise_auditor:
  enabled: true
  mode: transformers
  apply_to_gateway_audit: true
  transformers:
    model_path: "D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main"
    device_map: auto
    trust_remote_code: false

runtime:
  default_provider: openai_gpt_api

providers:
  openai_gpt_api:
    type: openai_compatible
    model_provider: openai
    model_family: gpt
    model_id: gpt-4.1-mini
    deployment_type: api
    parameter_scale: api
    endpoint: https://api.openai.com/v1/chat/completions
    api_key_env: OPENAI_API_KEY
    timeout_seconds: 60
```

---

## 5. 在线基模 API 接入方法

## 5.1 通用配置模板

```yaml
runtime:
  default_provider: vendor_api

providers:
  vendor_api:
    type: openai_compatible
    model_provider: vendor_name
    model_family: api
    model_id: your-model-id
    deployment_type: api
    parameter_scale: api
    endpoint: https://your-official-chat-completions-endpoint
    api_key_env: YOUR_VENDOR_API_KEY
    timeout_seconds: 60
```

然后在系统环境变量中设置：

```powershell
$env:YOUR_VENDOR_API_KEY="your-real-key"
```

或永久设置为系统环境变量。

## 5.2 已预留的在线 provider 名称

当前 `config.yaml` 已预留以下名字，便于直接改：

- `openai_gpt_api`
- `gemini_api`
- `deepseek_api`
- `kimi_api`
- `doubao_api`
- `minimax_api`

注意：

这些 provider 项的 `endpoint` 有些默认留空，是故意的。  
因为厂商接口地址和接入路径可能更新，部署时应填写你当前使用的官方地址，而不是依赖仓库里写死的旧地址。

---

## 6. 一键生成接入模板

项目新增了一个接入工具：

[tools/hpm_rt1_model_connector_setup.py](C:/Users/ZH/Documents/HPM_RT1工程化验证/tools/hpm_rt1_model_connector_setup.py)

### 列出所有可用模板

```powershell
.\.venv\Scripts\python.exe .\tools\hpm_rt1_model_connector_setup.py list
```

### 写出一个本地模板文件

```powershell
.\.venv\Scripts\python.exe .\tools\hpm_rt1_model_connector_setup.py write-snippet --profile local_qwen_transformers --output .\outputs\connector_snippets\local_qwen_transformers.json
```

### 做一次接入预检

```powershell
.\.venv\Scripts\python.exe .\tools\hpm_rt1_model_connector_setup.py preflight --config .\config.yaml
```

它会同时输出：

- 当前 provider 是否 ready
- premise auditor 是否启用
- premise auditor 当前模式

---

## 7. 接好以后怎么验证

## 7.1 看健康面板

访问：

```text
/health
/v1/providers
```

重点看：

- provider 是否 ready
- local premise auditor 是否 enabled
- 当前 mode 是 `transformers` 还是 `ollama`

## 7.2 看审计输出

调用 `/gateway/audit` 或浏览器连接器后，检查输出中：

```json
"auditor": {
  "mode": "transformers",
  "local_model_used": true,
  "status": "ok"
}
```

如果还是：

```json
"mode": "rule_based"
```

说明本地审计基模没有真正接上。

---

## 8. 当前建议的企业默认方案

如果目标是迁移后尽快恢复输入风险审计能力，建议默认采用：

```text
本地 premise auditor：Qwen 1.5B / 3B，transformers 或 Ollama
在线生成 provider：GPT / DeepSeek / Kimi / Doubao / MiniMax / Gemini
```

这个组合的优点是：

- 输入风险审计不依赖外部网络
- 本地成本可控
- 在线大模型仍保持更强回答能力
- HPM 的输入预警和输出审计都能恢复到更完整状态

---

## 9. 边界说明

接入本地小参数基模后，HPM-RT1 的输入风险审计会显著增强；  
但它依然是：

```text
黑盒外部 trust gateway
```

这不等于：

- 读取闭源模型内部推理参数
- 拿到 hidden state
- 从根本上替代模型侧内部幻觉控制

它的价值在于：

```text
在现实可部署约束下，把输入预警和输出审计做强
```

# HPM-RT1 在线审计模型快速接入指南

适用目标：

- 让在线模型同时参与输入侧审计和输出侧审计
- 不替换现有主回答模型也可以单独部署
- 适合纯本地模型不足、但允许外部 API 审计的场景

## 1. 当前已支持的在线审计接入方式

HPM-RT1 现在支持把在线模型作为：

- 输入侧前提审计器 `premise_auditor.mode = online_api`
- 输出侧回答审计器 `response_auditor.mode = online_api`

两者可以指向同一个在线模型。

## 2. 已内置的默认 Endpoint

当前前端“模型接入设置”页已经预置以下默认接口：

- OpenAI GPT  
  `https://api.openai.com/v1/chat/completions`
- Google Gemini（OpenAI 兼容层）  
  `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
- DeepSeek API  
  `https://api.deepseek.com/chat/completions`
- Moonshot Kimi  
  `https://api.moonshot.cn/v1/chat/completions`
- Doubao / 火山方舟  
  `https://ark.cn-beijing.volces.com/api/v3/chat/completions`
- MiniMax  
  `https://api.minimaxi.com/v1/chat/completions`

说明：

- 模型名可以在前端自行改
- API Key 不写入 `config.yaml`
- API Key 写入 `runtime.env`

## 3. 最短配置步骤

### 方法 A：前端配置

进入：

`安装与部署 -> 模型接入设置`

选择：

- `接入方案` -> `在线 API`
- `审计模型接入方式` -> `在线 API（输入+输出）`

然后填写：

- 回答模型 Endpoint / 模型标识 / API Key
- 审计模型 Endpoint / 模型标识 / API Key

如果回答模型和审计模型相同，可以直接复用同一套配置。

### 方法 B：手动准备 runtime.env

复制：

`runtime.env.example -> runtime.env`

只填写你实际要用的提供方，例如：

```env
DEEPSEEK_API_KEY=your-real-key
```

## 4. 推荐的首轮接法

如果目标是先把在线审计跑起来，我建议优先用：

1. `DeepSeek API`
2. `OpenAI GPT`
3. `Kimi`

原因：

- 接口相对稳定
- 与当前 HPM OpenAI-compatible 适配器最贴近
- 联调成本低

## 5. 配置完成后如何验证

前端里直接做两次验证：

### 输入侧验证

点击：

`运行真实预检验证`

重点看：

- `是否要求先核查前提`
- `前提风险`
- `审计模式`
- `审计器状态`

### 输出侧验证

点击：

`运行真实后验审计验证`

重点看：

- `输出侧审计模型已接入`
- `回答可信度`
- `风险分数`
- `在线审计可信度`
- `在线审计风险`

## 6. 当前边界

在线审计模型当前能做的是：

- 抽取输入中的前提命题
- 判断是否需要先核查
- 对输出做结构化后验审计
- 返回 `trust_score / risk_score / recommended_action`

当前不能做的是：

- 读取目标大模型内部隐藏状态
- 获得 provider 内部 CoT
- 获得 provider 私有检索链路

所以它仍然属于：

`黑盒外在线审计层`

## 7. 建议的企业部署策略

如果企业允许外部 API：

- 主回答模型：在线大模型
- 输入侧审计：在线审计模型
- 输出侧审计：在线审计模型

如果企业不允许敏感输入出网：

- 主回答模型：在线大模型
- 输入侧审计：本地小模型
- 输出侧审计：本地规则 + 本地小模型 / 受控在线审计

## 8. 下一步

完成接入后，建议立刻跑：

- 一次输入侧高风险 probe
- 一次输出侧高风险 probe
- 一次完整 `/v1/gateway` 联调

这样就能确认：

- 配置是否生效
- 在线审计模型是否真的参与了输入和输出两侧
- 风险分数是否进入最终策略决策

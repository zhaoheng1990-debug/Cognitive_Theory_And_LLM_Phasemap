# HPM-RT1 v1.8.6 全流程安装与激活 Guide

适用对象：

- 终端用户
- 企业测试人员
- 交付实施人员
- 使用 `Codex`、`Claude Code`、`Work Buddy` 等智能助理协助安装的人

这份文档的目标只有一个：

```text
把 HPM-RT1 从“拿到安装包”，一路带到“系统启动、插件激活、审计模型接入、联调验证通过”。
```

---

## 0. 先看结论

如果你只想快速跑起来，按这 6 步做：

1. 把安装包解压到短路径目录，例如 `C:\HPM_RT1\`
2. 运行 `HPM_RT1_OneClick_Setup.bat`
3. 打开 `http://127.0.0.1:8501`，确认审计台正常
4. 安装浏览器插件 `browser_extension\hpm_rt1_gateway_connector`
5. 复制 `runtime.env.example` 为 `runtime.env`，填入在线模型 API Key，或在审计台接入本地模型
6. 在审计台运行 `precheck probe` 和 `postaudit probe`，再去 ChatGPT 等网页做一次联调

如果你是用 IDE 助手代做，直接把这句发给它：

```text
请严格按 HPM_RT1_v1_8_6_Full_Install_and_Activation_Guide_zh.md 完成安装、插件激活、模型接入、探针验证和首轮联调，不要跳步，每完成一步都先自检再继续。
```

---

## 1. 你拿到的是哪种包

HPM-RT1 v1.8.6 当前有两类交付包。

### A. `Portable Package`

特点：

- 不带 `.venv`
- 体积更小
- 更适合开发、评估、二次集成、技术交付

### B. `Runtime Package`

特点：

- 带 `.venv`
- 更适合演示、迁移、快速落地
- 如果目标机器环境差异较大，仍可能需要重建 `.venv`

如果你不确定该用哪个，默认优先：

```text
Runtime Package
```

---

## 2. 推荐安装位置

建议解压到短路径目录：

```text
C:\HPM_RT1\
或
D:\HPM_RT1\
```

不建议解压到：

- 系统临时目录
- 路径特别深的中文目录
- 企业强权限同步盘目录
- 经常被清理的下载缓存目录

---

## 3. 安装前检查

在开始前，确认下面几项：

1. 系统是 Windows 10 或 Windows 11
2. 当前用户有安装软件或写入本地目录的权限
3. 浏览器支持安装“已解压扩展”
4. 如果要使用在线审计模型，机器能访问对应模型 API
5. 如果要使用本地审计模型，机器已有本地模型、Ollama 或 Transformers 环境

---

## 4. 推荐安装方式

### 标准入口

优先运行：

```text
HPM_RT1_OneClick_Setup.bat
```

它会自动完成：

1. 安装或修复 VC++ 运行库
2. 安装 Python 3.13.14（仅当系统里没有可用 Python 3.10+）
3. 创建或修复 `.venv`
4. 安装运行依赖
5. 生成部署配置
6. 启动 HPM-RT1

### 兼容入口

包内也保留了：

```text
HPM_RT1_OneClick_Setup.exe
```

但跨机器迁移时，仍建议优先使用 `.bat` 入口。

---

## 5. 安装失败时的处理顺序

如果一键安装没有完成，不要来回乱试，按这个顺序排查：

### 第一步：运行安装诊断

```text
HPM_RT1_Run_Installation_Doctor.bat
```

### 第二步：看安装与运行日志

重点看这几个文件：

```text
outputs\installer\one_click_install_summary.txt
outputs\hpm_rt1_api.log
outputs\hpm_rt1_streamlit.log
```

### 第三步：如果 `.venv` 不可复用

运行：

```text
HPM_RT1_Install_Prerequisites_If_Venv_Broken.bat
```

然后重新执行：

```text
HPM_RT1_OneClick_Setup.bat
```

---

## 6. 安装完成后的启动方式

### 推荐日常启动入口

```text
HPM_RT1_Start_Runtime.bat
```

### 兼容入口

```text
HPM_RT1_Launcher.exe
```

### 启动后默认会拉起

- FastAPI Gateway：`http://127.0.0.1:8000`
- Streamlit 审计台：`http://127.0.0.1:8501`

---

## 7. 第一次自检：系统是不是已经真的起来了

### 检查 1：审计台是否可打开

在浏览器里访问：

```text
http://127.0.0.1:8501
```

如果能看到 HPM-RT1 中文审计台，说明前端已正常。

### 检查 2：API 是否在线

访问：

```text
http://127.0.0.1:8000/health
```

部分部署还可用：

```text
http://127.0.0.1:8000/v1/admin/health
```

### 检查 3：后台日志是否在写入

看：

```text
outputs\hpm_rt1_api.log
outputs\hpm_rt1_streamlit.log
```

---

## 8. 浏览器插件安装与激活

插件目录在：

```text
browser_extension\hpm_rt1_gateway_connector\
```

### Chrome / Edge 安装步骤

1. 打开扩展管理页
2. 开启“开发者模式”
3. 选择“加载已解压的扩展程序”
4. 选中目录：

```text
browser_extension\hpm_rt1_gateway_connector
```

5. 确认扩展名称显示为：

```text
HPM-RT1 Gateway Connector
```

### 激活后的检查方式

1. 先启动 HPM-RT1 Runtime
2. 打开 `ChatGPT / Gemini / Kimi / Doubao / Claude / DeepSeek / MiniMax`
3. 输入一个问题
4. 右下角应出现 HPM-RT1 实时审计卡片

如果没有出现：

1. 确认 Gateway 已启动
2. 在扩展管理页点“重新加载”
3. 刷新目标网页

---

## 9. HPM-RT1 现在支持哪些审计接法

v1.8.6 当前支持：

1. 规则层基础审计
2. 本地模型输入侧审计
3. 在线模型输入侧审计
4. 在线模型输出侧后验审计

你可以这样理解：

- `/v1/precheck`：回答前先看输入有没有风险
- `/v1/postaudit`：回答后再看输出是否可信
- `/v1/gateway`：把“输入 -> 模型 -> 输出 -> 审计”整条链串起来

---

## 10. 推荐的模型接入策略

如果你是第一次部署，建议优先级如下：

### 方案 A：在线模型接入

优点：

- 部署快
- 不依赖本地 GPU
- 适合先跑通输入侧和输出侧双审计

### 方案 B：本地模型接入

优点：

- 不依赖外网
- 更适合企业内网或本地验证

### 推荐顺序

```text
先接在线模型 -> 跑通全链路 -> 再按需要补本地模型
```

---

## 11. 接入在线模型 API 审计

### 第一步：准备密钥文件

复制：

```text
runtime.env.example
```

为：

```text
runtime.env
```

然后填入你需要的密钥，例如：

```env
DEEPSEEK_API_KEY=你的密钥
OPENAI_API_KEY=
GEMINI_API_KEY=
KIMI_API_KEY=
DOUBAO_API_KEY=
MINIMAX_API_KEY=
```

### 第二步：了解加载顺序

HPM-RT1 启动时会按顺序读取：

1. `C:\Users\<用户名>\.codex\runtime.env`
2. 当前项目目录下的 `runtime.env`

项目目录下的 `runtime.env` 优先级更高，适合单项目覆盖。

### 第三步：在审计台里接入

进入“模型接入 / Model Connector”区域，选择：

```text
在线 API（输入+输出）
```

然后配置：

- Provider family
- Model ID
- API endpoint
- API key env

当前已适配主流在线模型族：

- OpenAI
- Gemini
- DeepSeek
- Kimi
- Doubao
- MiniMax

---

## 12. 接入本地模型审计

HPM-RT1 当前支持两种本地方式。

### 方式 A：Transformers

适合手头已有 HuggingFace 本地模型目录的场景。

常见环境变量：

```env
HPM_RT1_PREMISE_AUDITOR=transformers
HPM_RT1_PREMISE_AUDITOR_MODEL_PATH=你的本地模型目录
```

### 方式 B：Ollama

适合机器上已经跑着 Ollama 的场景。

示例：

```env
HPM_RT1_PREMISE_AUDITOR=ollama
HPM_RT1_PREMISE_AUDITOR_MODEL=qwen2.5:7b
```

### 实际建议

如果不是做底层调试，更推荐直接在审计台的模型接入页里完成配置。

---

## 13. 如何验证输入侧审计已经接上

在审计台里运行：

```text
precheck probe
```

重点看这几项：

- 前置审计器已调用
- 审计器位置
- 审计模型
- 建议动作
- 风险分数

判断方式：

- 如果走在线审计，应看到“审计器位置：在线”
- 如果走本地审计，应看到“审计器位置：本地”

---

## 14. 如何验证输出侧后验审计已经接上

在审计台里运行：

```text
postaudit probe
```

重点看：

- 输出侧审计模型已接入
- 回答可信度
- 风险分数
- 建议动作
- 风险原因

如果在线输出侧审计接入成功，通常应看到：

```text
response_auditor_status = ok
```

---

## 15. 插件 + Runtime 联调验证

建议做一次完整链路验证：

1. 启动 `HPM_RT1_Start_Runtime.bat`
2. 确认浏览器插件已启用
3. 打开 ChatGPT 或其他目标网页
4. 输入一个高风险测试问题
5. 观察右下角是否出现实时审计卡片
6. 回到审计台确认日志、告警、建议动作是否已写入

---

## 16. API 直接调用示例

### 输入前置审计

```powershell
curl -X POST http://127.0.0.1:8000/v1/precheck ^
  -H "Content-Type: application/json" ^
  -d "{\"input_text\":\"请介绍2025年最新出台的某项尚未发布政策\"}"
```

### 输出后验审计

```powershell
curl -X POST http://127.0.0.1:8000/v1/postaudit ^
  -H "Content-Type: application/json" ^
  -d "{\"input_text\":\"请介绍某政策\",\"output_text\":\"该政策于2025年正式发布...\"}"
```

### 完整 Gateway

```powershell
curl -X POST http://127.0.0.1:8000/v1/gateway ^
  -H "Content-Type: application/json" ^
  -d "{\"input_text\":\"请帮我判断一个尚未核实的政策事实\"}"
```

---

## 17. 日志、报告和导出物在哪里

默认重点目录：

```text
outputs\
outputs\hpm_rt1_api_output\
outputs\hpm_rt1_frontend_output\
outputs\installer\
```

桌面 PDF 报告默认目录：

```text
%USERPROFILE%\Desktop\HPM_RT1_Gateway_Audit_Reports\
```

---

## 18. 给 Codex / Claude Code / Work Buddy 的标准执行顺序

如果你要让智能助理代你安装，建议它严格按这个顺序执行：

1. 解压安装包到短路径目录
2. 执行 `HPM_RT1_OneClick_Setup.bat`
3. 检查 `http://127.0.0.1:8000/health`
4. 检查 `http://127.0.0.1:8501`
5. 指导安装并重载浏览器插件
6. 复制 `runtime.env.example -> runtime.env`
7. 根据需要接入在线或本地审计模型
8. 运行 `precheck probe`
9. 运行 `postaudit probe`
10. 打开目标网页做首轮联调

### 可直接复制给 IDE 助手的指令

```text
请按 HPM_RT1_v1_8_6_Full_Install_and_Activation_Guide_zh.md 的顺序完成 HPM-RT1 安装、插件激活、在线或本地审计模型接入、precheck/postaudit 探针验证和首轮联调。每一步完成后先自检，再进入下一步；如果失败，优先查看 outputs 目录下日志，不要跳步。
```

---

## 19. 最终验收标准

安装完成后，至少满足以下 8 条：

1. `HPM_RT1_Start_Runtime.bat` 能正常启动
2. `http://127.0.0.1:8000/health` 可访问
3. `http://127.0.0.1:8501` 可访问
4. 浏览器插件已安装并启用
5. 右下角实时审计卡片可出现
6. `/v1/precheck` 能返回结果
7. `/v1/postaudit` 能返回结果
8. 本地或在线审计模型至少成功接入一种

---

## 20. 边界声明

HPM-RT1 v1.8.6 仍然是：

```text
黑盒外 AI Trust Gateway / Advisory Runtime
```

它能做的是：

- 输入侧风险预警
- 输出侧后验审计
- 日志、PDF、趋势和验收支撑

它当前不做的是：

- 读取在线大模型内部推理参数
- 等同于模型内生式幻觉控制
- 默认自动改写模型回答

---

## 21. 还可以继续看的配套文档

如果需要更细的局部说明，可以继续看：

- `HPM_RT1_OneClick_Installer_Guide_zh.md`
- `HPM_RT1_Chat_HPM_Gateway_Package_Guide_zh.md`
- `HPM_RT1_Online_Auditor_Quickstart_zh.md`
- `HPM_RT1_Model_Connector_UI_Guide_zh.md`
- `HPM_RT1_Local_and_Online_Base_Model_Integration_Guide_zh.md`

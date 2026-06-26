# HPM-RT1 可迁移安装说明

版本：v1.1

定位：LLM 黑盒外实时预警与审计层。该工具只读取可见输入、可见输出、证据项和页面/API 元数据；不读取模型内部推理过程、logits、hidden states、attention 或服务商侧检索链路。

## 1. 环境

建议 Python 3.10 以上。Windows 迁移机器需要能通过 `py -3` 或 `python` 找到 Python。

## 2. Windows 三 exe 推荐流程

解压迁移包后，在项目根目录按顺序运行：

```powershell
.\HPM_RT1_Install_Dependencies.exe
.\HPM_RT1_Deploy_Modules.exe
.\HPM_RT1_Launcher.exe
```

三个 exe 的职责分别是：

- `HPM_RT1_Install_Dependencies.exe`：创建本地 `.venv`，优先从 `wheelhouse/` 离线安装依赖，再安装 HPM-RT1 模块。
- `HPM_RT1_Deploy_Modules.exe`：检查核心模块、API、前端是否齐全，创建输出目录、运行配置和桌面快捷方式。
- `HPM_RT1_Launcher.exe`：启动 FastAPI gateway 与 Streamlit 中文前端。

启动后访问：

- API：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:8501`
- Gateway PDF 日志：桌面 `HPM_RT1_Gateway_Audit_Reports/hpm_rt1_gateway_audit_report.pdf`

## 3. Web 端大模型过程预警接入

项目内置浏览器扩展连接器：

```text
browser_extension/hpm_rt1_gateway_connector
```

安装方式：

1. 先运行 `HPM_RT1_Launcher.exe`，确认本地 gateway 已启动。
2. 打开 Chrome/Edge 扩展管理页。
3. 打开开发者模式。
4. 选择“加载已解压的扩展”。
5. 选择 `browser_extension/hpm_rt1_gateway_connector` 目录。

扩展会执行三段式过程预警：

- 发送前预检：输入框内容会发送到 `/audit`，用于 prompt 风险预警。
- 生成中监听：页面中正在增长的回答会发送到 `/audit`，用于近实时风险提示。
- 完成后审计：完整问答会发送到 `/gateway/audit`，用于生成 JSONL 日志和桌面 PDF 报告。

当前覆盖 ChatGPT、Gemini、Kimi、Doubao、Claude 等主流网页域名。

## 4. 手动安装

在线环境：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

离线环境可使用包内 `wheelhouse/`：

```bash
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python -m pip install -e .
```

或安装完整开发依赖：

```bash
python -m pip install -e ".[dev]"
```

## 5. 手动启动 API

```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

访问：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 6. 手动启动前端

```bash
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

访问：

```text
http://127.0.0.1:8501
```

## 7. 验证

```bash
python -m pytest
python -m hpm_rt1_beta.cli --input examples/hpm_rt1_external_testset_min32.jsonl --output-dir outputs/cli_external_min32_output
```

## 8. 边界

HPM-RT1 是 advisory-only 风险治理模块：

```text
不自动改写答案
不做事实自动修正
不更新模型权重
不激活 HPM-12 repair
```

每个审计输出包含 `audit_confidence_scores`，所有分数范围均为 0-1。
Gateway 实时审计会额外生成桌面 PDF 报告：

```text
桌面\HPM_RT1_Gateway_Audit_Reports\hpm_rt1_gateway_audit_report.pdf
```

浏览器扩展只读取网页中已经显示的文本；不接管登录、不修改模型回答、不绕过平台权限。

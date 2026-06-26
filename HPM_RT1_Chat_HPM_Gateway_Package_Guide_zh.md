# HPM-RT1 Chat HPM Gateway 打包说明

## 当前交付形态

当前 `Chat HPM Gateway` 提供两类可迁移交付物：

1. `portable` 无依赖包
2. `runtime` 有依赖包

它们都服务于同一条产品主线：

```text
Chat HPM Gateway
```

默认部署边界：

```text
chat_hpm: true
agent_hpm: false
ide_hpm: false
```

## portable 包

适合：

- 技术团队评估
- 开发调试
- API / 前端 / 插件联调
- 轻量分发

特点：

- 不带 `.venv`
- 体积更小
- 环境依赖略高

推荐顺序：

1. `HPM_RT1_OneClick_Setup.bat`
2. `HPM_RT1_Start_Runtime.bat`

## runtime 包

适合：

- 演示
- 迁移
- 现场快速验证

特点：

- 带 `.venv`
- 开箱更快
- 跨机器时如环境差异过大，仍可能需要重建 `.venv`

推荐顺序：

1. `HPM_RT1_OneClick_Setup.bat`
2. `HPM_RT1_Start_Runtime.bat`

如果 `.venv` 不可复用，再运行：

3. `HPM_RT1_Install_Prerequisites_If_Venv_Broken.bat`
4. 重新执行 `HPM_RT1_OneClick_Setup.bat`

## v1.8.6 新增重点

这次收口进包的重点包括：

- 新增在线输入侧审计 API 接入
- 新增在线输出侧后验审计 API 接入
- 审计台模型接入页更新
- 浏览器插件版本对齐到 `1.8.6`
- `runtime.env.example` 入包
- 新增完整总引导文档

## 推荐先读

```text
HPM_RT1_v1_8_6_Full_Install_and_Activation_Guide_zh.md
```

## 当前边界声明

当前版本仍属于：

```text
黑盒外 AI Trust Gateway / Advisory Runtime
```

也就是：

- 可以做输入前置风险预警
- 可以做输出后验审计
- 可以输出日志、PDF 报告、趋势和验收结果
- 不读取在线大模型内部参数
- 不等同于模型内生式幻觉控制

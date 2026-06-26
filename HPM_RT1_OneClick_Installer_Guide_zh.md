# HPM-RT1 一键安装说明

适用场景：希望在一台较干净的 Windows 机器上，尽量少折腾地完成安装、部署和首次启动。

## 默认入口

优先双击：

```text
HPM_RT1_OneClick_Setup.bat
```

它会按顺序完成：

1. 安装或修复 VC++ 运行库
2. 安装 Python 3.13.14（仅当系统里没有可用 Python 3.10+）
3. 创建或修复 `.venv`
4. 安装运行依赖
5. 生成本地部署配置
6. 启动 HPM-RT1

## 兼容入口

包内仍保留：

```text
HPM_RT1_OneClick_Setup.exe
```

但迁移到新机器时，仍推荐优先使用 `.bat` 入口。

## 日常启动

安装完成后，推荐使用：

```text
HPM_RT1_Start_Runtime.bat
```

兼容入口：

```text
HPM_RT1_Launcher.exe
```

## 建议先读的总文档

如果你需要从安装一路看到插件激活、API 使用和模型接入，请优先阅读：

```text
HPM_RT1_v1_8_6_Full_Install_and_Activation_Guide_zh.md
```

## 关键目录

```text
prereqs/      前置环境安装器
wheelhouse/   离线依赖轮子
outputs/      安装、运行、日志输出
```

## 如果安装失败

先看：

```text
HPM_RT1_Run_Installation_Doctor.bat
outputs/installer/one_click_install_summary.txt
outputs/hpm_rt1_api.log
outputs/hpm_rt1_streamlit.log
```

## 边界说明

一键安装器只是交付入口，不改变 HPM-RT1 的运行边界：

```text
黑盒外 advisory only
不读取模型内部推理参数
不自动改写模型回答
```

r"""Prepare a migrated HPM-RT1 package for local service use."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = _root()
DEPLOYMENT_PROFILE = "chat_only"
PRODUCT_SURFACE = "chat_hpm_gateway"
NONINTERACTIVE = "--no-pause" in sys.argv or os.environ.get("HPM_RT1_NONINTERACTIVE") == "1"


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {label}: {path}")


def _desktop() -> Path:
    return Path.home() / "Desktop"


def _discover_local_model_path() -> str:
    env_candidates = [
        os.environ.get("HPM_RT1_PREMISE_AUDITOR_MODEL_PATH"),
        os.environ.get("HPM_RT1_LOCAL_QWEN_PATH"),
    ]
    for raw_path in env_candidates:
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if candidate.exists():
            return str(candidate)

    runtime_candidates = [
        ROOT / "models" / "Qwen2.5-1.5B-Instruct",
        ROOT / "models" / "Qwen" / "Qwen2.5-1.5B-Instruct",
        ROOT / "local_models" / "Qwen2.5-1.5B-Instruct",
    ]
    for candidate in runtime_candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _create_shortcut(shortcut_path: Path, target: Path, working_dir: Path) -> None:
    ps = f"""
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut('{shortcut_path}')
$shortcut.TargetPath = '{target}'
$shortcut.WorkingDirectory = '{working_dir}'
$shortcut.IconLocation = '{target},0'
$shortcut.Save()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _pause() -> None:
    if not NONINTERACTIVE:
        input("按 Enter 关闭窗口...")


def main() -> None:
    print("HPM-RT1 模块部署器")
    print(f"部署目录: {ROOT}")

    _require(ROOT / "hpm_rt1_beta" / "__init__.py", "核心模块 hpm_rt1_beta")
    _require(ROOT / "api" / "server.py", "FastAPI gateway")
    _require(ROOT / "frontend" / "streamlit_app.py", "Streamlit 前端")
    _require(ROOT / "requirements.txt", "依赖清单")

    for rel in [
        "outputs",
        "outputs/hpm_rt1_api_output",
        "outputs/hpm_rt1_frontend_output",
        "logs",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    runtime_config = {
        "project": "HPM-RT1",
        "version": "1.8.6",
        "deployment_profile": DEPLOYMENT_PROFILE,
        "product_surface": PRODUCT_SURFACE,
        "positioning": "black_box_llm_external_ai_trust_gateway_runtime",
        "advisory_only": True,
        "model_internal_state_access": False,
        "api_url": "http://127.0.0.1:8000",
        "frontend_url": "http://127.0.0.1:8501",
        "premise_integrity_gate": True,
        "pre_answer_warning": True,
        "evidence_verification_loop": True,
        "answer_grounding": True,
        "reference_grounded_answer_audit": True,
        "trend_gate": True,
        "gateway_log_compaction": True,
        "enabled_modules": {
            "chat_hpm": True,
            "agent_hpm": False,
            "ide_hpm": False,
        },
        "local_premise_auditor": {
            "enable_env": "HPM_RT1_PREMISE_AUDITOR=ollama or transformers",
            "default_model": "deepseek-r1:32b",
            "qwen_hf_model_path": _discover_local_model_path(),
            "portable_model_discovery": "env_first_then_runtime_relative_paths",
        },
        "gateway_pdf_report": str(
            _desktop() / "HPM_RT1_Gateway_Audit_Reports" / "hpm_rt1_gateway_audit_report.pdf"
        ),
        "output_dir": str(ROOT / "outputs" / "hpm_rt1_api_output"),
        "release_dir": str(ROOT),
    }
    (ROOT / "hpm_rt1_runtime_config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    launcher = ROOT / "HPM_RT1_Launcher.exe"
    if launcher.exists():
        _create_shortcut(_desktop() / "HPM-RT1 审计台.lnk", launcher, ROOT)
        print("已创建桌面快捷方式：HPM-RT1 审计台.lnk")
    else:
        print("未找到 HPM_RT1_Launcher.exe，跳过桌面快捷方式创建。")

    print("")
    print("模块部署完成。下一步请运行 HPM_RT1_Launcher.exe。")
    print("Gateway PDF 日志将输出到桌面 HPM_RT1_Gateway_Audit_Reports 文件夹。")
    _pause()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("")
        print(f"部署失败: {exc}")
        _pause()
        raise

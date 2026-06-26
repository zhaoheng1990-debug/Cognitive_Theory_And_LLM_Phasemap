r"""Windows launcher for HPM-RT1 v1.8.6."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


API_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:8501"
DEPLOYMENT_PROFILE = "chat_only"
PRODUCT_SURFACE = "chat_hpm_gateway"


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


RUNTIME_ROOT = _runtime_root()


def _discover_local_model_path() -> Path | None:
    env_candidates = [
        os.environ.get("HPM_RT1_PREMISE_AUDITOR_MODEL_PATH"),
        os.environ.get("HPM_RT1_LOCAL_QWEN_PATH"),
    ]
    for raw_path in env_candidates:
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if candidate.exists():
            return candidate

    runtime_candidates = [
        RUNTIME_ROOT / "models" / "Qwen2.5-1.5B-Instruct",
        RUNTIME_ROOT / "models" / "Qwen" / "Qwen2.5-1.5B-Instruct",
        RUNTIME_ROOT / "local_models" / "Qwen2.5-1.5B-Instruct",
    ]
    for candidate in runtime_candidates:
        if candidate.exists():
            return candidate
    return None


def _load_runtime_env_file() -> None:
    env_candidates = [
        Path.home() / ".codex" / "runtime.env",
        RUNTIME_ROOT / "runtime.env",
    ]
    for env_file in env_candidates:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                os.environ.setdefault(key, value.strip())


def _configure_environment() -> None:
    _load_runtime_env_file()
    output_dir = RUNTIME_ROOT / "outputs" / "hpm_rt1_api_output"
    frontend_output_dir = RUNTIME_ROOT / "outputs" / "hpm_rt1_frontend_output"
    qwen_model_path = _discover_local_model_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HPM_RT1_OUTPUT_DIR", str(output_dir))
    os.environ.setdefault("HPM_RT1_FRONTEND_OUTPUT_DIR", str(frontend_output_dir))
    os.environ.setdefault("HPM_RT1_DEPLOYMENT_PROFILE", DEPLOYMENT_PROFILE)
    os.environ.setdefault("HPM_RT1_PRODUCT_SURFACE", PRODUCT_SURFACE)
    os.environ.setdefault("HPM_RT1_RELEASE_DIR", str(RUNTIME_ROOT))
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    if qwen_model_path is not None:
        os.environ.setdefault("HPM_RT1_PREMISE_AUDITOR", "transformers")
        os.environ.setdefault("HPM_RT1_PREMISE_AUDITOR_MODEL_PATH", str(qwen_model_path))


def _python_executable() -> Path:
    candidates = [
        RUNTIME_ROOT / ".venv" / "Scripts" / "python.exe",
        RUNTIME_ROOT / "venv" / "Scripts" / "python.exe",
    ]
    if not getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("python")


def _spawn_process(args: list[str], log_name: str) -> subprocess.Popen:
    _configure_environment()
    log_dir = RUNTIME_ROOT / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / log_name).open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RUNTIME_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        args,
        cwd=str(RUNTIME_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _spawn_streamlit() -> subprocess.Popen:
    python = str(_python_executable())
    app_path = RUNTIME_ROOT / "frontend" / "streamlit_app.py"
    return _spawn_process(
        [
            python,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        "hpm_rt1_streamlit.log",
    )


def _spawn_api() -> subprocess.Popen:
    python = str(_python_executable())
    return _spawn_process(
        [
            python,
            "-m",
            "uvicorn",
            "api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        "hpm_rt1_api.log",
    )


def _child_mode() -> None:
    if "--streamlit-child" in sys.argv:
        _spawn_streamlit().wait()
        return
    if "--api-child" in sys.argv:
        _spawn_api().wait()
        return


def _print_banner() -> None:
    report_path = Path.home() / "Desktop" / "HPM_RT1_Gateway_Audit_Reports" / "hpm_rt1_gateway_audit_report.pdf"
    print("")
    print("HPM-RT1 v1.8.6 黑盒外 AI Trust Gateway 运行时已启动")
    print(f"API:      {API_URL}")
    print(f"前端:     {FRONTEND_URL}")
    print(f"Gateway PDF 日志: {report_path}")
    print("")
    print("说明：本工具位于 LLM 黑盒外部，只读取可见输入/输出，不读取模型内部推理状态。")
    print("边界：仅输出过程预警、审计与建议动作，不自动改写模型回答。")
    print("服务会在后台运行；如需停止，可在任务管理器结束 python.exe/streamlit/uvicorn 相关进程。")
    print("")


def main() -> None:
    _configure_environment()
    if "--streamlit-child" in sys.argv or "--api-child" in sys.argv:
        _child_mode()
        return

    python = _python_executable()
    if str(python) == "python":
        print("未在安装目录找到 .venv\\Scripts\\python.exe，将尝试使用系统 Python。")
        print("如启动失败，请先运行 HPM_RT1_Install_Dependencies.exe。")

    _spawn_streamlit()
    _spawn_api()
    time.sleep(2)
    _print_banner()
    try:
        webbrowser.open(FRONTEND_URL)
    except Exception:
        pass
    print("服务正在后台运行。关闭本窗口不会自动关闭已启动的后台服务。")
    print("日志文件位于 outputs\\hpm_rt1_api.log 和 outputs\\hpm_rt1_streamlit.log。")
    input("按 Enter 关闭此启动器窗口...")


if __name__ == "__main__":
    main()

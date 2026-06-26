r"""Portable one-click installer for the Chat HPM Gateway package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PYTHON_MIN = (3, 10)
PYTHON_VERSION_LABEL = "3.13.14"
VC_INSTALLER_NAME = "vc_redist.x64.exe"
PYTHON_INSTALLER_NAME = f"python-{PYTHON_VERSION_LABEL}-amd64.exe"


def _root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = _root()


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("")
    print("执行:", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, check=check, env=env)


def _prereq_dir() -> Path:
    return ROOT / "prereqs"


def _python_ok(cmd: list[str]) -> bool:
    version_expr = (
        "import sys; "
        f"raise SystemExit(0 if sys.version_info >= {PYTHON_MIN} else 1)"
    )
    try:
        completed = subprocess.run(
            cmd + ["-c", version_expr],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return completed.returncode == 0


def _python_command() -> list[str]:
    for candidate in (["py", "-3"], ["python"]):
        if _python_ok(candidate):
            return candidate
    raise RuntimeError("未检测到可用 Python 3.10+。")


def _run_vc_installer_if_present() -> None:
    installer = _prereq_dir() / VC_INSTALLER_NAME
    if not installer.exists():
        print("未找到内置 VC++ 运行库安装器，跳过这一步。")
        return
    print("检测到内置 VC++ 运行库安装器，开始静默安装/修复。")
    _run([str(installer), "/install", "/passive", "/norestart"])


def _run_python_installer_if_needed() -> None:
    if _python_ok(["py", "-3"]) or _python_ok(["python"]):
        print("系统已具备可用 Python 3.10+，跳过 Python 安装。")
        return

    installer = _prereq_dir() / PYTHON_INSTALLER_NAME
    if not installer.exists():
        raise FileNotFoundError(
            f"未找到内置 Python 安装器 {installer.name}，请确认 prereqs 目录完整。"
        )

    print(f"未检测到可用 Python，开始安装 Python {PYTHON_VERSION_LABEL}。")
    _run(
        [
            str(installer),
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_pip=1",
            "Include_launcher=1",
            "Shortcuts=0",
        ]
    )

    if not (_python_ok(["py", "-3"]) or _python_ok(["python"])):
        raise RuntimeError("Python 安装后仍未检测到可用解释器。")


def _run_dependency_installer() -> None:
    extra_env = {"HPM_RT1_NONINTERACTIVE": "1"}
    script = ROOT / "hpm_rt1_install_dependencies.py"
    if script.exists():
        _run(_python_command() + [str(script), "--no-pause"], extra_env=extra_env)
        return
    exe = ROOT / "HPM_RT1_Install_Dependencies.exe"
    if exe.exists():
        _run([str(exe), "--no-pause"], extra_env=extra_env)
        return
    raise FileNotFoundError("未找到依赖安装入口。")


def _run_deployer() -> None:
    extra_env = {"HPM_RT1_NONINTERACTIVE": "1"}
    script = ROOT / "hpm_rt1_deploy_modules.py"
    if script.exists():
        _run(_python_command() + [str(script), "--no-pause"], extra_env=extra_env)
        return
    exe = ROOT / "HPM_RT1_Deploy_Modules.exe"
    if exe.exists():
        _run([str(exe), "--no-pause"], extra_env=extra_env)
        return
    raise FileNotFoundError("未找到模块部署入口。")


def _launch_runtime() -> None:
    launcher_script = ROOT / "hpm_rt1_launcher.py"
    if launcher_script.exists():
        print("安装完成，正在启动 HPM-RT1。")
        subprocess.Popen(_python_command() + [str(launcher_script)], cwd=str(ROOT))
        return

    launcher_exe = ROOT / "HPM_RT1_Launcher.exe"
    if launcher_exe.exists():
        print("安装完成，正在启动 HPM-RT1。")
        subprocess.Popen([str(launcher_exe)], cwd=str(ROOT))
        return

    print("安装完成，但未找到启动入口，请手动运行 HPM_RT1_Start_Runtime.bat。")


def _write_summary() -> None:
    summary_dir = ROOT / "outputs" / "installer"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_dir / "one_click_install_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "HPM-RT1 One-Click Install Summary",
                f"root={ROOT}",
                f"venv_exists={(ROOT / '.venv' / 'Scripts' / 'python.exe').exists()}",
                f"wheelhouse_exists={(ROOT / 'wheelhouse').exists()}",
                f"python_installer_present={(_prereq_dir() / PYTHON_INSTALLER_NAME).exists()}",
                f"vc_installer_present={(_prereq_dir() / VC_INSTALLER_NAME).exists()}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    print("")
    print("HPM-RT1 一键安装器")
    print(f"安装目录: {ROOT}")
    print("")
    print("流程：前置环境 -> Python -> 依赖 -> 部署 -> 启动")

    _run_vc_installer_if_present()
    _run_python_installer_if_needed()
    _run_dependency_installer()
    _run_deployer()
    _write_summary()
    _launch_runtime()

    print("")
    print("一键安装完成。")
    print(r"如需查看安装记录，可打开 outputs\installer\one_click_install_summary.txt")
    input("按 Enter 关闭窗口...")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("")
        print(f"一键安装失败: {exc}")
        print("建议检查 prereqs、wheelhouse 和安装目录完整性。")
        input("按 Enter 关闭窗口...")
        raise

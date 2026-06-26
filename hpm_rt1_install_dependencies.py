r"""Install HPM-RT1 runtime dependencies into a local Windows virtualenv."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = _root()
VENV = ROOT / ".venv"
PYTHON = VENV / "Scripts" / "python.exe"
NONINTERACTIVE = "--no-pause" in sys.argv or os.environ.get("HPM_RT1_NONINTERACTIVE") == "1"


def _venv_creator() -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    for candidate in (["py", "-3"], ["python"]):
        try:
            subprocess.run(
                candidate + ["--version"],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "未找到可用于创建 .venv 的 Python。请先安装 Python 3.10+ 并加入 PATH，或安装 Python Launcher。"
    )


def _run(cmd: list[str]) -> None:
    print("")
    print("执行:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def _requirements_file() -> Path:
    runtime_requirements = ROOT / "requirements-runtime.txt"
    if runtime_requirements.exists():
        return runtime_requirements
    return ROOT / "requirements.txt"


def _install_runtime_dependencies(requirements: Path) -> None:
    wheelhouse = ROOT / "wheelhouse"
    if wheelhouse.exists() and any(wheelhouse.glob("*.whl")):
        _run(
            [
                str(PYTHON),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                "-r",
                str(requirements),
            ]
        )
        return
    _run([str(PYTHON), "-m", "pip", "install", "-r", str(requirements)])


def _install_project_if_needed() -> None:
    if getattr(sys, "frozen", False):
        print("冻结交付包模式：跳过项目 editable 安装，运行时将直接使用安装目录源码。")
        return
    _run([str(PYTHON), "-m", "pip", "install", "-e", str(ROOT)])


def _pause() -> None:
    if not NONINTERACTIVE:
        input("按 Enter 关闭窗口...")


def main() -> None:
    print("HPM-RT1 依赖安装器")
    print(f"安装目录: {ROOT}")

    if not PYTHON.exists():
        _run(_venv_creator() + ["-m", "venv", str(VENV)])
    else:
        print(".venv 已存在，继续检查依赖。")

    _run([str(PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    requirements = _requirements_file()
    print(f"依赖清单: {requirements.name}")
    _install_runtime_dependencies(requirements)
    _install_project_if_needed()

    print("")
    print("依赖安装完成。下一步请运行 HPM_RT1_Deploy_Modules.exe。")
    _pause()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("")
        print(f"安装失败，退出码: {exc.returncode}")
        print("请确认 Python 3.10+ 可用，或查看上方 pip 输出。")
        _pause()
        raise

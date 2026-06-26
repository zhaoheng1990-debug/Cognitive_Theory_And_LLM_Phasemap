@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "ROOT=%~dp0"
set "PY="

if exist "%ROOT%.venv\Scripts\python.exe" set "PY=%ROOT%.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%venv\Scripts\python.exe" set "PY=%ROOT%venv\Scripts\python.exe"

if not defined PY (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)"') do set "PY=%%I"
    )
)

if not defined PY (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PY=%%I"
    )
)

if not defined PY (
    echo 未检测到可用 Python 3.10+。
    echo 请先运行 HPM_RT1_OneClick_Setup.bat
    pause
    exit /b 1
)

"%PY%" "%ROOT%hpm_rt1_launcher.py"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo 启动失败，退出码: %EXITCODE%
    echo 可先运行 HPM_RT1_Run_Installation_Doctor.bat 排查环境。
    pause
)
endlocal
exit /b %EXITCODE%

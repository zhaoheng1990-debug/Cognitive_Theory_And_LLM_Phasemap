@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "ROOT=%~dp0"
set "PY="

title HPM-RT1 一键安装
echo.
echo ============================================
echo HPM-RT1 一键安装
echo 安装目录: %ROOT%
echo ============================================
echo.

call :detect_python

if exist "%ROOT%prereqs\vc_redist.x64.exe" (
    echo [1/5] 安装或修复 VC++ 运行库...
    start /wait "" "%ROOT%prereqs\vc_redist.x64.exe" /install /passive /norestart
)

if not defined PY (
    if exist "%ROOT%prereqs\python-3.13.14-amd64.exe" (
        echo [2/5] 安装 Python 3.13.14...
        start /wait "" "%ROOT%prereqs\python-3.13.14-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Shortcuts=0
    ) else (
        echo 未找到内置 Python 安装器: %ROOT%prereqs\python-3.13.14-amd64.exe
        goto :fail
    )
)

call :detect_python
if not defined PY (
    echo 安装后仍未检测到可用 Python 3.10+。
    goto :fail
)

echo [3/5] 安装运行依赖...
"%PY%" "%ROOT%hpm_rt1_install_dependencies.py" --no-pause
if errorlevel 1 goto :fail

echo [4/5] 生成本地部署配置...
"%PY%" "%ROOT%hpm_rt1_deploy_modules.py" --no-pause
if errorlevel 1 goto :fail

echo [5/5] 启动 HPM-RT1...
start "" "%PY%" "%ROOT%hpm_rt1_launcher.py"

echo.
echo 安装完成。
echo 默认启动入口:
echo   HPM_RT1_Start_Runtime.bat
echo.
echo 如果你需要排查安装状态，可以运行:
echo   HPM_RT1_Run_Installation_Doctor.bat
goto :end

:detect_python
set "PY="
if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PY=%ROOT%.venv\Scripts\python.exe"
    goto :eof
)
if exist "%ROOT%venv\Scripts\python.exe" (
    set "PY=%ROOT%venv\Scripts\python.exe"
    goto :eof
)
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)"') do set "PY=%%I"
    goto :eof
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PY=%%I"
    goto :eof
)
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
    if exist "%%~fD\python.exe" (
        "%%~fD\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PY=%%~fD\python.exe"
            goto :eof
        )
    )
)
goto :eof

:fail
echo.
echo 一键安装未完成。
echo 建议先运行 HPM_RT1_Install_Prerequisites_If_Venv_Broken.bat
echo 然后再重新执行 HPM_RT1_OneClick_Setup.bat

:end
echo.
pause
endlocal

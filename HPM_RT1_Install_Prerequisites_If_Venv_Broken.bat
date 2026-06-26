@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

title HPM-RT1 前置环境补装器

echo.
echo ============================================
echo HPM-RT1 前置环境补装器
echo 适用场景：.venv 不可复用、启动器报缺 Python / DLL / VC++ 运行库
echo ============================================
echo.

set "WORKDIR=%~dp0"
set "TMPDIR=%TEMP%\hpm_rt1_prereqs"
set "VC_URL=https://aka.ms/vc14/vc_redist.x64.exe"
set "VC_FILE=%TMPDIR%\vc_redist.x64.exe"
set "PY_URL=https://www.python.org/ftp/python/3.13.14/python-3.13.14-amd64.exe"
set "PY_FILE=%TMPDIR%\python-3.13.14-amd64.exe"
set "PY_OK=0"

if not exist "%TMPDIR%" mkdir "%TMPDIR%"

echo [1/4] 检查当前 Python...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if !errorlevel! == 0 set "PY_OK=1"
)

if "%PY_OK%"=="0" (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if !errorlevel! == 0 set "PY_OK=1"
    )
)

if "%PY_OK%"=="1" (
    echo 已检测到可用 Python（>= 3.10），将跳过 Python 安装。
) else (
    echo 未检测到可用 Python（>= 3.10），稍后将安装 Python 3.13.14 x64。
)

echo.
echo [2/4] 下载并安装 Microsoft Visual C++ Redistributable x64...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri '%VC_URL%' -OutFile '%VC_FILE%'"
if errorlevel 1 (
    echo VC++ 运行库下载失败。
    echo 你也可以手动打开官方页面安装：
    echo https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
    goto :fail
)

start /wait "" "%VC_FILE%" /install /passive /norestart
if errorlevel 1 (
    echo VC++ 运行库安装未成功完成，请手动检查。
    goto :fail
)
echo VC++ 运行库安装完成。

if "%PY_OK%"=="0" (
    echo.
    echo [3/4] 下载并安装 Python 3.13.14 x64...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_FILE%'"
    if errorlevel 1 (
        echo Python 下载失败。
        echo 你也可以手动打开官方页面安装：
        echo https://www.python.org/downloads/windows/
        goto :fail
    )

    start /wait "" "%PY_FILE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Shortcuts=0
    if errorlevel 1 (
        echo Python 安装未成功完成，请手动检查。
        goto :fail
    )
    echo Python 安装完成。
) else (
    echo.
    echo [3/4] 跳过 Python 安装。
)

echo.
echo [4/4] 后续动作建议
echo 1. 关闭当前窗口后，回到 HPM-RT1 根目录
echo 2. 优先运行 HPM_RT1_OneClick_Setup.bat
echo 3. 如需日常启动，运行 HPM_RT1_Start_Runtime.bat
echo 4. 如需排查，再运行 HPM_RT1_Run_Installation_Doctor.bat
echo.
echo 如果你使用浏览器插件，再确认 Chrome 或 Edge 已安装。
echo.
echo 完成。
goto :end

:fail
echo.
echo 预置环境补装未完整完成。
echo 建议手动安装以下两项后再回来：
echo - Microsoft Visual C++ Redistributable x64
echo - Python 3.11 x64
echo.

:end
echo.
pause
endlocal

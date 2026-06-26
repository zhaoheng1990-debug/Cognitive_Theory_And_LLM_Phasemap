@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "PY="

if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"
if not defined PY set "PY=python"

echo.
echo ============================================
echo HPM-RT1 安装医生
echo 1. 安装前预检
echo 2. 安装后验收
echo ============================================
echo.
set /p MODE=请输入模式编号 ^(1 或 2^): 

if "%MODE%"=="1" goto preflight
if "%MODE%"=="2" goto postflight

echo 无效输入，默认执行安装前预检。
goto preflight

:preflight
"%PY%" "%ROOT%tools\hpm_rt1_install_doctor.py" --mode preflight --output-json "%ROOT%outputs\install_doctor_preflight.json"
goto end

:postflight
"%PY%" "%ROOT%tools\hpm_rt1_install_doctor.py" --mode postflight --output-json "%ROOT%outputs\install_doctor_postflight.json"
goto end

:end
echo.
pause
endlocal

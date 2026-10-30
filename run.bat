@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ============================================================
rem  基于 OpenCV 的停车位识别系统 - Windows 启动脚本
rem  用法:
rem     run.bat                      -> 打开桌面界面
rem     run.bat evaluate --curve     -> 执行精度评估
rem     run.bat demo                 -> 重新生成演示数据
rem  可用环境变量 PARKING_PYTHON 指定解释器路径
rem ============================================================

set "PY="

if defined PARKING_PYTHON (
    set "PY=%PARKING_PYTHON%"
    goto :run
)

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    goto :check
)

where python >nul 2>&1
if %errorlevel%==0 (
    set "PY=python"
    goto :check
)

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
    goto :check
)

echo.
echo [错误] 没有找到 Python，请先安装 Python 3.8+ 并勾选 Add to PATH
echo        下载地址: https://www.python.org/downloads/
echo.
pause
exit /b 1

:check
%PY% -c "import cv2, numpy" >nul 2>&1
if not %errorlevel%==0 (
    echo.
    echo [提示] 当前 Python 缺少依赖，正在尝试安装 ...
    %PY% -m pip install -r requirements.txt
    if not %errorlevel%==0 (
        echo.
        echo [错误] 依赖安装失败，请手动执行:  pip install -r requirements.txt
        pause
        exit /b 1
    )
)

:run
if "%~1"=="" (
    set "ARGS=gui"
) else (
    set "ARGS=%*"
)

echo 使用解释器: %PY%
%PY% main.py %ARGS%
if not %errorlevel%==0 (
    echo.
    echo [提示] 命令执行结束（返回码 %errorlevel%），按任意键关闭窗口
    pause >nul
)
endlocal

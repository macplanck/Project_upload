@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==================================================
REM Configuration (adjust the directory if needed)
REM ==================================================
set "MSYS_SHELL=C:\msys64\msys2_shell.cmd"
set "WORKDIR=/d/NYCU/project/EE_project/NSTC_project/designs/python_simulation/Project_upload-mac_branch/B0_CMD"

REM ==================================================
REM Help
REM ==================================================
if "%~1"=="" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--help" goto :help

REM ==================================================
REM Argument dispatch
REM ==================================================
if /i "%~1"=="init" goto :mem_init
if /i "%~1"=="combine" goto :mem_combine

echo [ERROR] Unknown command: %1
echo.
goto :help

:mem_init
call "%MSYS_SHELL%" -ucrt64 -defterm -no-start -lc "cd %WORKDIR% && ./41_mem_init"
goto :eof

:mem_combine
call "%MSYS_SHELL%" -ucrt64 -defterm -no-start -lc "cd %WORKDIR% && ./40_mem_combine"
goto :eof

:help
echo Usage:
echo   mem ^<command^>
echo.
echo Commands:
echo   init        Run 41_mem_init
echo   combine     Run 40_mem_combine
echo.
echo Options:
echo   -h, --help  Show this help
exit /b 1

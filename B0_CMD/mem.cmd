@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==================================================
REM Configuration
REM ==================================================
set MSYS_BASH=C:\msys64\ucrt64.exe
set WORKDIR=/d/NYCU/project/EE_project/NSTC_project/designs/python_simulation/Project_upload-mac_branch/B0_CMD

REM ==================================================
REM Help
REM ==================================================
if "%~1"=="" goto :help
if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help

REM ==================================================
REM Argument dispatch
REM ==================================================
if "%~1"=="init" goto :mem_init
if "%~1"=="combine" goto :mem_combine

echo [ERROR] Unknown command: %1
echo.
goto :help

REM ==================================================
REM Commands
REM ==================================================
:mem_init
"%MSYS_BASH%" -lc "cd %WORKDIR% && ./41_mem_init"
goto :eof

:mem_combine
"%MSYS_BASH%" -lc "cd %WORKDIR% && ./40_mem_combine"
goto :eof

REM ==================================================
REM Help text
REM ==================================================
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

REM command: .\mem {init, combine, or --help}
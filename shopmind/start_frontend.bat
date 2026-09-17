@echo off
chcp 65001 >nul
title RAG Frontend (port 5173)
rem Prefer user-installed Node; fall back to WorkBuddy managed Node.
where npm >nul 2>nul
if errorlevel 1 set "PATH=%USERPROFILE%\.workbuddy\binaries\node\versions\22.12.0;%PATH%"
cd /d "%~dp0frontend"
if not exist "node_modules" (
    echo [INFO] Installing frontend dependencies...
    call npm install
)
call npm run dev
pause

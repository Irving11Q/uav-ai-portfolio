@echo off
chcp 65001 >nul
title RAG Backend (port 8000)
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run.py
pause

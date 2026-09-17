@echo off
start "RAG-Backend" cmd /k ""%~dp0start_backend.bat""
timeout /t 2 >nul
start "RAG-Frontend" cmd /k ""%~dp0start_frontend.bat""
echo Backend and Frontend windows launched.
echo Browser: http://127.0.0.1:5173   (login as admin)
timeout /t 5 >nul

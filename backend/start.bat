@echo off
echo ==============================================
echo   LearnPath AI - Backend Startup
echo ==============================================
echo.
echo [1/2] Starting FastAPI backend on port 8000...
cd /d "%~dp0"
call venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

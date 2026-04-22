@echo off
setlocal

cd /d "%~dp0"

start "" powershell -NoProfile -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:8000'"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000
) else (
  python -m uvicorn main:app --host 0.0.0.0 --port 8000
)

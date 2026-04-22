@echo off
setlocal

cd /d "%~dp0"

start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 10; Start-Process 'http://localhost:8000'"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000
) else (
  python -m uvicorn main:app --host 0.0.0.0 --port 8000
)

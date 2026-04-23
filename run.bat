@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "CHECK_IMPORTS=import fastapi, httpx, numpy, requests, soundfile, torch, torchaudio, uvicorn; import dotenv; import nemo.collections.asr"
set "CHECK_CUDA=import os, sys, torch; requested = os.environ.get('CANARY_DEVICE', 'cuda').strip().lower(); sys.exit(0 if (requested != 'cuda' or torch.cuda.is_available()) else 1)"

if not exist "%VENV_PYTHON%" goto install_first
call "%VENV_PYTHON%" -c "import sys" >nul 2>nul
if errorlevel 1 goto install_first
call "%VENV_PYTHON%" -c "%CHECK_IMPORTS%" >nul 2>nul
if errorlevel 1 goto install_first
call "%VENV_PYTHON%" -c "%CHECK_CUDA%" >nul 2>nul
if errorlevel 1 goto install_first
goto start_app

:install_first
echo Dependencies are not installed yet. Running install.bat...
call "%~dp0install.bat"
if errorlevel 1 exit /b %errorlevel%
call "%VENV_PYTHON%" -c "%CHECK_CUDA%" >nul 2>nul
if errorlevel 1 (
  echo CUDA mode requested but torch.cuda.is_available() is still false.
  exit /b 1
)

:start_app

start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 10; Start-Process 'http://localhost:8888'"

"%VENV_PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8888

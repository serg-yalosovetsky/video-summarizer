@echo off
setlocal

cd /d "%~dp0"

set /p PYTHON_VERSION=<"%~dp0.python-version"
set "VENV_DIR=%~dp0.venv"

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "CHECK_IMPORTS=import fastapi, httpx, numpy, requests, soundfile, torch, torchaudio, uvicorn; import dotenv; import nemo.collections.asr"
set "CHECK_CUDA=import os, sys, torch; requested = os.environ.get('CANARY_DEVICE', 'cuda').strip().lower(); sys.exit(0 if (requested != 'cuda' or torch.cuda.is_available()) else 1)"
set "SHOW_TORCH_DIAG=import shutil, torch; print(f'torch={torch.__version__} torch.version.cuda={torch.version.cuda} torch.cuda.is_available()={torch.cuda.is_available()} nvidia-smi={shutil.which(\"nvidia-smi\") is not None}')"
if "%TORCH_CUDA_INDEX_URL%"=="" set "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128"

where uv >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe"
  ) else (
  echo uv is not installed. Installing uv...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 exit /b 1

    if exist "%USERPROFILE%\.local\bin\uv.exe" (
      set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
      where uv >nul 2>nul
      if errorlevel 1 (
        echo uv installation completed, but uv is still not available in PATH.
        exit /b 1
      )
      set "UV_BIN=uv"
    )
  )
) else (
  set "UV_BIN=uv"
)

echo Using uv: %UV_BIN%
echo Installing Python %PYTHON_VERSION% via uv...
call "%UV_BIN%" python install %PYTHON_VERSION%
if errorlevel 1 exit /b 1

if exist "%VENV_PYTHON%" (
  call "%VENV_PYTHON%" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Existing virtual environment is broken. Recreating...
    rmdir /s /q "%VENV_DIR%"
  )
)

if not exist "%VENV_DIR%" (
  echo Creating virtual environment with Python %PYTHON_VERSION%...
  call "%UV_BIN%" venv --python %PYTHON_VERSION% "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

if not exist "%VENV_PYTHON%" (
  echo Virtual environment was created, but "%VENV_PYTHON%" is missing.
  exit /b 1
)

call "%VENV_PYTHON%" -c "%CHECK_IMPORTS%" >nul 2>nul
if errorlevel 1 (
  echo Installing Python dependencies...
  set "UV_LINK_MODE=copy"
  call "%UV_BIN%" pip install --python "%VENV_PYTHON%" -r "%~dp0requirements.txt"
  if errorlevel 1 exit /b 1
) else (
  echo Python dependencies already installed.
)

call "%VENV_PYTHON%" -c "%CHECK_CUDA%" >nul 2>nul
if errorlevel 1 (
  echo Repairing PyTorch CUDA build...
  set "UV_LINK_MODE=copy"
  call "%UV_BIN%" pip install --python "%VENV_PYTHON%" --index-url "%TORCH_CUDA_INDEX_URL%" --upgrade torch torchaudio
  if errorlevel 1 exit /b 1
  call "%VENV_PYTHON%" -c "%CHECK_CUDA%" >nul 2>nul
  if errorlevel 1 (
    echo CUDA mode is still unavailable after PyTorch reinstall.
    call "%VENV_PYTHON%" -c "%SHOW_TORCH_DIAG%"
    echo Ensure NVIDIA drivers are installed and the CUDA wheel index matches your driver stack.
    exit /b 1
  )
)

echo Installation complete.
exit /b 0

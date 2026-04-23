#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_VERSION=$(tr -d '\r\n' < "$SCRIPT_DIR/.python-version")
VENV_DIR="$SCRIPT_DIR/.venv"

CHECK_IMPORTS='import fastapi, httpx, numpy, requests, soundfile, torch, torchaudio, uvicorn; import dotenv; import nemo.collections.asr; import pyannote.audio; import desktop_notifier'
CHECK_CUDA='import os, sys, torch; requested = os.environ.get("CANARY_DEVICE", "cuda").strip().lower(); sys.exit(0 if (requested != "cuda" or torch.cuda.is_available()) else 1)'
SHOW_TORCH_DIAG='import shutil, torch; print(f"torch={torch.__version__} torch.version.cuda={torch.version.cuda} torch.cuda.is_available()={torch.cuda.is_available()} nvidia-smi={shutil.which(\"nvidia-smi\") is not None}")'
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if command -v uv >/dev/null 2>&1; then
  UV_BIN=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv is not installed. Installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available to install uv." >&2
    exit 1
  fi

  if command -v uv >/dev/null 2>&1; then
    UV_BIN=uv
  elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "uv installation completed, but uv is still not available in PATH." >&2
    exit 1
  fi
fi

echo "Using uv: $UV_BIN"
echo "Installing Python $PYTHON_VERSION via uv..."
"$UV_BIN" python install "$PYTHON_VERSION"

VENV_PYTHON="$VENV_DIR/bin/python"

if [ -x "$VENV_PYTHON" ] && ! "$VENV_PYTHON" -c "import sys" >/dev/null 2>&1; then
  echo "Existing virtual environment is broken. Recreating..."
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment with Python $PYTHON_VERSION..."
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Virtual environment was created, but $VENV_PYTHON is missing." >&2
  exit 1
fi

if "$VENV_PYTHON" -c "$CHECK_IMPORTS" >/dev/null 2>&1; then
  echo "Python dependencies already installed."
else
  echo "Installing Python dependencies..."
  UV_LINK_MODE=copy "$UV_BIN" pip install --python "$VENV_PYTHON" -r "$SCRIPT_DIR/requirements.txt"
fi

if ! "$VENV_PYTHON" -c "$CHECK_CUDA" >/dev/null 2>&1; then
  echo "Repairing PyTorch CUDA build..."
  UV_LINK_MODE=copy "$UV_BIN" pip install --python "$VENV_PYTHON" --index-url "$TORCH_CUDA_INDEX_URL" --upgrade torch torchaudio
  if ! "$VENV_PYTHON" -c "$CHECK_CUDA" >/dev/null 2>&1; then
    echo "CUDA mode is still unavailable after PyTorch reinstall." >&2
    "$VENV_PYTHON" -c "$SHOW_TORCH_DIAG"
    echo "Ensure NVIDIA drivers are installed and the CUDA wheel index matches your driver stack." >&2
    exit 1
  fi
fi

echo "Installation complete."

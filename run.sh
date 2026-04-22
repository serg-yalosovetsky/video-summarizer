#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_VERSION=$(tr -d '\r\n' < "$SCRIPT_DIR/.python-version")
VENV_DIR="$SCRIPT_DIR/.venv"

CHECK_IMPORTS='import fastapi, httpx, numpy, requests, soundfile, torch, torchaudio, uvicorn; import dotenv; import nemo.collections.asr'
CHECK_CUDA='import os, sys, torch; requested = os.environ.get("CANARY_DEVICE", "cuda").strip().lower(); sys.exit(0 if (requested != "cuda" or torch.cuda.is_available()) else 1)'

if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1 || ! "$VENV_DIR/bin/python" -c "$CHECK_IMPORTS" >/dev/null 2>&1 || ! "$VENV_DIR/bin/python" -c "$CHECK_CUDA" >/dev/null 2>&1; then
  echo "Dependencies are not installed yet. Running install.sh..."
  sh "$SCRIPT_DIR/install.sh"
fi

if ! "$VENV_DIR/bin/python" -c "$CHECK_CUDA" >/dev/null 2>&1; then
  echo "CUDA mode requested but torch.cuda.is_available() is still false." >&2
  exit 1
fi

(
  sleep 6
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8000" >/dev/null 2>&1
  elif command -v open >/dev/null 2>&1; then
    open "http://localhost:8000" >/dev/null 2>&1
  fi
) &

exec "$VENV_DIR/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8000

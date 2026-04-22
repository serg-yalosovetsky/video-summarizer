#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

(
  sleep 6
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8000" >/dev/null 2>&1
  elif command -v open >/dev/null 2>&1; then
    open "http://localhost:8000" >/dev/null 2>&1
  fi
) &

if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  exec "$SCRIPT_DIR/venv/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8000
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 8000

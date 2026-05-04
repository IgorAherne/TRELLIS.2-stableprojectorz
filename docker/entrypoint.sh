#!/bin/bash
set -euo pipefail

MODE=${MODE:-ui}

echo "[entrypoint] Starting in MODE=$MODE"

case "$MODE" in
  ui)
    exec python /app/app.py --host 0.0.0.0 --port 8080
    ;;
  api)
    exec python /app/api_spz/main_api.py --host 0.0.0.0 --port 7960 --device cuda
    ;;
  both)
    python /app/api_spz/main_api.py --host 0.0.0.0 --port 7960 --device cuda &
    exec python /app/app.py --host 0.0.0.0 --port 8080
    ;;
  *)
    echo "[entrypoint] Unknown MODE='$MODE'. Valid values: ui | api | both"
    exit 1
    ;;
esac

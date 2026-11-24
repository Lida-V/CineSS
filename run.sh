#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  if [ -x ".venv/bin/python3" ]; then
    VENV_PY=".venv/bin/python3"
  else
    echo "[エラー] .venv がありません。先に ./setup.sh を実行してください。" >&2
    exit 1
  fi
fi

PORT="${CINESS_PORT:-8780}"
URL="http://127.0.0.1:${PORT}"

# Apple Silicon MPS 未対応演算の CPU フォールバックと UTF-8 入出力を有効化
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# 既に起動済みならブラウザを開くだけ
if "$VENV_PY" -c "import socket, sys; s = socket.socket(); s.settimeout(0.5); rc = s.connect_ex(('127.0.0.1', int(sys.argv[1]))); s.close(); sys.exit(0 if rc == 0 else 1)" "$PORT" 2>/dev/null; then
  echo "既に CineSS が起動しています ($URL)。ブラウザを開きます..."
  if command -v open >/dev/null 2>&1; then
    open "$URL" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  fi
  exit 0
fi

echo "============================================"
echo "  CineSS を起動しています..."
echo "  URL: $URL"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "  準備ができたらブラウザで開きます"
echo "  終了するには Ctrl+C を押してください"
echo "============================================"

# サーバーが応答したらブラウザを開く監視をバックグラウンドで開始
(
  for _ in $(seq 1 120); do
    if "$VENV_PY" -c "import socket, sys; s = socket.socket(); s.settimeout(0.3); rc = s.connect_ex(('127.0.0.1', int(sys.argv[1]))); s.close(); sys.exit(0 if rc == 0 else 1)" "$PORT" 2>/dev/null; then
      if command -v open >/dev/null 2>&1; then
        open "$URL" || true
      elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" >/dev/null 2>&1 || true
      fi
      break
    fi
    sleep 0.5
  done
) &

exec "$VENV_PY" -X utf8 -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"

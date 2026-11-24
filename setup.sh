#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  CineSS セットアップ (macOS / Linux)"
echo "  (映像サラウンドMixMaster & バイノーラル出力)"
echo "  ※ uv と ffmpeg が必要です (README参照)"
echo "============================================"

if ! command -v uv >/dev/null 2>&1; then
  echo "[エラー] uv が見つかりません。https://docs.astral.sh/uv/ からインストールしてください。" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[エラー] ffmpeg が見つかりません。Homebrew (brew install ffmpeg) や apt 等で導入し PATH に追加してください。" >&2
  exit 1
fi

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

echo "[1/3] Python 3.11 仮想環境 (.venv) を作成しています..."
uv venv --python 3.11 .venv

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ] && [ -x ".venv/bin/python3" ]; then
  VENV_PY=".venv/bin/python3"
fi

echo "[2/3] PyTorch をインストールしています..."
if [ "$OS_NAME" = "Darwin" ]; then
  if [ "$ARCH_NAME" = "arm64" ]; then
    echo "  Apple Silicon ($ARCH_NAME) を検出しました。MPS 対応の標準 PyTorch を導入します..."
  else
    echo "  Intel Mac ($ARCH_NAME) を検出しました。標準 PyTorch (CPU) を導入します..."
  fi
  if ! uv pip install --python "$VENV_PY" "torch==2.6.0" "torchaudio==2.6.0"; then
    uv pip install --python "$VENV_PY" torch torchaudio
  fi
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "  Linux + NVIDIA GPU を検出しました。CUDA 12.4 版 PyTorch (約2.5GB) を導入します..."
  if ! uv pip install --python "$VENV_PY" "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124; then
    uv pip install --python "$VENV_PY" torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  fi
else
  echo "  Linux (NVIDIA GPU 未検出) のため CPU 版 PyTorch を導入します..."
  if ! uv pip install --python "$VENV_PY" "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cpu; then
    uv pip install --python "$VENV_PY" torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "[3/3] 依存パッケージをインストールしています（audio-separator は導入しません）..."
uv pip install --python "$VENV_PY" -r requirements.txt

echo ""
echo "============================================"
echo "  セットアップ完了! ./run.sh で起動してください"
echo "  ※ AIモデルの自動ダウンロードは行いません。"
echo "    起動後「AIモデル管理」から必要なモデルを取得してください。"
echo "    (隣接フォルダに SoundSeparator がある場合はモデルを自動共有します)"
echo "============================================"

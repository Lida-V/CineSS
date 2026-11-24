@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================
echo   CineSS セットアップ
echo   (映像サラウンドMixMaster ^& バイノーラル出力)
echo   ※ uv と ffmpeg が必要です (README参照)
echo ============================================

where uv > nul 2>&1
if errorlevel 1 (
    echo [エラー] uv が見つかりません。https://docs.astral.sh/uv/ からインストールしてください。
    pause & exit /b 1
)
where ffmpeg > nul 2>&1
if errorlevel 1 (
    echo [エラー] ffmpeg が見つかりません。FFmpeg を導入し PATH に追加してください。
    pause & exit /b 1
)

echo [1/3] Python 3.11 仮想環境 (.venv) を作成しています...
uv venv --python 3.11 .venv || (pause & exit /b 1)

echo [2/3] PyTorch をインストールしています...
where nvidia-smi > nul 2>&1
if errorlevel 1 (
    echo   NVIDIA GPU が検出されなかったため CPU 版 PyTorch を導入します...
    uv pip install --python .venv "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cpu || (
        uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cpu || (pause & exit /b 1)
    )
) else (
    echo   NVIDIA GPU を検出しました。CUDA 12.4 版 PyTorch (約2.5GB) を導入します...
    uv pip install --python .venv "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124 || (
        uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || (pause & exit /b 1)
    )
)

echo [3/3] 依存パッケージをインストールしています...
uv pip install --python .venv -r requirements.txt || (pause & exit /b 1)

echo.
echo ============================================
echo   セットアップ完了! run.bat で起動してください
echo   ※ AIモデルの自動ダウンロードは行いません。
echo     起動後「AIモデル管理」から必要なモデルを取得してください。
echo     (隣接フォルダに SoundSeparator がある場合はモデルを自動共有します)
echo ============================================
pause

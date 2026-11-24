# coding: utf-8
"""CineSS 共通設定・パス定義。

アプリのルート位置はこのファイルからの相対位置で求め、絶対パスを固定しない。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_ROOT / "output"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
JOBS_DIR = OUTPUT_DIR / "jobs"
MODELS_DIR = APP_ROOT / "models"
ENGINE_DIR = APP_ROOT / "engine"
AGENT_STATE_PATH = OUTPUT_DIR / "_agent_state.json"
SHUTDOWN_FLAG_PATH = OUTPUT_DIR / "_shutdown.flag"

SAMPLE_RATE = 48000


def _parse_port(raw: str | None, default: int = 8780) -> int:
    if not raw:
        return default
    try:
        port = int(raw.strip())
        if 1 <= port <= 65535:
            return port
    except ValueError:
        pass
    return default


DEFAULT_PORT = _parse_port(os.environ.get("CINESS_PORT"), 8780)


def ensure_directories() -> None:
    """実行時に必要な出力・モデルディレクトリを作成する。"""
    for directory in (OUTPUT_DIR, UPLOADS_DIR, JOBS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def find_venv_python(root: Path | str | None = None) -> Path:
    """指定ディレクトリ配下の .venv Python 実行ファイルを検出する（Windows / macOS / Linux 対応）。

    実ファイルが存在しない場合は OS 標準の想定パスを返す（is_file() == False）。
    """
    base = Path(root).expanduser() if root is not None else APP_ROOT
    venv_dir = base / ".venv"
    if os.name == "nt":
        candidates = [
            venv_dir / "Scripts" / "python.exe",
            venv_dir / "bin" / "python",
            venv_dir / "bin" / "python3",
        ]
    else:
        candidates = [
            venv_dir / "bin" / "python",
            venv_dir / "bin" / "python3",
            venv_dir / "Scripts" / "python.exe",
        ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return candidates[0]


def get_torch_device() -> tuple[str, str]:
    """利用可能な PyTorch 演算デバイスと表示ラベルを返す (cuda -> mps -> cpu)。"""
    try:
        import torch  # noqa: PLC0415

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda", "NVIDIA CUDA"
        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        if mps_backend is not None and mps_backend.is_available():
            return "mps", "Apple Silicon MPS"
    except Exception:
        pass
    return "cpu", "CPU"


def find_soundseparator_dir() -> Path | None:
    """隣接または環境変数指定の SoundSeparator フォルダを探索する。

    優先順:
    1. 環境変数 `SOUNDSEPARATOR_DIR`
    2. 隣接 `../SoundtSeparator`
    3. 隣接 `../SoundSeparator`
    4. ホーム配下 `~/AI/SoundSeparator`, `~/SoundSeparator`
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("SOUNDSEPARATOR_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(APP_ROOT.parent / "SoundtSeparator")
    candidates.append(APP_ROOT.parent / "SoundSeparator")
    try:
        home = Path.home()
        candidates.append(home / "AI" / "SoundSeparator")
        candidates.append(home / "SoundSeparator")
    except RuntimeError:
        pass

    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _looks_like_foley_dir(path: Path) -> bool:
    return (
        (path / "app" / "infer.py").is_file()
        or (path / "infer.py").is_file()
        or (path / "models").is_dir()
    )


def find_hunyuan_foley_dir() -> Path | None:
    """ローカルに導入済みの HunyuanVideo-Foley ディレクトリを探索する。

    優先順:
    1. 環境変数 `HUNYUAN_FOLEY_DIR`
    2. `engine/hunyuan_foley` / `engine/HunyuanVideo-Foley`
    3. 隣接 `../HunyuanVideo-Foley` / `../hunyuan_foley`
    4. ホーム配下 `~/AI/HunyuanVideo-Foley` / `~/HunyuanVideo-Foley`
    5. Windows 標準配置 `C:/AI/HunyuanVideo-Foley`
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("HUNYUAN_FOLEY_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(ENGINE_DIR / "hunyuan_foley")
    candidates.append(ENGINE_DIR / "HunyuanVideo-Foley")
    candidates.append(APP_ROOT.parent / "HunyuanVideo-Foley")
    candidates.append(APP_ROOT.parent / "hunyuan_foley")
    try:
        home = Path.home()
        candidates.append(home / "AI" / "HunyuanVideo-Foley")
        candidates.append(home / "HunyuanVideo-Foley")
    except RuntimeError:
        pass
    if os.name == "nt":
        candidates.append(Path("C:/AI/HunyuanVideo-Foley"))

    for candidate in candidates:
        try:
            if candidate.is_dir() and _looks_like_foley_dir(candidate):
                return candidate.resolve()
        except OSError:
            continue
    return None

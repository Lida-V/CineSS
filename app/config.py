# coding: utf-8
"""CineSS 共通設定・パス定義。

アプリのルート位置はこのファイルからの相対位置で求め、絶対パスを固定しない。
"""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_ROOT / "output"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
JOBS_DIR = OUTPUT_DIR / "jobs"
MODELS_DIR = APP_ROOT / "models"
ENGINE_DIR = APP_ROOT / "engine"
AGENT_STATE_PATH = OUTPUT_DIR / "_agent_state.json"

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


def find_soundseparator_dir() -> Path | None:
    """隣接または環境変数指定の SoundSeparator フォルダを探索する。

    優先順:
    1. 環境変数 `SOUNDSEPARATOR_DIR`
    2. 隣接 `../SoundtSeparator`
    3. 隣接 `../SoundSeparator`
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("SOUNDSEPARATOR_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(APP_ROOT.parent / "SoundtSeparator")
    candidates.append(APP_ROOT.parent / "SoundSeparator")

    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return None

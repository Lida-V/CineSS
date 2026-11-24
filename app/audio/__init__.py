# coding: utf-8
"""CineSS 音声処理パッケージ。

- ``separator_bridge``: FFmpeg 音声抽出・SoundSeparator DME分離/Dereverb連携・ローカルDSPフォールバック
- ``speaker_splitter``: Nemotron-3-Diarization / 音響クラスタリング・重なり分離・話者↔映像人物自動紐付け
- ``foley_engine``: 無音動画向け HunyuanVideo-Foley 連携・モーション同期プレビュー合成
"""
from __future__ import annotations

from .foley_engine import build_foley_prompt, find_hunyuan_foley, generate_foley
from .separator_bridge import detect_soundseparator, extract_audio_from_video, separate_dme
from .speaker_splitter import split_speakers

__all__ = [
    "extract_audio_from_video",
    "detect_soundseparator",
    "separate_dme",
    "split_speakers",
    "build_foley_prompt",
    "find_hunyuan_foley",
    "generate_foley",
]

# coding: utf-8
"""CineSS 映像解析パッケージ。

動画の基本情報・無音判定を行う ``inspect_video`` と、
人物/顔追跡・口元モーション・動き重心・シーン特徴を抽出する
``analyze_video_spatial`` を提供する。
"""
from __future__ import annotations

from .analyzer import (
    SILENT_RMS_THRESHOLD_DB,
    analyze_video_spatial,
    inspect_video,
    pixel_to_azimuth_elevation,
)

__all__ = [
    "SILENT_RMS_THRESHOLD_DB",
    "inspect_video",
    "analyze_video_spatial",
    "pixel_to_azimuth_elevation",
]

# coding: utf-8
"""映像解析モジュール。

FFprobe / FFmpeg による動画検査・無音自動判定と、OpenCV による
人物・顔マルチトラック追跡、口元モーション活性度計測、動き重心
（オプティカルフロー＋フレーム差分）抽出、シーン特徴サマリー生成を行う。

座標規約（CineSS共通）:
- ``azimuth_deg`` ∈ [-180, 180]（0°=正面、負=左（画面左端≈-48°）、正=右（画面右端≈+48°））
- ``elevation_deg`` ∈ [-45, 45]（画面上=正）
- ``distance_m`` ∈ [0.8, 12.0]（大きく映る人物ほど近く≈1.2〜2.0m、小さいほど遠く≈3.5〜8.0m）
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

SILENT_RMS_THRESHOLD_DB = -55.0
HALF_HFOV_DEG = 46.0  # 水平FOV約92°想定（画面左右端で ±46°、安全上限 ±48°）
MIN_DISTANCE_M = 0.8
MAX_DISTANCE_M = 12.0


def _find_binary(name: str) -> str:
    """FFmpeg / FFprobe の実行ファイルパスを探す。"""
    found = shutil.which(name)
    if found:
        return found
    if name == "ffprobe":
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            sibling = Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
            if sibling.is_file():
                return str(sibling)
    raise RuntimeError(f"{name} が見つかりません。FFmpeg をインストールして PATH に通してください。")


def _parse_fps(rate_str: str | None) -> float:
    """ffprobe の分数FPS表現（例 '30000/1001'）を float に変換する。"""
    if not rate_str or rate_str in ("0/0", "N/A"):
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        try:
            n, d = float(num), float(den)
            return n / d if d > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def _measure_audio_rms(video_path: Path, max_seconds: float = 600.0) -> tuple[float, float]:
    """FFmpeg で音声を 16kHz モノラル float32 に抽出して RMS / Peak (dBFS) を計測する。"""
    ffmpeg = _find_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-v", "error",
        "-i", str(video_path),
        "-vn",
        "-t", str(max_seconds),
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        return -120.0, -120.0
    byte_len = len(proc.stdout) - (len(proc.stdout) % 4)
    if byte_len <= 0:
        return -120.0, -120.0
    samples = np.frombuffer(proc.stdout[:byte_len], dtype=np.float32)
    if samples.size == 0 or not np.isfinite(samples).any():
        return -120.0, -120.0
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(samples)))
    rms_db = round(20.0 * math.log10(rms), 2) if rms > 1e-10 else -120.0
    peak_db = round(20.0 * math.log10(peak), 2) if peak > 1e-10 else -120.0
    return rms_db, peak_db


def inspect_video(video_path: str | Path) -> dict[str, Any]:
    """動画の基本メタデータ・音声RMS・無音判定・推奨処理モードを返す。"""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

    ffprobe = _find_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe による動画解析に失敗しました: {proc.stderr.strip()}")

    try:
        probe = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe 出力のJSON解析に失敗しました: {exc}") from exc

    streams = probe.get("streams", [])
    fmt = probe.get("format", {})
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(v_stream.get("width", 0)) if v_stream else 0
    height = int(v_stream.get("height", 0)) if v_stream else 0
    fps = 0.0
    if v_stream:
        fps = _parse_fps(v_stream.get("avg_frame_rate")) or _parse_fps(v_stream.get("r_frame_rate"))

    duration_sec = 0.0
    for candidate in (
        fmt.get("duration"),
        v_stream.get("duration") if v_stream else None,
        a_stream.get("duration") if a_stream else None,
    ):
        if candidate is not None and candidate != "N/A":
            try:
                val = float(candidate)
                if math.isfinite(val) and val > 0:
                    duration_sec = val
                    break
            except ValueError:
                pass

    # ffprobeで解像度・尺が取れない場合のOpenCVフォールバック
    if width <= 0 or height <= 0 or duration_sec <= 0.0 or fps <= 0.0:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            try:
                if width <= 0:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                if height <= 0:
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if fps <= 0.0:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                if duration_sec <= 0.0:
                    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                    if frame_count > 0 and fps > 0:
                        duration_sec = frame_count / fps
            finally:
                cap.release()

    has_audio_stream = a_stream is not None
    audio_channels = int(a_stream.get("channels", 0)) if a_stream else 0
    audio_sample_rate = int(a_stream.get("sample_rate", 0) or 0) if a_stream else 0

    if has_audio_stream:
        audio_rms_db, audio_peak_db = _measure_audio_rms(path)
    else:
        audio_rms_db, audio_peak_db = -120.0, -120.0

    is_silent = (not has_audio_stream) or (audio_rms_db < SILENT_RMS_THRESHOLD_DB)
    suggested_mode = "silent_foley" if is_silent else "voiced_remix"

    return {
        "video_path": str(path.resolve()),
        "filename": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "duration_sec": round(float(duration_sec), 3),
        "width": int(width),
        "height": int(height),
        "fps": round(float(fps if fps > 0 else 30.0), 3),
        "has_audio_stream": bool(has_audio_stream),
        "audio_channels": audio_channels,
        "audio_sample_rate": audio_sample_rate,
        "audio_rms_db": float(audio_rms_db),
        "audio_peak_db": float(audio_peak_db),
        "is_silent": bool(is_silent),
        "suggested_mode": suggested_mode,
    }


def pixel_to_azimuth_elevation(
    cx: float,
    cy: float,
    width: int,
    height: int,
    half_hfov_deg: float = HALF_HFOV_DEG,
) -> tuple[float, float]:
    """画面座標 (cx, cy) を方位角（左=負, 右=正）と仰角（上=正）へ変換する。"""
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    nx = (cx / w) * 2.0 - 1.0  # -1 (左端) .. +1 (右端)
    ny = 1.0 - (cy / h) * 2.0  # +1 (上端) .. -1 (下端)

    tan_h = math.tan(math.radians(half_hfov_deg))
    tan_v = tan_h * (h / w)
    az = math.degrees(math.atan(nx * tan_h))
    el = math.degrees(math.atan(ny * tan_v))
    az = max(-48.0, min(48.0, az))
    el = max(-45.0, min(45.0, el))
    return round(az, 2), round(el, 2)


def _estimate_distance_m(
    height_ratio: float,
    kind: str = "face",
    depth_norm: float | None = None,
) -> float:
    """バウンディングボックスの高さ比率（＋任意の相対深度値）から距離[m]を推定する。"""
    if kind == "face":
        # 画面高さの35%の顔→約1.3m、22%→約2.0m、10%→約4.5m、6%→約7.5m
        geom_dist = 0.45 / max(height_ratio, 0.03)
    elif kind == "upperbody":
        geom_dist = 1.05 / max(height_ratio, 0.06)
    else:
        geom_dist = 1.65 / max(height_ratio, 0.09)
    geom_dist = max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, geom_dist))

    if depth_norm is not None and math.isfinite(depth_norm):
        # Depth Anything V2 出力の正規化視差（1=手前, 0=奥）を 1.0〜8.0m に写像して融合
        clamped = max(0.0, min(1.0, float(depth_norm)))
        depth_dist = 1.0 + (1.0 - clamped) * 7.0
        dist = 0.55 * depth_dist + 0.45 * geom_dist
    else:
        dist = geom_dist
    return round(max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, dist)), 2)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1))


def _nms_detections(detections: list[dict[str, Any]], iou_thresh: float = 0.35) -> list[dict[str, Any]]:
    """顔検出を優先したシンプルなNMS。"""
    if len(detections) <= 1:
        return detections
    priority = {"face": 3, "profile_face": 2, "upperbody": 1, "person": 0}
    ordered = sorted(
        detections,
        key=lambda d: (priority.get(d["kind"], 0), d.get("confidence", 0.5), d["bbox"][2] * d["bbox"][3]),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for det in ordered:
        duplicate = False
        for k in kept:
            if _bbox_iou(det["bbox"], k["bbox"]) >= iou_thresh:
                duplicate = True
                break
            # 顔が既に上半身/全身枠の中に含まれていれば重複扱い
            if k["kind"] in ("face", "profile_face") and det["kind"] in ("upperbody", "person"):
                kx, ky, kw, kh = k["bbox"]
                dx, dy, dw, dh = det["bbox"]
                kcx, kcy = kx + kw * 0.5, ky + kh * 0.5
                if dx <= kcx <= dx + dw and dy <= kcy <= dy + dh:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)
    return kept


class _OptionalDepthEstimator:
    """ローカルに導入済みの場合のみ Depth Anything V2 Small を利用する（暗黙DLしない）。"""

    def __init__(self, enabled: bool) -> None:
        self.available = False
        self._processor = None
        self._model = None
        self._device = "cpu"
        if not enabled:
            return
        self._try_load()

    def _try_load(self) -> None:
        try:
            from app import config as ciness_config
            models_dir = ciness_config.MODELS_DIR
        except Exception:
            models_dir = Path(__file__).resolve().parents[2] / "models"

        candidates = [
            models_dir / "depth-anything-v2-small",
            models_dir / "depth_anything_v2_small",
        ]
        model_dir = next(
            (p for p in candidates if p.is_dir() and (p / "config.json").is_file()),
            None,
        )
        if model_dir is None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
            self._model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), local_files_only=True)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device).eval()
            self.available = True
        except Exception:
            self._processor = None
            self._model = None
            self.available = False

    def estimate_map(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """フレームの相対視差マップ（0..1、1が最も手前）を返す。"""
        if not self.available or self._model is None or self._processor is None:
            return None
        try:
            import torch

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
                depth = outputs.predicted_depth.squeeze().detach().cpu().numpy().astype(np.float32)
            d_min, d_max = float(np.percentile(depth, 2)), float(np.percentile(depth, 98))
            if d_max - d_min < 1e-6:
                return None
            norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
            return cv2.resize(norm, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None


class _DetectorBundle:
    """OpenCV 4.x/5.x 両対応の顔・人物検出器群（ネットワーク通信不要）。

    優先順:
    1. ローカル YuNet ONNX（`models/*yunet*.onnx` 導入時、`cv2.FaceDetectorYN`）
    2. OpenCV 4.x Haar Cascade（正面顔・横顔・上半身）＋ HOG 人物検出
    3. OpenCV 5.x 等で Haar が無い環境やスタイル映像向けの
       YCrCb/HSV 肌領域＋左右対称性・目口エッジ構造ベースの顔検出フォールバック
    """

    def __init__(self) -> None:
        haar_dir = Path(getattr(getattr(cv2, "data", None), "haarcascades", ""))
        self._has_cascade = hasattr(cv2, "CascadeClassifier")
        self.frontal_default = self._load_cascade(haar_dir / "haarcascade_frontalface_default.xml")
        self.frontal_alt = self._load_cascade(haar_dir / "haarcascade_frontalface_alt2.xml")
        self.profile = self._load_cascade(haar_dir / "haarcascade_profileface.xml")
        self.upperbody = self._load_cascade(haar_dir / "haarcascade_upperbody.xml")

        self.hog = None
        if hasattr(cv2, "HOGDescriptor") and hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.hog = hog
            except Exception:
                self.hog = None

        # YuNet ONNX がローカル models/ に置かれている場合は最優先で併用
        self.yunet = None
        try:
            from app import config as ciness_config
            models_dir = ciness_config.MODELS_DIR
        except Exception:
            models_dir = Path(__file__).resolve().parents[2] / "models"
        for onnx_path in sorted(models_dir.glob("*yunet*.onnx")) if models_dir.is_dir() else []:
            if hasattr(cv2, "FaceDetectorYN"):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(str(onnx_path), "", (320, 320), 0.7, 0.3, 50)
                    break
                except Exception:
                    self.yunet = None

    def _load_cascade(self, path: Path) -> Any | None:
        if not self._has_cascade or not path.is_file():
            return None
        try:
            clf = cv2.CascadeClassifier(str(path))
            return None if clf.empty() else clf
        except Exception:
            return None

    @staticmethod
    def _detect_faces_skin_geometry(frame_bgr: np.ndarray, gray: np.ndarray) -> list[dict[str, Any]]:
        """外部重み不要の顔検出（肌色YCrCb/HSV＋楕円幾何＋左右対称性＋目口コントラスト検証）。"""
        h, w = gray.shape[:2]
        min_face = max(20, int(min(w, h) * 0.055))
        max_face_w = int(w * 0.78)
        max_face_h = int(h * 0.85)

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 誤検出（暗い暖色背景やグレー壁）を抑えつつ幅広い肌トーンを捉える複合条件
        mask_ycrcb = cv2.inRange(
            ycrcb,
            np.array([62, 135, 77], dtype=np.uint8),
            np.array([250, 178, 127], dtype=np.uint8),
        )
        mask_hsv = cv2.inRange(
            hsv,
            np.array([0, 28, 65], dtype=np.uint8),
            np.array([28, 225, 255], dtype=np.uint8),
        )
        mask_hsv2 = cv2.inRange(
            hsv,
            np.array([165, 28, 65], dtype=np.uint8),
            np.array([180, 225, 255], dtype=np.uint8),
        )
        # BGR チャネル条件（R > G, R > B, 色差あり）で無彩色・薄茶背景を除外
        b_ch, g_ch, r_ch = cv2.split(frame_bgr)
        bgr_ok = (
            (r_ch > 75)
            & (r_ch > g_ch)
            & (r_ch > b_ch)
            & ((r_ch.astype(np.int16) - np.minimum(g_ch, b_ch).astype(np.int16)) > 18)
        ).astype(np.uint8) * 255

        skin_mask = cv2.bitwise_and(mask_ycrcb, cv2.bitwise_or(mask_hsv, mask_hsv2))
        skin_mask = cv2.bitwise_and(skin_mask, bgr_ok)

        k_sz = max(3, int(round(min(w, h) * 0.015)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        # 目や口で凹んだ内部穴を埋めて1つの顔楕円として捉える
        k_close = max(5, int(round(min(w, h) * 0.035)) | 1)
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_big)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < (min_face * min_face) * 0.55:
                continue
            fx, fy, fw, fh = cv2.boundingRect(cnt)
            if fw < min_face or fh < min_face or fw > max_face_w or fh > max_face_h:
                continue
            aspect = fw / max(fh, 1)
            if not (0.50 <= aspect <= 1.45):
                continue
            extent = area / max(float(fw * fh), 1.0)
            if extent < 0.42:
                continue

            roi = gray[fy:fy + fh, fx:fx + fw]
            if roi.size < 64:
                continue

            # 1) 内部テクスチャ確認（単色の壁や箱を除外）
            roi_std = float(np.std(roi))
            if roi_std < 9.0:
                continue

            # 2) 左右対称性スコア（正面〜斜め顔は高い左右相関を持つ）
            roi_small = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            left_h = roi_small[:, :16]
            right_h = roi_small[:, 16:][:, ::-1]
            lc = left_h - float(np.mean(left_h))
            rc = right_h - float(np.mean(right_h))
            sym_denom = float(np.linalg.norm(lc) * np.linalg.norm(rc))
            sym_corr = float(np.sum(lc * rc) / sym_denom) if sym_denom > 1e-6 else 0.0
            if sym_corr < 0.25:
                continue

            # 3) 上部（目・眉）と下部（口・顎）の水平エッジ構造チェック
            grad_y = np.abs(cv2.Sobel(roi_small, cv2.CV_32F, 0, 1, ksize=3))
            eye_band = float(np.mean(grad_y[6:17, 4:28]))
            mouth_band = float(np.mean(grad_y[19:30, 6:26]))
            if eye_band < 8.0 or (eye_band + mouth_band) < 18.0:
                continue

            conf = min(0.88, max(0.48, 0.42 + 0.30 * sym_corr + 0.16 * min(1.0, eye_band / 60.0)))
            detections.append({
                "bbox": (int(fx), int(fy), int(fw), int(fh)),
                "kind": "face",
                "confidence": round(conf, 3),
            })

        return detections

    def detect(self, frame_bgr: np.ndarray, gray: np.ndarray) -> list[dict[str, Any]]:
        h, w = gray.shape[:2]
        min_face = max(20, int(min(w, h) * 0.045))
        detections: list[dict[str, Any]] = []

        # 1) YuNet ONNX（存在時）
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                _, faces = self.yunet.detect(frame_bgr)
                if faces is not None:
                    for row in faces:
                        fx, fy, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                        score = float(row[-1])
                        if fw >= min_face and fh >= min_face:
                            fx = max(0, min(w - 1, fx))
                            fy = max(0, min(h - 1, fy))
                            fw = max(1, min(w - fx, fw))
                            fh = max(1, min(h - fy, fh))
                            detections.append({"bbox": (fx, fy, fw, fh), "kind": "face", "confidence": score})
            except Exception:
                pass

        # 2) Haar 正面顔検出（OpenCV 4.x 等で利用可能な場合）
        eq_gray = cv2.equalizeHist(gray)
        if not detections and self.frontal_alt is not None:
            rects = self.frontal_alt.detectMultiScale(
                eq_gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face, min_face)
            )
            for (fx, fy, fw, fh) in rects:
                detections.append({"bbox": (int(fx), int(fy), int(fw), int(fh)), "kind": "face", "confidence": 0.85})

        if self.frontal_default is not None:
            rects = self.frontal_default.detectMultiScale(
                eq_gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face, min_face)
            )
            for (fx, fy, fw, fh) in rects:
                detections.append({"bbox": (int(fx), int(fy), int(fw), int(fh)), "kind": "face", "confidence": 0.78})

        # 3) 正面顔が1つも取れなければ横顔も試行
        if not detections and self.profile is not None:
            rects = self.profile.detectMultiScale(
                eq_gray, scaleFactor=1.12, minNeighbors=4, minSize=(min_face, min_face)
            )
            for (fx, fy, fw, fh) in rects:
                detections.append({"bbox": (int(fx), int(fy), int(fw), int(fh)), "kind": "profile_face", "confidence": 0.65})
            flipped = cv2.flip(eq_gray, 1)
            rects_flip = self.profile.detectMultiScale(
                flipped, scaleFactor=1.12, minNeighbors=4, minSize=(min_face, min_face)
            )
            for (fx, fy, fw, fh) in rects_flip:
                orig_x = max(0, w - int(fx) - int(fw))
                detections.append({"bbox": (orig_x, int(fy), int(fw), int(fh)), "kind": "profile_face", "confidence": 0.65})

        # 4) OpenCV 5.x（Haar非同梱）または顔未検出時: 肌色＋対称性＋目口エッジ顔検出
        if not detections:
            detections.extend(self._detect_faces_skin_geometry(frame_bgr, gray))

        # 5) 顔が見つからないフレームのみ上半身・HOG人物フォールバック
        if not detections and self.upperbody is not None:
            min_ub = max(40, int(min(w, h) * 0.12))
            rects = self.upperbody.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_ub, min_ub)
            )
            for (fx, fy, fw, fh) in rects:
                detections.append({"bbox": (int(fx), int(fy), int(fw), int(fh)), "kind": "upperbody", "confidence": 0.5})

        if not detections and self.hog is not None and w >= 128 and h >= 128:
            scale = min(1.0, 480.0 / float(max(w, h)))
            small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray
            if small.shape[0] >= 128 and small.shape[1] >= 64:
                rects, weights = self.hog.detectMultiScale(
                    small, winStride=(8, 8), padding=(4, 4), scale=1.08
                )
                for (fx, fy, fw, fh), wt in zip(rects, weights):
                    if float(wt) >= 0.65:
                        inv = 1.0 / scale
                        detections.append({
                            "bbox": (int(fx * inv), int(fy * inv), int(fw * inv), int(fh * inv)),
                            "kind": "person",
                            "confidence": min(0.8, float(wt) * 0.5),
                        })

        return _nms_detections(detections, iou_thresh=0.35)


def _extract_mouth_patch(gray: np.ndarray, bbox: tuple[int, int, int, int], kind: str) -> np.ndarray | None:
    """検出枠から口元領域（顔下部1/3）を切り出し正規化パッチ (24, 32) を返す。"""
    h_img, w_img = gray.shape[:2]
    x, y, w, h = bbox
    if kind in ("face", "profile_face"):
        x1 = int(max(0, round(x + 0.18 * w)))
        x2 = int(min(w_img, round(x + 0.82 * w)))
        y1 = int(max(0, round(y + 0.60 * h)))
        y2 = int(min(h_img, round(y + 0.96 * h)))
    else:
        x1 = int(max(0, round(x + 0.28 * w)))
        x2 = int(min(w_img, round(x + 0.72 * w)))
        y1 = int(max(0, round(y + 0.14 * h)))
        y2 = int(min(h_img, round(y + 0.36 * h)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = gray[y1:y2, x1:x2]
    patch = cv2.resize(roi, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    # 軽度の照明変動を吸収しつつ口の開閉コントラストを保持
    patch = patch - float(np.mean(patch))
    return patch


class _PersonTracker:
    """IoU + 重心距離ベースのマルチ人物トラッカー。"""

    def __init__(self, width: int, height: int, max_missed_sec: float = 2.0) -> None:
        self.width = max(width, 1)
        self.height = max(height, 1)
        self.diag = math.hypot(self.width, self.height)
        self.max_missed_sec = max_missed_sec
        self._next_id = 0
        self.tracks: list[dict[str, Any]] = []

    def update(
        self,
        t: float,
        detections: list[dict[str, Any]],
        gray: np.ndarray,
        depth_map: np.ndarray | None = None,
    ) -> None:
        active_indices = [
            idx for idx, tr in enumerate(self.tracks)
            if (t - tr["last_t"]) <= self.max_missed_sec
        ]
        matched_det: set[int] = set()
        matched_tr: set[int] = set()

        if active_indices and detections:
            cost = np.full((len(active_indices), len(detections)), 1e6, dtype=np.float64)
            for r, tr_idx in enumerate(active_indices):
                tb = self.tracks[tr_idx]["last_bbox"]
                tcx, tcy = tb[0] + tb[2] * 0.5, tb[1] + tb[3] * 0.5
                for c, det in enumerate(detections):
                    db = det["bbox"]
                    dcx, dcy = db[0] + db[2] * 0.5, db[1] + db[3] * 0.5
                    iou = _bbox_iou(tb, db)
                    norm_dist = math.hypot(dcx - tcx, dcy - tcy) / max(self.diag, 1.0)
                    if iou >= 0.15 or norm_dist <= 0.22:
                        cost[r, c] = (1.0 - iou) * 0.5 + norm_dist * 1.5
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1.0:
                    tr_idx = active_indices[r]
                    self._append_observation(self.tracks[tr_idx], t, detections[c], gray, depth_map)
                    matched_tr.add(tr_idx)
                    matched_det.add(c)

        for c, det in enumerate(detections):
            if c not in matched_det:
                tr = {
                    "raw_id": self._next_id,
                    "last_bbox": det["bbox"],
                    "last_t": t,
                    "prev_mouth_patch": None,
                    "keyframes": [],
                }
                self._next_id += 1
                self._append_observation(tr, t, det, gray, depth_map)
                self.tracks.append(tr)

    def _append_observation(
        self,
        track: dict[str, Any],
        t: float,
        det: dict[str, Any],
        gray: np.ndarray,
        depth_map: np.ndarray | None,
    ) -> None:
        x, y, w, h = det["bbox"]
        kind = det["kind"]
        cx, cy = x + w * 0.5, y + h * 0.5
        az, el = pixel_to_azimuth_elevation(cx, cy, self.width, self.height)

        depth_val = None
        if depth_map is not None:
            patch_d = depth_map[max(0, y):min(self.height, y + h), max(0, x):min(self.width, x + w)]
            if patch_d.size > 0:
                depth_val = float(np.percentile(patch_d, 70))

        dist_m = _estimate_distance_m(h / self.height, kind=kind, depth_norm=depth_val)

        mouth_patch = _extract_mouth_patch(gray, det["bbox"], kind)
        mouth_activity = 0.0
        if mouth_patch is not None and track["prev_mouth_patch"] is not None:
            dt = t - track["last_t"]
            if 0.0 < dt <= 0.8:
                diff = float(np.mean(np.abs(mouth_patch - track["prev_mouth_patch"])))
                mouth_activity = round(min(1.0, diff / 40.0), 4)
        if mouth_patch is not None:
            track["prev_mouth_patch"] = mouth_patch

        track["last_bbox"] = det["bbox"]
        track["last_t"] = t
        track["keyframes"].append({
            "t": round(float(t), 3),
            "azimuth_deg": az,
            "elevation_deg": el,
            "distance_m": dist_m,
            "bbox": [int(x), int(y), int(w), int(h)],
            "mouth_activity": mouth_activity,
            "confidence": round(float(det.get("confidence", 0.7)), 3),
            "kind": kind,
        })

    def finalize(self, total_sampled_frames: int) -> list[dict[str, Any]]:
        min_obs = 2 if total_sampled_frames >= 3 else 1
        kept = [tr for tr in self.tracks if len(tr["keyframes"]) >= min_obs]

        # 左から右へ並び替え（person_0=画面左、person_1=画面右…）
        kept.sort(key=lambda tr: float(np.mean([k["azimuth_deg"] for k in tr["keyframes"]])))

        results: list[dict[str, Any]] = []
        for idx, tr in enumerate(kept):
            kfs = tr["keyframes"]
            # 3点移動平均で方位角・仰角・距離の微ブレを滑らかにする
            if len(kfs) >= 3:
                for field in ("azimuth_deg", "elevation_deg", "distance_m"):
                    vals = np.array([k[field] for k in kfs], dtype=np.float64)
                    padded = np.pad(vals, (1, 1), mode="edge")
                    smooth = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
                    for k, sv in zip(kfs, smooth):
                        k[field] = round(float(sv), 2)

            mean_az = round(float(np.mean([k["azimuth_deg"] for k in kfs])), 2)
            mean_el = round(float(np.mean([k["elevation_deg"] for k in kfs])), 2)
            mean_dist = round(float(np.mean([k["distance_m"] for k in kfs])), 2)
            mouth_vals = [k["mouth_activity"] for k in kfs]
            side = "左" if mean_az < -8.0 else ("右" if mean_az > 8.0 else "中央")

            results.append({
                "person_id": f"person_{idx}",
                "label": f"人物{idx + 1}（{side}）",
                "first_t": kfs[0]["t"],
                "last_t": kfs[-1]["t"],
                "detection_count": len(kfs),
                "mean_azimuth_deg": mean_az,
                "mean_elevation_deg": mean_el,
                "mean_distance_m": mean_dist,
                "mean_mouth_activity": round(float(np.mean(mouth_vals)), 4) if mouth_vals else 0.0,
                "peak_mouth_activity": round(float(np.max(mouth_vals)), 4) if mouth_vals else 0.0,
                "keyframes": kfs,
            })
        return results


def _build_scene_summary(
    brightness_list: list[float],
    warmth_list: list[float],
    motion_energies: list[float],
    person_count: int,
    cut_times: list[float],
    duration_sec: float,
) -> dict[str, Any]:
    """無音Foleyのプロンプト生成補助とUI表示に使う客観的な映像特徴サマリーを構築する。"""
    mean_bright = float(np.mean(brightness_list)) if brightness_list else 128.0
    mean_warmth = float(np.mean(warmth_list)) if warmth_list else 0.0
    mean_motion = float(np.mean(motion_energies)) if motion_energies else 0.0
    peak_motion = float(np.max(motion_energies)) if motion_energies else 0.0

    if mean_bright < 75.0:
        brightness_label = "dark"
        bright_ja = "暗め・夜間/室内風"
    elif mean_bright > 170.0:
        brightness_label = "bright"
        bright_ja = "明るい・屋外/日中風"
    else:
        brightness_label = "medium"
        bright_ja = "標準的な明るさ"

    if mean_warmth > 12.0:
        color_tone = "warm"
        tone_ja = "暖色系"
    elif mean_warmth < -12.0:
        color_tone = "cool"
        tone_ja = "寒色系"
    else:
        color_tone = "neutral"
        tone_ja = "ニュートラル"

    if peak_motion < 0.04 and mean_motion < 0.015:
        motion_label = "static"
        motion_ja = "静かなシーン"
    elif peak_motion < 0.14:
        motion_label = "gentle"
        motion_ja = "穏やかな動き"
    elif peak_motion < 0.30:
        motion_label = "moderate"
        motion_ja = "中程度の動き"
    else:
        motion_label = "dynamic"
        motion_ja = "激しい動き・アクション"

    tags: list[str] = []
    if brightness_label == "dark":
        tags.append("quiet night or dim interior room ambience")
    elif brightness_label == "bright":
        tags.append("open airy outdoor daylight ambience")
    else:
        tags.append("natural cinematic room tone")

    if motion_label == "dynamic":
        tags.append("dynamic movement whoosh, impacts, footsteps, cloth rustle")
    elif motion_label == "moderate":
        tags.append("natural object handling, gentle movement and footsteps")
    elif motion_label == "gentle":
        tags.append("subtle ambient breeze and soft foley details")
    else:
        tags.append("calm atmospheric background tone")

    if person_count >= 2:
        tags.append("multiple people presence, subtle clothing movement")
    elif person_count == 1:
        tags.append("single person presence, subtle movement")

    if len(cut_times) >= 2:
        tags.append("scene transitions with smooth ambient continuity")

    description_ja = (
        f"{bright_ja}・{tone_ja} / {motion_ja} / "
        f"検出人物 {person_count}名 / カット変化 {len(cut_times)}回"
    )

    return {
        "duration_sec": round(duration_sec, 3),
        "mean_brightness": round(mean_bright, 1),
        "brightness_label": brightness_label,
        "color_tone": color_tone,
        "mean_motion_energy": round(mean_motion, 4),
        "peak_motion_energy": round(peak_motion, 4),
        "motion_label": motion_label,
        "person_count": int(person_count),
        "cut_count": len(cut_times),
        "cut_times": [round(float(ct), 3) for ct in cut_times],
        "suggested_foley_tags": tags,
        "description_ja": description_ja,
    }


def analyze_video_spatial(
    video_path: str | Path,
    sample_fps: float = 4.0,
    use_depth_model: bool = False,
    progress_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """映像から人物の3D軌跡・口元活性度・動き重心・シーン特徴を解析する。"""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV で動画を開けませんでした: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = frame_count / fps if frame_count > 0 else 0.0

        eff_sample_fps = max(1.0, min(float(sample_fps), fps))
        step = max(1, int(round(fps / eff_sample_fps)))

        detectors = _DetectorBundle()
        depth_est = _OptionalDepthEstimator(enabled=use_depth_model)
        tracker = _PersonTracker(width, height, max_missed_sec=max(1.5, 6.0 / eff_sample_fps))

        motion_track: list[dict[str, Any]] = []
        brightness_list: list[float] = []
        warmth_list: list[float] = []
        motion_energies: list[float] = []
        cut_times: list[float] = []

        prev_small_gray: np.ndarray | None = None
        prev_hist: np.ndarray | None = None
        last_motion_pos = (0.0, 0.0, 3.2)

        frame_idx = 0
        sampled_count = 0
        last_t = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            if width <= 0 or height <= 0:
                height, width = frame.shape[:2]
                tracker.width, tracker.height = width, height
                tracker.diag = math.hypot(width, height)

            t = frame_idx / fps
            last_t = t
            sampled_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 明るさ・色調の客観計測
            brightness_list.append(float(np.mean(gray)))
            b_mean, _, r_mean = [float(np.mean(ch)) for ch in cv2.split(frame)]
            warmth_list.append(r_mean - b_mean)

            # 任意: Depth Anything V2 深度推定
            depth_map = depth_est.estimate_map(frame) if depth_est.available else None

            # 人物・顔検出とマルチトラック追跡
            detections = detectors.detect(frame, gray)
            tracker.update(t, detections, gray, depth_map=depth_map)

            # 動き重心（フレーム間差分 + Farneback オプティカルフロー）
            work_w = min(320, width)
            work_h = max(16, int(round(height * (work_w / max(width, 1)))))
            small_gray = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)

            hsv = cv2.cvtColor(cv2.resize(frame, (work_w, work_h)), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_small_gray is not None:
                diff = cv2.absdiff(small_gray, prev_small_gray).astype(np.float32) / 255.0
                mean_diff = float(np.mean(diff))
                hist_corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)) if prev_hist is not None else 1.0

                is_cut = (mean_diff > 0.28 and hist_corr < 0.55)
                if is_cut:
                    cut_times.append(round(t, 3))
                    motion_energies.append(min(1.0, mean_diff))
                    motion_track.append({
                        "t": round(t, 3),
                        "azimuth_deg": last_motion_pos[0],
                        "elevation_deg": last_motion_pos[1],
                        "distance_m": last_motion_pos[2],
                        "motion_energy": round(min(1.0, mean_diff), 4),
                        "is_cut": True,
                    })
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_small_gray, small_gray, None,
                        pyr_scale=0.5, levels=2, winsize=11,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    motion_map = 0.5 * diff + 0.5 * np.clip(mag / 8.0, 0.0, 1.0)
                    energy = float(np.mean(motion_map))
                    p98 = float(np.percentile(motion_map, 98))
                    motion_energies.append(energy)

                    thresh = max(0.05, p98 * 0.45)
                    mask = motion_map >= thresh
                    if int(np.count_nonzero(mask)) >= 8:
                        weights = motion_map * mask
                        w_sum = float(np.sum(weights))
                        ys, xs = np.indices(motion_map.shape, dtype=np.float32)
                        mx = float(np.sum(xs * weights) / max(w_sum, 1e-8))
                        my = float(np.sum(ys * weights) / max(w_sum, 1e-8))
                        az, el = pixel_to_azimuth_elevation(mx, my, work_w, work_h)
                        area_ratio = float(np.mean(mask))
                        if depth_map is not None:
                            d_small = cv2.resize(depth_map, (work_w, work_h), interpolation=cv2.INTER_AREA)
                            d_val = float(np.mean(d_small[mask]))
                            dist_m = round(max(1.0, min(8.0, 1.0 + (1.0 - d_val) * 7.0)), 2)
                        else:
                            dist_m = round(max(1.2, min(7.5, 1.4 + (1.0 - min(area_ratio * 3.5, 1.0)) * 4.5)), 2)
                        last_motion_pos = (az, el, dist_m)
                    else:
                        az, el, dist_m = last_motion_pos

                    motion_track.append({
                        "t": round(t, 3),
                        "azimuth_deg": az,
                        "elevation_deg": el,
                        "distance_m": dist_m,
                        "motion_energy": round(energy, 4),
                        "is_cut": False,
                    })
            else:
                motion_track.append({
                    "t": round(t, 3),
                    "azimuth_deg": 0.0,
                    "elevation_deg": 0.0,
                    "distance_m": 3.2,
                    "motion_energy": 0.0,
                    "is_cut": False,
                })

            prev_small_gray = small_gray
            prev_hist = hist

            if progress_cb and frame_count > 0:
                pct = min(98, int(100 * frame_idx / frame_count))
                try:
                    progress_cb(pct, f"映像空間解析中... ({pct}%)")
                except TypeError:
                    progress_cb(pct)

            frame_idx += 1
    finally:
        cap.release()

    if duration_sec <= 0.0:
        duration_sec = max(last_t + (1.0 / max(fps, 1.0)), 0.1)

    person_tracks = tracker.finalize(sampled_count)
    scene_summary = _build_scene_summary(
        brightness_list=brightness_list,
        warmth_list=warmth_list,
        motion_energies=motion_energies,
        person_count=len(person_tracks),
        cut_times=cut_times,
        duration_sec=duration_sec,
    )

    if progress_cb:
        try:
            progress_cb(100, f"映像解析完了（人物 {len(person_tracks)}名検出）")
        except TypeError:
            progress_cb(100)

    return {
        "video_path": str(path.resolve()),
        "width": int(width),
        "height": int(height),
        "fps": round(float(fps), 3),
        "sample_fps": round(float(eff_sample_fps), 3),
        "duration_sec": round(float(duration_sec), 3),
        "sampled_frames": sampled_count,
        "depth_model_used": bool(depth_est.available),
        "person_tracks": person_tracks,
        "motion_track": motion_track,
        "scene_summary": scene_summary,
    }

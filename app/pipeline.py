# coding: utf-8
"""CineSS 統合パイプライン（映像解析 → 分離/Foley → 話者別声分け → 3D配置 → サラウンドMixMaster → MP4結合）。

主要エントリ:
- ``run_job(job_dir, request, progress_cb=None, log_cb=None) -> dict``:
  動画1本のエンドツーエンド処理を実行し、``scene.json``・``result.json``・
  ``binaural_output.mp4``・サラウンドマスター（Atmos ADM BWF / Ambisonics）・
  ``original_preview.wav``・``stems.zip`` を生成する。
- ``remix_job(job_dir, updated_scene, spatial_format="atmos", target_lufs=-18.0, progress_cb=None, log_cb=None) -> dict``:
  分離済みステムを再利用し、Web UI 等で編集された 3D 配置（``updated_scene``）から
  数秒で再MixMaster＆動画再結合を行う。
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from app.audio.foley_engine import generate_foley
from app.audio.separator_bridge import extract_audio_from_video, separate_dme
from app.audio.sfx_splitter import split_sfx_layers
from app.audio.speaker_splitter import (
    _normalize_dialog_pan_mode,
    apply_dialog_pan_mode_to_keyframes,
    split_speakers,
)
from app.config import MODELS_DIR, SAMPLE_RATE
from app.director import (
    apply_director_plan_to_scene,
    build_auto_director_plan,
    validate_director_plan,
)
from app.vision.analyzer import (
    analyze_video_spatial,
    export_director_contact_sheet,
    inspect_video,
)

SPEAKER_COLORS = [
    "#38bdf8",
    "#f472b6",
    "#34d399",
    "#fbbf24",
    "#a78bfa",
    "#fb7185",
    "#2dd4bf",
    "#f97316",
]


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _emit_progress(progress_cb: Callable[..., Any] | None, pct: float, stage: str, message: str) -> None:
    if not progress_cb:
        return
    try:
        progress_cb(round(float(pct), 1), stage, message)
    except TypeError:
        try:
            progress_cb(round(float(pct), 1), message)
        except TypeError:
            try:
                progress_cb(round(float(pct), 1))
            except Exception:
                pass


def _emit_log(log_cb: Callable[..., Any] | None, message: str) -> None:
    if not log_cb or not message:
        return
    try:
        log_cb(str(message))
    except Exception:
        pass


def _make_sub_progress(
    progress_cb: Callable[..., Any] | None,
    stage: str,
    start_pct: float,
    end_pct: float,
    unit_scale: bool = False,
) -> Callable[..., Any]:
    span = max(0.0, end_pct - start_pct)

    def _sub(*args: Any, **kwargs: Any) -> None:
        raw_pct = kwargs.get("progress", args[0] if args else 0.0)
        msg = kwargs.get("message", args[1] if len(args) >= 2 else "")
        try:
            val = float(raw_pct)
            frac = max(0.0, min(1.0, val if unit_scale else (val / 100.0)))
        except (TypeError, ValueError):
            frac = 0.0
        _emit_progress(progress_cb, start_pct + span * frac, stage, str(msg))

    return _sub


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません。FFmpeg をインストールして PATH に通してください。")
    return ffmpeg


def _normalize_mode(raw: str | None) -> str:
    m = (raw or "auto").strip().lower()
    if m in ("voiced", "voiced_remix", "remix"):
        return "voiced"
    if m in ("silent_foley", "silent", "foley"):
        return "silent_foley"
    return "auto"


def _normalize_spatial_format(raw: str | None) -> str:
    """サラウンド規格文字列を共通フォーマットIDへ正規化する。

    対応規格:
    - ``"atmos"``: Dolby Atmos 互換（完全フリー構成: ITU-R BS.2076 ADM BWF + 7.1.4 + 直接バイノーラル）
    - ``"ambisonics_foa"``: 1次 FOA（4ch AmbiX）
    - ``"ambisonics_hoa"``: 3次 HOA（16ch AmbiX・既定、ambisonics_order 1〜7指定時は優先）
    - ``"ambisonics_hoa5"``: 5次 HOA（36ch AmbiX）
    - ``"ambisonics_hoa7"``: 7次 HOA（64ch AmbiX）
    """
    f = (raw or "atmos").strip().lower()
    if f in ("atmos", "dolby_atmos", "dolby_atmos_compat", "7.1.4"):
        return "atmos"
    if f in ("ambisonics_hoa7", "hoa7", "ambisonics_7", "ambisonics7"):
        return "ambisonics_hoa7"
    if f in ("ambisonics_hoa5", "hoa5", "ambisonics_5", "ambisonics5"):
        return "ambisonics_hoa5"
    if f in ("ambisonics_hoa", "hoa", "hoa3", "ambisonics_3", "ambisonics3"):
        return "ambisonics_hoa"
    if f in ("ambisonics_foa", "foa", "hoa1", "ambisonics_1", "ambisonics1"):
        return "ambisonics_foa"
    return "atmos"


def _resolve_ambisonics_order(spatial_format: str, explicit_order: Any = None) -> int:
    """spatial_format と明示指定 ambisonics_order (1..7) から Ambisonics 次数を決定する。

    `ambisonics_foa` / `ambisonics_hoa5` / `ambisonics_hoa7` のように次数が
    フォーマット名に固定されている場合はその次数を優先し、`ambisonics_hoa` 時のみ
    明示指定 `explicit_order` (1〜7、既定3) を採用する。
    """
    fixed_mapping = {
        "ambisonics_foa": 1,
        "ambisonics_hoa5": 5,
        "ambisonics_hoa7": 7,
    }
    if spatial_format in fixed_mapping:
        return fixed_mapping[spatial_format]
    if explicit_order is not None and explicit_order != "":
        try:
            val = int(explicit_order)
            if 1 <= val <= 7:
                return val
        except (TypeError, ValueError):
            pass
    return 3


def _resolve_quality_preset(raw: str | None, duration_sec: float) -> str:
    q = (raw or "auto").strip().lower()
    if q in ("high", "standard", "fast"):
        return q
    try:
        from app.hardware_advisor import check_system

        hw = check_system()
        tier = str((hw.get("recommendations") or {}).get("tier", "B")).upper()
        if tier == "S":
            return "high" if duration_sec <= 600.0 else "standard"
        if tier == "A":
            return "standard"
        return "fast"
    except Exception:
        return "standard"


def _find_local_hrtf_sofa(explicit_path: str | None = None) -> str | None:
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p.resolve())
    preferred = MODELS_DIR / "hrtf" / "mit_kemar_normal_pinna.sofa"
    if preferred.is_file() and preferred.stat().st_size > 0:
        return str(preferred.resolve())
    for folder in (MODELS_DIR / "hrtf", MODELS_DIR):
        if folder.is_dir():
            for cand in sorted(folder.glob("*.sofa")):
                if cand.is_file() and cand.stat().st_size > 0:
                    return str(cand.resolve())
    return None


def _wav_rms_db(wav_path: str | Path) -> float:
    p = Path(wav_path)
    if not p.is_file():
        return -120.0
    audio, _ = sf.read(str(p), dtype="float32", always_2d=True)
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return round(20.0 * math.log10(rms), 2) if rms > 1e-10 else -120.0


def _resolve_loudness_settings(
    loudness_mode: Any,
    target_lufs: Any,
    default_fixed: float = -18.0,
) -> tuple[str, float | None]:
    """loudness_mode と target_lufs 指定から (mode, fixed_lufs_or_none) を決定する。"""
    raw_mode = str(loudness_mode).strip().lower() if loudness_mode not in (None, "") else ""
    if raw_mode in ("match_original", "match", "auto"):
        return "match_original", None
    if raw_mode == "fixed":
        if target_lufs is not None and not isinstance(target_lufs, bool):
            try:
                val = float(target_lufs)
                if math.isfinite(val):
                    return "fixed", val
            except (TypeError, ValueError):
                pass
        return "fixed", float(default_fixed)

    if target_lufs is None:
        return "match_original", None
    if isinstance(target_lufs, str) and target_lufs.strip().lower() in ("", "match_original", "match", "auto"):
        return "match_original", None
    if not isinstance(target_lufs, bool):
        try:
            val = float(target_lufs)
            if math.isfinite(val):
                return "fixed", val
        except (TypeError, ValueError):
            pass
    return "match_original", None


def _measure_original_lufs(wav_path: Path | str | None) -> float | None:
    """元音声 WAV の ITU-R BS.1770-4 統合ラウドネス (LKFS) を計測する。"""
    if not wav_path:
        return None
    p = Path(wav_path)
    if not p.is_file():
        return None
    try:
        audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if audio.size == 0:
            return None
        from app.spatial.mastering import measure_lufs

        val = float(measure_lufs(audio, sample_rate=int(sr)))
        return round(val, 2) if math.isfinite(val) else None
    except Exception:
        return None


def _compute_effective_target_lufs(
    loudness_mode: str,
    fixed_target: float | None,
    original_lufs: float | None,
    is_silent: bool = False,
) -> float:
    """loudness_mode と実測 original_lufs から実効ターゲット LUFS を算出する。"""
    if loudness_mode == "fixed":
        return round(float(fixed_target if fixed_target is not None else -18.0), 2)
    if is_silent or original_lufs is None or not math.isfinite(original_lufs) or original_lufs <= -60.0:
        return -14.0
    return round(max(-24.0, min(-11.0, float(original_lufs))), 2)


def _call_render_spatial_mix(
    *,
    scene: dict[str, Any],
    spatial_dir: Path,
    spatial_format: str,
    ambisonics_order: int,
    target_lufs: float,
    hrtf_sofa_path: str | None,
    base_dir: Path,
    enable_dialog_ducking: bool,
    progress_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """spatial パッケージのバージョン差異を inspect.signature で吸収して render_spatial_mix を呼ぶ。"""
    from app.spatial import render_spatial_mix

    kwargs: dict[str, Any] = {
        "scene_dict": scene,
        "out_dir": str(spatial_dir),
        "spatial_format": spatial_format,
        "ambisonics_order": ambisonics_order,
        "target_lufs": float(target_lufs),
        "hrtf_sofa_path": hrtf_sofa_path,
        "base_dir": str(base_dir),
        "progress_cb": progress_cb,
    }
    sig = inspect.signature(render_spatial_mix)
    if "enable_dialog_ducking" in sig.parameters:
        kwargs["enable_dialog_ducking"] = bool(enable_dialog_ducking)
    return render_spatial_mix(**kwargs)


def _rel_or_abs(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _slice_preview_video(
    src_video: Path,
    dst_video: Path,
    start_sec: float,
    duration_sec: float | None,
    has_audio: bool,
) -> Path:
    """お試し区間（preview_start / preview_duration）をフレーム精度で切り出す。"""
    ffmpeg = _find_ffmpeg()
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-v", "error"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(src_video)]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(dst_video)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode != 0 or not dst_video.is_file():
        raise RuntimeError(f"プレビュー区間の切り出しに失敗しました: {proc.stderr.strip()}")
    return dst_video


def _mux_video_with_binaural(
    source_video: Path,
    binaural_wav: Path,
    out_mp4: Path,
    log_cb: Callable[..., Any] | None = None,
) -> Path:
    """元映像（無再エンコード優先）＋48kHzバイノーラル音声を MP4 へ結合する。"""
    ffmpeg = _find_ffmpeg()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = out_mp4.with_name(f".{out_mp4.stem}.{uuid.uuid4().hex[:8]}.tmp.mp4")

    # 1) ビデオ無再エンコード（-c:v copy）を最優先で試行
    cmd_copy = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(source_video),
        "-i", str(binaural_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "320k",
        "-ar", str(SAMPLE_RATE),
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(tmp_mp4),
    ]
    proc = subprocess.run(cmd_copy, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode == 0 and tmp_mp4.is_file() and tmp_mp4.stat().st_size > 0:
        os.replace(tmp_mp4, out_mp4)
        return out_mp4

    tmp_mp4.unlink(missing_ok=True)
    _emit_log(log_cb, "元映像コーデックをMP4コンテナ互換へ変換しながらバイノーラル音声を結合します。")

    # 2) WebM 等 MP4 にそのまま入らないコーデックは H.264 で高品質変換
    cmd_reenc = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(source_video),
        "-i", str(binaural_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-ar", str(SAMPLE_RATE),
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(tmp_mp4),
    ]
    proc2 = subprocess.run(cmd_reenc, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if proc2.returncode != 0 or not tmp_mp4.is_file():
        tmp_mp4.unlink(missing_ok=True)
        raise RuntimeError(f"バイノーラル映像(MP4)の結合に失敗しました: {proc2.stderr.strip()}")
    os.replace(tmp_mp4, out_mp4)
    return out_mp4


def _build_stems_zip(job_dir: Path, stems_dir: Path, scene_path: Path, zip_path: Path) -> Path:
    """分離済み全ステムWAVと scene.json を ZIP にまとめる。"""
    tmp_zip = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        if stems_dir.is_dir():
            for wav in sorted(stems_dir.glob("*.wav")):
                zf.write(wav, arcname=f"stems/{wav.name}")
        if scene_path.is_file():
            zf.write(scene_path, arcname="scene.json")
    os.replace(tmp_zip, zip_path)
    return zip_path


def _motion_track_to_keyframes(
    motion_track: list[dict[str, Any]],
    default_distance: float = 3.0,
) -> list[dict[str, Any]]:
    """動き重心トラックを空間音響キーフレーム配列（滑らか・適度な間引き）へ変換する。"""
    if not motion_track:
        return [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": default_distance}]

    kfs: list[dict[str, Any]] = []
    last_kept_t = -10.0
    last_az = 0.0
    for item in motion_track:
        t = round(float(item.get("t", 0.0)), 3)
        az = round(max(-180.0, min(180.0, float(item.get("azimuth_deg", 0.0)))), 2)
        el = round(max(-45.0, min(45.0, float(item.get("elevation_deg", 0.0)))), 2)
        dist = round(max(0.8, min(12.0, float(item.get("distance_m", default_distance)))), 2)
        if not kfs or (t - last_kept_t) >= 0.35 or abs(az - last_az) >= 6.0:
            kfs.append({"t": t, "azimuth_deg": az, "elevation_deg": el, "distance_m": dist})
            last_kept_t = t
            last_az = az

    if kfs[0]["t"] > 0.05:
        kfs.insert(0, {**kfs[0], "t": 0.0})
    return kfs


def _estimate_room_params(scene_summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = scene_summary or {}
    bright = summary.get("brightness_label", "medium")
    persons = int(summary.get("person_count", 0))
    if bright == "dark" and persons > 0:
        return {"rt60_sec": 0.38, "size": "small"}
    if bright == "bright" and persons == 0:
        return {"rt60_sec": 0.55, "size": "large"}
    return {"rt60_sec": 0.45, "size": "medium"}


def _mono_pan_compensation_db(wav_path: str | Path) -> float:
    """片ch寄りのステレオ効果音WAVを点音源モノ化 (0.5*(L+R)) した際のエネルギー減衰を最大+3dB補う。"""
    p = Path(wav_path)
    if not p.is_file():
        return 0.0
    try:
        audio, _ = sf.read(str(p), dtype="float32", always_2d=True)
        if audio.size == 0 or audio.shape[1] < 2:
            return 0.0
        stereo_rms = float(np.sqrt(np.mean(0.5 * (audio[:, 0].astype(np.float64) ** 2 + audio[:, 1].astype(np.float64) ** 2))))
        mono_rms = float(np.sqrt(np.mean((0.5 * (audio[:, 0].astype(np.float64) + audio[:, 1].astype(np.float64))) ** 2)))
        if mono_rms < 1e-10 or stereo_rms < 1e-10:
            return 0.0
        comp = 20.0 * math.log10(stereo_rms / mono_rms)
        return round(max(0.0, min(3.0, comp)), 2)
    except Exception:
        return 0.0


def _append_sfx_objects(
    objects: list[dict[str, Any]],
    job_dir: Path,
    stems_dir: Path,
    sfx_path: Path,
    vision_result: dict[str, Any],
    enable_sfx_multilayer: bool,
    single_id: str = "sfx",
    single_label: str = "効果音 (SFX)",
    single_role: str = "sfx",
    single_gain_db: float = 0.0,
    single_spread: float = 0.25,
    default_distance: float = 3.0,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """効果音WAVを 4層ロスレス分解または単一トラックとして scene objects へ追加する。"""
    motion_kfs = _motion_track_to_keyframes(vision_result.get("motion_track") or [], default_distance=default_distance)

    if not enable_sfx_multilayer:
        objects.append({
            "id": single_id,
            "label": single_label,
            "role": single_role,
            "audio_path": _rel_or_abs(sfx_path, job_dir),
            "is_bed": False,
            "gain_db": float(single_gain_db),
            "spread": float(single_spread),
            "color": "#fbbf24",
            "keyframes": motion_kfs,
        })
        return None

    sfx_info = split_sfx_layers(sfx_path, stems_dir, sample_rate=SAMPLE_RATE)
    color_map = {
        "sfx_sub": "#f97316",
        "sfx_ambience": "#2dd4bf",
        "sfx_impact_left": "#f59e0b",
        "sfx_impact_center": "#fbbf24",
        "sfx_impact_right": "#eab308",
    }

    for layer in sfx_info.get("layers") or []:
        layer_path = Path(layer["path"])
        name = str(layer["name"])
        role = str(layer["role"])
        is_bed = bool(layer["is_bed"])
        if not layer_path.is_file():
            continue

        if is_bed:
            base_dist = float(layer.get("distance_m", 2.0 if role == "sfx_sub" else 4.5))
            entry: dict[str, Any] = {
                "id": name,
                "label": layer.get("label", name),
                "role": role,
                "audio_path": _rel_or_abs(layer_path, job_dir),
                "is_bed": True,
                "gain_db": 0.0 if role == "sfx_sub" else -0.5,
                "spread": float(layer.get("spread", 0.5 if role == "sfx_sub" else 0.9)),
                "color": color_map.get(name, "#2dd4bf"),
                "keyframes": [{"t": 0.0, "t_sec": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": base_dist}],
            }
            if role == "sfx_sub":
                entry["duck_exempt"] = True
            objects.append(entry)
            continue

        # 衝撃層 (sfx_impact_left / center / right)
        if float(layer.get("rms_db", -120.0)) < -85.0:
            continue

        if name == "sfx_impact_left":
            kfs = [
                {
                    **kf,
                    "t_sec": kf["t"],
                    "azimuth_deg": round(max(-60.0, min(-15.0, -35.0 * 0.55 + float(kf["azimuth_deg"]) * 0.45)), 2),
                }
                for kf in motion_kfs
            ]
        elif name == "sfx_impact_right":
            kfs = [
                {
                    **kf,
                    "t_sec": kf["t"],
                    "azimuth_deg": round(max(15.0, min(60.0, 35.0 * 0.55 + float(kf["azimuth_deg"]) * 0.45)), 2),
                }
                for kf in motion_kfs
            ]
        else:
            kfs = [{**kf, "t_sec": kf["t"]} for kf in motion_kfs]

        objects.append({
            "id": name,
            "label": layer.get("label", name),
            "role": role,
            "audio_path": _rel_or_abs(layer_path, job_dir),
            "is_bed": False,
            "gain_db": _mono_pan_compensation_db(layer_path),
            "spread": float(layer.get("spread", 0.22)),
            "color": color_map.get(name, "#fbbf24"),
            "keyframes": kfs,
        })

    _emit_log(
        log_cb,
        f"SFX 4層ロスレス分解完了: 層数={len(sfx_info.get('layers') or [])} / "
        f"再構成誤差={sfx_info.get('reconstruction_error_db')} dB / 衝撃イベント={len(sfx_info.get('impact_events') or [])}件",
    )
    return sfx_info


def _build_voiced_scene(
    job_dir: Path,
    stems_dir: Path,
    duration_sec: float,
    vision_result: dict[str, Any],
    sep_result: dict[str, Any],
    speaker_result: dict[str, Any],
    enable_sfx_multilayer: bool = True,
    log_cb: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """有音モード用の 3D scene.json と SFX 層メタデータを構築する。"""
    objects: list[dict[str, Any]] = []

    # 1) 人物別の話者オブジェクト（48kHz モノラル点音源）
    speakers = speaker_result.get("speakers") or []
    for idx, sp in enumerate(speakers):
        wav_p = Path(sp["wav_path"])
        if not wav_p.is_file() or _wav_rms_db(wav_p) < -85.0:
            continue
        person_note = sp.get("matched_person_label") or "オフスクリーン"
        objects.append({
            "id": sp["id"],
            "label": f"{sp['label']} · {person_note}",
            "role": "dialog",
            "audio_path": _rel_or_abs(wav_p, job_dir),
            "is_bed": False,
            "gain_db": 0.0,
            "spread": 0.06,
            "color": SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
            "matched_person_id": sp.get("matched_person_id"),
            "match_confidence": float(sp.get("match_confidence", 0.0)),
            "is_offscreen": bool(sp.get("is_offscreen", False)),
            "raw_keyframes": sp.get("raw_keyframes") or sp.get("keyframes") or [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 1.8}],
            "keyframes": sp.get("keyframes") or [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 1.8}],
        })

    # 重なり残留・未検出セリフがある場合もエネルギー欠損防止のため配置
    if speaker_result.get("overlap_wav"):
        ov_p = Path(speaker_result["overlap_wav"])
        if ov_p.is_file() and _wav_rms_db(ov_p) > -72.0:
            mean_az = (
                round(float(np.mean([sp.get("mean_azimuth_deg", 0.0) for sp in speakers])), 2)
                if speakers else 0.0
            )
            objects.append({
                "id": "dialog_overlap",
                "label": "重なり会話成分",
                "role": "dialog",
                "audio_path": _rel_or_abs(ov_p, job_dir),
                "is_bed": False,
                "gain_db": 0.0,
                "spread": 0.25,
                "color": "#94a3b8",
                "keyframes": [{"t": 0.0, "azimuth_deg": mean_az, "elevation_deg": 0.0, "distance_m": 2.0}],
            })

    if speaker_result.get("residual_wav"):
        res_p = Path(speaker_result["residual_wav"])
        if res_p.is_file() and _wav_rms_db(res_p) > -66.0:
            objects.append({
                "id": "dialog_residual",
                "label": "環境会話・余韻",
                "role": "dialog",
                "audio_path": _rel_or_abs(res_p, job_dir),
                "is_bed": False,
                "gain_db": -1.5,
                "spread": 0.35,
                "color": "#64748b",
                "keyframes": [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.6}],
            })

    # 話者分離で0件だった場合の安全フォールバック（dialog.wavをそのまま正面配置）
    if not objects and sep_result.get("dialog_wav"):
        dlg_p = Path(sep_result["dialog_wav"])
        if dlg_p.is_file() and _wav_rms_db(dlg_p) > -75.0:
            objects.append({
                "id": "dialog_main",
                "label": "話し声（メイン）",
                "role": "dialog",
                "audio_path": _rel_or_abs(dlg_p, job_dir),
                "is_bed": False,
                "gain_db": 0.0,
                "spread": 0.10,
                "color": "#38bdf8",
                "keyframes": [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 1.8}],
            })

    # 2) 効果音（SFX）: 映画方式4層分解 または 単一SFX配置
    sfx_info: dict[str, Any] | None = None
    if sep_result.get("effect_wav"):
        sfx_p = Path(sep_result["effect_wav"])
        if sfx_p.is_file() and _wav_rms_db(sfx_p) > -72.0:
            sfx_info = _append_sfx_objects(
                objects=objects,
                job_dir=job_dir,
                stems_dir=stems_dir,
                sfx_path=sfx_p,
                vision_result=vision_result,
                enable_sfx_multilayer=enable_sfx_multilayer,
                single_id="sfx",
                single_label="効果音 (SFX)",
                single_role="sfx",
                single_gain_db=0.0,
                single_spread=0.25,
                default_distance=2.6,
                log_cb=log_cb,
            )

    # 3) BGM / 音楽: サラウンドベッドとして空間展開
    if sep_result.get("music_wav"):
        mus_p = Path(sep_result["music_wav"])
        if mus_p.is_file() and _wav_rms_db(mus_p) > -72.0:
            objects.append({
                "id": "music",
                "label": "BGM / 音楽",
                "role": "music",
                "audio_path": _rel_or_abs(mus_p, job_dir),
                "is_bed": True,
                "gain_db": 0.0,
                "spread": 0.85,
                "color": "#c084fc",
                "keyframes": [{"t": 0.0, "t_sec": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 4.0}],
            })

    scene = {
        "version": 1,
        "mode": "voiced",
        "duration_sec": round(float(duration_sec), 3),
        "sample_rate": SAMPLE_RATE,
        "room": _estimate_room_params(vision_result.get("scene_summary")),
        "objects": objects,
    }
    return scene, sfx_info


def _build_silent_foley_scene(
    job_dir: Path,
    stems_dir: Path,
    duration_sec: float,
    vision_result: dict[str, Any],
    foley_result: dict[str, Any],
    sfx_wav: Path | None,
    ambience_wav: Path | None,
    enable_sfx_multilayer: bool = True,
    log_cb: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """無音Foleyモード用の 3D scene.json と SFX 層メタデータを構築する。"""
    objects: list[dict[str, Any]] = []
    motion_kfs = _motion_track_to_keyframes(vision_result.get("motion_track") or [], default_distance=2.6)
    sfx_info: dict[str, Any] | None = None

    if sfx_wav and sfx_wav.is_file() and _wav_rms_db(sfx_wav) > -75.0:
        sfx_info = _append_sfx_objects(
            objects=objects,
            job_dir=job_dir,
            stems_dir=stems_dir,
            sfx_path=sfx_wav,
            vision_result=vision_result,
            enable_sfx_multilayer=enable_sfx_multilayer,
            single_id="foley_sfx",
            single_label="Foley モーション効果音",
            single_role="foley",
            single_gain_db=0.5,
            single_spread=0.20,
            default_distance=2.6,
            log_cb=log_cb,
        )

    if ambience_wav and ambience_wav.is_file() and _wav_rms_db(ambience_wav) > -75.0:
        objects.append({
            "id": "foley_ambience",
            "label": "Foley 空間環境音",
            "role": "ambience",
            "audio_path": _rel_or_abs(ambience_wav, job_dir),
            "is_bed": True,
            "gain_db": -1.0,
            "spread": 0.90,
            "color": "#2dd4bf",
            "keyframes": [{"t": 0.0, "t_sec": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 4.5}],
        })

    if not objects and foley_result.get("audio_wav"):
        full_p = Path(foley_result["audio_wav"])
        if full_p.is_file():
            if enable_sfx_multilayer:
                sfx_info = _append_sfx_objects(
                    objects=objects,
                    job_dir=job_dir,
                    stems_dir=stems_dir,
                    sfx_path=full_p,
                    vision_result=vision_result,
                    enable_sfx_multilayer=True,
                    single_id="foley_main",
                    single_label="Foley 音声",
                    single_role="foley",
                    single_gain_db=0.0,
                    single_spread=0.35,
                    default_distance=2.6,
                    log_cb=log_cb,
                )
            else:
                objects.append({
                    "id": "foley_main",
                    "label": "Foley 音声",
                    "role": "foley",
                    "audio_path": _rel_or_abs(full_p, job_dir),
                    "is_bed": False,
                    "gain_db": 0.0,
                    "spread": 0.35,
                    "color": "#fbbf24",
                    "keyframes": motion_kfs,
                })

    scene = {
        "version": 1,
        "mode": "silent_foley",
        "duration_sec": round(float(duration_sec), 3),
        "sample_rate": SAMPLE_RATE,
        "room": _estimate_room_params(vision_result.get("scene_summary")),
        "objects": objects,
    }
    return scene, sfx_info


def _collect_stem_files(stems_dir: Path) -> dict[str, str]:
    if not stems_dir.is_dir():
        return {}
    return {p.name: str(p.resolve()) for p in sorted(stems_dir.glob("*.wav"))}


def _public_sfx_layers(sfx_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    """director_context / result.json 用の簡潔な SFX 層メタデータ一覧を返す。"""
    if not isinstance(sfx_info, dict):
        return []
    out: list[dict[str, Any]] = []
    for layer in sfx_info.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        out.append({
            "name": layer.get("name"),
            "label": layer.get("label"),
            "filename": layer.get("filename"),
            "path": layer.get("path"),
            "role": layer.get("role"),
            "is_bed": bool(layer.get("is_bed", False)),
            "azimuth_deg": layer.get("azimuth_deg"),
            "elevation_deg": layer.get("elevation_deg"),
            "distance_m": layer.get("distance_m"),
            "rms_db": layer.get("rms_db"),
            "peak_db": layer.get("peak_db"),
            "energy_ratio": layer.get("energy_ratio"),
        })
    return out


def _write_director_context(
    *,
    job_dir: Path,
    work_dir: Path,
    job_id: str,
    duration_sec: float,
    enable_sfx_multilayer: bool,
    enable_director: bool,
    vision_result: dict[str, Any] | None,
    sfx_info: dict[str, Any] | None,
    director_plan: dict[str, Any] | None,
    sheet_info: dict[str, Any] | None,
    sheet_path: Path,
) -> dict[str, Any]:
    """work/director_context.json を構築・保存し、その内容を返す。"""
    vr = vision_result or {}
    has_sheet = sheet_path.is_file()
    context = {
        "job_id": job_id,
        "duration_sec": round(float(duration_sec), 3),
        "enable_sfx_multilayer": bool(enable_sfx_multilayer),
        "enable_director": bool(enable_director),
        "intensity_track": vr.get("intensity_track") or [],
        "scene_segments": vr.get("scene_segments") or [],
        "scene_summary": vr.get("scene_summary") or {},
        "sfx_layers": _public_sfx_layers(sfx_info),
        "sfx_events": list((sfx_info or {}).get("impact_events") or []),
        "director_plan": director_plan,
        "director_sheet": sheet_info,
        "director_sheet_path": str(sheet_path.resolve()) if has_sheet else None,
        "director_sheet_url": f"/api/jobs/{job_id}/media/director_sheet" if has_sheet else None,
    }
    _atomic_write_json(work_dir / "director_context.json", context)
    return context


def run_job(
    job_dir: Path | str,
    request: dict[str, Any],
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
    enable_sfx_multilayer: bool | None = None,
    enable_director: bool | None = None,
    director_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CineSS 統合パイプラインのメイン実行関数。"""
    t0 = time.time()
    jdir = Path(job_dir).resolve()
    work_dir = jdir / "work"
    stems_dir = jdir / "stems"
    spatial_dir = jdir / "spatial"
    for d in (jdir, work_dir, stems_dir, spatial_dir):
        d.mkdir(parents=True, exist_ok=True)

    video_path = Path(str(request.get("video_path") or "")).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"入力動画が見つかりません: {video_path}")

    job_id = str(request.get("job_id") or jdir.name)
    req_mode = _normalize_mode(request.get("mode"))
    spatial_format = _normalize_spatial_format(request.get("spatial_format"))
    ambisonics_order = _resolve_ambisonics_order(spatial_format, request.get("ambisonics_order"))
    dialog_pan_mode = _normalize_dialog_pan_mode(request.get("dialog_pan_mode"))
    enable_dialog_ducking = bool(request.get("enable_dialog_ducking", True))
    raw_target_lufs = request.get("target_lufs") if "target_lufs" in request else None
    loudness_mode, fixed_target_lufs = _resolve_loudness_settings(request.get("loudness_mode"), raw_target_lufs)
    foley_prompt = str(request.get("foley_prompt") or "")
    allow_foley_fallback = bool(request.get("allow_foley_fallback", False))
    preview_start = max(0.0, float(request.get("preview_start") or 0.0))
    preview_duration_raw = request.get("preview_duration")
    preview_duration = float(preview_duration_raw) if preview_duration_raw not in (None, "", 0, 0.0) else None
    enable_dereverb = bool(request.get("enable_dereverb", True))
    force_dsp_fallback = bool(request.get("force_dsp_fallback", False))
    use_depth_model = bool(request.get("use_depth_model", False))

    eff_sfx_multilayer = (
        bool(enable_sfx_multilayer)
        if enable_sfx_multilayer is not None
        else bool(request.get("enable_sfx_multilayer", True))
    )
    eff_director = (
        bool(enable_director)
        if enable_director is not None
        else bool(request.get("enable_director", True))
    )
    eff_director_plan = director_plan if director_plan is not None else request.get("director_plan")

    # --- 1) 映像検査 & 有音/無音自動判定 ---
    _emit_progress(progress_cb, 3, "vision", "動画メタデータと音声レベルを検査中...")
    video_info = inspect_video(video_path)
    effective_mode = (
        ("silent_foley" if video_info["is_silent"] else "voiced")
        if req_mode == "auto"
        else req_mode
    )
    quality = _resolve_quality_preset(request.get("quality_preset"), video_info["duration_sec"])
    _emit_log(
        log_cb,
        f"映像検査完了: {video_info['width']}x{video_info['height']} / {video_info['duration_sec']:.1f}秒 / "
        f"音声RMS={video_info['audio_rms_db']} dBFS → モード={effective_mode}, 品質={quality}, "
        f"規格={spatial_format}, セリフ定位={dialog_pan_mode}, ラウドネス={loudness_mode}, "
        f"SFX4層={'ON' if eff_sfx_multilayer else 'OFF'}, AI監督={'ON' if eff_director else 'OFF'}",
    )

    # --- 2) 任意のプレビュー区間切り出し ---
    working_video = video_path
    is_preview_sliced = False
    if preview_start > 0.0 or (preview_duration is not None and preview_duration < video_info["duration_sec"] - 0.1):
        _emit_progress(progress_cb, 6, "vision", "指定されたお試し区間を切り出し中...")
        working_video = _slice_preview_video(
            video_path,
            jdir / "preview_input.mp4",
            start_sec=preview_start,
            duration_sec=preview_duration,
            has_audio=video_info["has_audio_stream"],
        )
        video_info = inspect_video(working_video)
        is_preview_sliced = True
        _emit_log(log_cb, f"プレビュー区間を切り出しました（長さ {video_info['duration_sec']:.2f} 秒）。")

    duration_sec = float(video_info["duration_sec"])

    # --- 3) 映像空間解析（人物・顔追跡 / 口元モーション / 動き重心 / シーン温度） ---
    sample_fps = {"fast": 3.0, "standard": 4.0, "high": 5.0}.get(quality, 4.0)
    if duration_sec > 180.0:
        sample_fps = min(sample_fps, 2.5)
    _emit_progress(progress_cb, 8, "vision", "OpenCV で人物・口元モーション・動き重心・シーン温度を解析中...")
    vision_result = analyze_video_spatial(
        working_video,
        sample_fps=sample_fps,
        use_depth_model=use_depth_model,
        progress_cb=_make_sub_progress(progress_cb, "vision", 8.0, 24.0),
    )
    _atomic_write_json(work_dir / "vision_analysis.json", vision_result)
    person_tracks = vision_result.get("person_tracks") or []
    _emit_log(
        log_cb,
        f"映像解析完了: 検出人物={len(person_tracks)}名 / 区間数={len(vision_result.get('scene_segments') or [])} / "
        f"{vision_result['scene_summary']['description_ja']}",
    )

    sheet_path = work_dir / "director_sheet.jpg"
    sheet_info: dict[str, Any] | None = None
    try:
        sheet_info = export_director_contact_sheet(working_video, vision_result, sheet_path, max_frames=8)
    except Exception as exc:
        _emit_log(log_cb, f"コンタクトシート生成をスキップしました: {exc}")

    original_preview_wav = jdir / "original_preview.wav"
    models_used: dict[str, Any] = {
        "vision": "opencv_haar_flow" + ("+depth_anything_v2" if vision_result.get("depth_model_used") else ""),
        "separator": None,
        "dereverb": False,
        "diarization": None,
        "overlap": None,
        "foley": None,
        "sfx_splitter": None,
        "director": None,
        "hrtf": None,
        "spatial_format": spatial_format,
    }
    speakers_summary: list[dict[str, Any]] = []
    original_lufs: float | None = None
    effective_target_lufs: float
    sfx_info: dict[str, Any] | None = None

    # --- 4) 有音モード or 無音Foleyモード ---
    if effective_mode == "voiced":
        _emit_progress(progress_cb, 26, "separate", "動画から 48kHz ステレオ音声を抽出中...")
        raw_audio_wav = work_dir / "original_48k.wav"
        extract_audio_from_video(working_video, raw_audio_wav, sample_rate=SAMPLE_RATE)
        shutil.copy2(raw_audio_wav, original_preview_wav)

        original_lufs = _measure_original_lufs(original_preview_wav)
        effective_target_lufs = _compute_effective_target_lufs(
            loudness_mode, fixed_target_lufs, original_lufs, is_silent=False
        )
        if loudness_mode == "match_original":
            _emit_log(
                log_cb,
                f"元音声ラウドネス実測: {original_lufs if original_lufs is not None else '計測不能'} LKFS "
                f"→ 目標ラウドネス {effective_target_lufs:.1f} LKFS に整合します。",
            )

        _emit_progress(progress_cb, 30, "separate", "音声を 話し声 / 効果音 / BGM に3分離中...")
        sep_result = separate_dme(
            raw_audio_wav,
            stems_dir,
            quality=quality,
            enable_dereverb=enable_dereverb,
            force_fallback=force_dsp_fallback,
            progress_cb=_make_sub_progress(progress_cb, "separate", 30.0, 52.0),
            log_cb=log_cb,
        )
        models_used["separator"] = sep_result["method"]
        models_used["dereverb"] = bool(sep_result.get("dereverb_applied", False))

        _emit_progress(progress_cb, 54, "speakers", "人物別に話し声を分離＆映像の口元と紐付け中...")
        speaker_result = split_speakers(
            sep_result["dialog_wav"],
            stems_dir,
            vision_tracks=person_tracks,
            quality=quality,
            force_fallback=force_dsp_fallback,
            dialog_pan_mode=dialog_pan_mode,
            progress_cb=_make_sub_progress(progress_cb, "speakers", 54.0, 68.0),
            log_cb=log_cb,
        )
        models_used["diarization"] = speaker_result["method"]
        models_used["overlap"] = speaker_result["overlap_method"]
        speakers_summary = [
            {
                "id": sp["id"],
                "label": sp["label"],
                "matched_person_id": sp.get("matched_person_id"),
                "matched_person_label": sp.get("matched_person_label"),
                "match_confidence": sp.get("match_confidence", 0.0),
                "is_offscreen": sp.get("is_offscreen", False),
                "mean_azimuth_deg": sp.get("mean_azimuth_deg", 0.0),
                "mean_distance_m": sp.get("mean_distance_m", 2.0),
                "duration_speaking_sec": sp.get("duration_speaking_sec", 0.0),
            }
            for sp in speaker_result.get("speakers") or []
        ]

        _emit_progress(progress_cb, 70, "mapping", "人物別3D空間シーンと SFX 多層分解を構築中...")
        scene, sfx_info = _build_voiced_scene(
            jdir,
            stems_dir,
            duration_sec,
            vision_result,
            sep_result,
            speaker_result,
            enable_sfx_multilayer=eff_sfx_multilayer,
            log_cb=log_cb,
        )
    else:
        # 無音比較用の無音 original_preview.wav を作成
        n_silent = max(SAMPLE_RATE // 2, int(round(duration_sec * SAMPLE_RATE)))
        sf.write(str(original_preview_wav), np.zeros((n_silent, 2), dtype=np.float32), SAMPLE_RATE, subtype="FLOAT")

        effective_target_lufs = _compute_effective_target_lufs(
            loudness_mode, fixed_target_lufs, None, is_silent=True
        )

        _emit_progress(progress_cb, 28, "separate", "無音動画から Foley 音声を生成中...")
        foley_full_wav = stems_dir / "foley_full.wav"
        foley_result = generate_foley(
            working_video,
            foley_full_wav,
            scene_analysis=vision_result,
            user_prompt=foley_prompt,
            quality=quality,
            allow_preview_fallback=allow_foley_fallback,
            progress_cb=_make_sub_progress(progress_cb, "separate", 28.0, 58.0),
            log_cb=log_cb,
        )
        models_used["foley"] = foley_result["method"]

        sfx_wav: Path | None = Path(foley_result["sfx_mono_wav"]) if foley_result.get("sfx_mono_wav") else None
        amb_wav: Path | None = Path(foley_result["ambience_wav"]) if foley_result.get("ambience_wav") else None

        # 実HunyuanVideo-Foley出力が単一WAVの場合は、可能ならDME分離して点音源SFXと環境音へ分ける
        if sfx_wav is None and amb_wav is None and foley_full_wav.is_file():
            _emit_progress(progress_cb, 60, "speakers", "Foley音声を点音源SFXと空間環境音に分離中...")
            try:
                foley_sep = separate_dme(
                    foley_full_wav,
                    stems_dir,
                    quality="fast" if quality == "fast" else "standard",
                    enable_dereverb=False,
                    force_fallback=force_dsp_fallback,
                    progress_cb=_make_sub_progress(progress_cb, "speakers", 60.0, 70.0),
                    log_cb=log_cb,
                )
                models_used["separator"] = foley_sep["method"]
                sfx_wav = Path(foley_sep["effect_wav"])
                amb_wav = Path(foley_sep["music_wav"])
            except Exception as exc:
                _emit_log(log_cb, f"Foleyの追加分離をスキップし、単一Foleyトラックで配置します: {exc}")
        else:
            _emit_progress(progress_cb, 68, "speakers", "Foley効果音と環境音レイヤーを準備完了")

        _emit_progress(progress_cb, 70, "mapping", "動き重心同期 3D空間シーンと SFX 多層分解を構築中...")
        scene, sfx_info = _build_silent_foley_scene(
            jdir,
            stems_dir,
            duration_sec,
            vision_result,
            foley_result,
            sfx_wav,
            amb_wav,
            enable_sfx_multilayer=eff_sfx_multilayer,
            log_cb=log_cb,
        )

    if sfx_info is not None:
        models_used["sfx_splitter"] = "hpss_4layer_lossless"

    scene["mix_settings"] = {
        "dialog_pan_mode": dialog_pan_mode,
        "enable_dialog_ducking": enable_dialog_ducking,
        "loudness_mode": loudness_mode,
        "original_lufs": original_lufs,
        "target_lufs": effective_target_lufs,
        "enable_sfx_multilayer": eff_sfx_multilayer,
        "enable_director": eff_director,
    }

    # 監督プラン適用前のクリーンなベースシーンを保存（再適用時の冪等ベース復元用）
    _atomic_write_json(work_dir / "scene_base.json", scene)

    active_director_plan: dict[str, Any] | None = None
    if eff_director:
        _emit_progress(progress_cb, 73, "mapping", "AIダビング監督がシーン温度に基づく時間変化演出を設計中...")
        raw_plan = eff_director_plan if eff_director_plan is not None else build_auto_director_plan(vision_result, sfx_info=sfx_info)
        active_director_plan = validate_director_plan(raw_plan, duration_sec)
        scene = apply_director_plan_to_scene(scene, active_director_plan)
        models_used["director"] = active_director_plan.get("source", "auto_director")
        _emit_log(log_cb, f"AIダビング監督演出を適用しました: {active_director_plan.get('summary_ja')}")

    scene_path = jdir / "scene.json"
    _atomic_write_json(scene_path, scene)
    _write_director_context(
        job_dir=jdir,
        work_dir=work_dir,
        job_id=job_id,
        duration_sec=duration_sec,
        enable_sfx_multilayer=eff_sfx_multilayer,
        enable_director=eff_director,
        vision_result=vision_result,
        sfx_info=sfx_info,
        director_plan=active_director_plan,
        sheet_info=sheet_info,
        sheet_path=sheet_path,
    )
    _emit_log(log_cb, f"3D空間シーンを構築しました（音源オブジェクト数={len(scene['objects'])}）。")

    # --- 5) サラウンド MixMaster ＆ バイノーラルレンダリング ---
    _emit_progress(progress_cb, 75, "spatial", f"空間音響MixMasterを実行中 ({spatial_format})...")
    sofa_path = _find_local_hrtf_sofa(request.get("hrtf_sofa_path"))
    spatial_out = _call_render_spatial_mix(
        scene=scene,
        spatial_dir=spatial_dir,
        spatial_format=spatial_format,
        ambisonics_order=ambisonics_order,
        target_lufs=effective_target_lufs,
        hrtf_sofa_path=sofa_path,
        base_dir=jdir,
        enable_dialog_ducking=enable_dialog_ducking,
        progress_cb=_make_sub_progress(progress_cb, "spatial", 75.0, 91.0, unit_scale=True),
    )
    metrics = dict(spatial_out.get("metrics") or {})
    models_used["hrtf"] = metrics.get("hrtf_source", "physics_hrtf")

    binaural_wav = Path(spatial_out["binaural_wav"]).resolve()
    master_spatial_wav = Path(spatial_out["master_spatial_wav"]).resolve()
    surround_714_wav = (
        Path(spatial_out["surround_714_wav"]).resolve()
        if spatial_out.get("surround_714_wav") and Path(spatial_out["surround_714_wav"]).is_file()
        else None
    )

    # ルート直下にも分かりやすくバイノーラルWAVコピーを用意
    top_binaural_wav = jdir / "binaural.wav"
    if binaural_wav != top_binaural_wav:
        shutil.copy2(binaural_wav, top_binaural_wav)
        binaural_wav = top_binaural_wav

    # --- 6) 元映像＋バイノーラル48kHz音声の MP4 結合 ＆ ステムZIP作成 ---
    _emit_progress(progress_cb, 93, "mux", "元映像とバイノーラル音声をMP4へ結合＆ステムZIPを作成中...")
    binaural_mp4 = _mux_video_with_binaural(working_video, binaural_wav, jdir / "binaural_output.mp4", log_cb=log_cb)
    stems_zip = _build_stems_zip(jdir, stems_dir, scene_path, jdir / "stems.zip")

    director_ctx_path = work_dir / "director_context.json"
    elapsed = round(time.time() - t0, 2)
    result = {
        "ok": True,
        "mode": effective_mode,
        "spatial_format": spatial_format,
        "ambisonics_order": ambisonics_order if spatial_format != "atmos" else None,
        "quality_preset": quality,
        "target_lufs": effective_target_lufs,
        "effective_target_lufs": effective_target_lufs,
        "loudness_mode": loudness_mode,
        "original_lufs": original_lufs,
        "dialog_pan_mode": dialog_pan_mode,
        "enable_dialog_ducking": enable_dialog_ducking,
        "enable_sfx_multilayer": eff_sfx_multilayer,
        "enable_director": eff_director,
        "duration_sec": duration_sec,
        "is_preview_sliced": is_preview_sliced,
        "elapsed_sec": elapsed,
        "working_video": str(working_video.resolve()),
        "scene_path": str(scene_path.resolve()),
        "video_info": video_info,
        "vision_summary": vision_result.get("scene_summary"),
        "scene_segments": vision_result.get("scene_segments") or [],
        "sfx_layers": _public_sfx_layers(sfx_info),
        "sfx_events": list((sfx_info or {}).get("impact_events") or []),
        "director_plan": active_director_plan,
        "director_context_path": str(director_ctx_path.resolve()),
        "director_sheet_path": str(sheet_path.resolve()) if sheet_path.is_file() else None,
        "person_tracks_count": len(person_tracks),
        "speakers": speakers_summary,
        "models_used": models_used,
        "metrics": metrics,
        # サーバー＆エージェント両互換のトップレベル＋artifactsパス群
        "binaural_output_mp4": str(binaural_mp4.resolve()),
        "binaural_video": str(binaural_mp4.resolve()),
        "binaural_wav": str(binaural_wav.resolve()),
        "master_spatial_wav": str(master_spatial_wav.resolve()),
        "surround_714_wav": str(surround_714_wav.resolve()) if surround_714_wav else None,
        "original_preview_wav": str(original_preview_wav.resolve()),
        "stems_zip": str(stems_zip.resolve()),
        "artifacts": {
            "original_video": str(working_video.resolve()),
            "binaural_video": str(binaural_mp4.resolve()),
            "original_audio": str(original_preview_wav.resolve()),
            "binaural_audio": str(binaural_wav.resolve()),
            "master_spatial": str(master_spatial_wav.resolve()),
            "surround_714": str(surround_714_wav.resolve()) if surround_714_wav else None,
            "stems_zip": str(stems_zip.resolve()),
            "director_sheet": str(sheet_path.resolve()) if sheet_path.is_file() else None,
            "director_context": str(director_ctx_path.resolve()),
            "stem_files": _collect_stem_files(stems_dir),
        },
    }
    _atomic_write_json(jdir / "result.json", result)
    _emit_progress(progress_cb, 100, "done", f"処理が完了しました（{elapsed:.1f}秒）")
    _emit_log(
        log_cb,
        f"全工程完了（{elapsed:.1f}秒）: LUFS={metrics.get('integrated_lufs')} / "
        f"TP={metrics.get('true_peak_dbtp')} dBTP / HRTF={models_used['hrtf']}",
    )
    return result


def remix_job(
    job_dir: Path | str,
    updated_scene: dict[str, Any],
    spatial_format: str = "atmos",
    target_lufs: float | str | None = -18.0,
    ambisonics_order: int | None = None,
    loudness_mode: str | None = None,
    enable_dialog_ducking: bool | None = None,
    dialog_pan_mode: str | None = None,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """分離済みステムを再利用し、更新された3D配置 (updated_scene) で高速再MixMaster＆MP4再結合を行う。"""
    t0 = time.time()
    jdir = Path(job_dir).resolve()
    spatial_dir = jdir / "spatial"
    stems_dir = jdir / "stems"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    prev_result = _read_json(jdir / "result.json", default={}) or {}
    req_data = _read_json(jdir / "request.json", default={}) or {}

    if not isinstance(updated_scene, dict) or not updated_scene.get("objects"):
        raise ValueError("updated_scene には 1 つ以上の objects が必要です。")

    norm_format = _normalize_spatial_format(spatial_format)
    # ambisonics_order は明示指定時のみ優先し、省略時は指定フォーマット名から次数を決定する
    # （前回 request.json の旧次数へ巻き戻る不具合を防止）
    eff_order = _resolve_ambisonics_order(norm_format, ambisonics_order)

    prev_mix = dict(updated_scene.get("mix_settings") or {})
    eff_pan_mode = _normalize_dialog_pan_mode(
        dialog_pan_mode
        if dialog_pan_mode is not None
        else (prev_mix.get("dialog_pan_mode") or prev_result.get("dialog_pan_mode") or req_data.get("dialog_pan_mode"))
    )
    if enable_dialog_ducking is not None:
        eff_ducking = bool(enable_dialog_ducking)
    elif "enable_dialog_ducking" in prev_mix:
        eff_ducking = bool(prev_mix["enable_dialog_ducking"])
    else:
        eff_ducking = bool(prev_result.get("enable_dialog_ducking", req_data.get("enable_dialog_ducking", True)))

    eff_loudness_mode, fixed_target = _resolve_loudness_settings(loudness_mode, target_lufs)

    orig_audio = prev_result.get("original_preview_wav") or (
        str((jdir / "original_preview.wav").resolve()) if (jdir / "original_preview.wav").is_file() else None
    )
    is_silent_job = (prev_result.get("mode") == "silent_foley") or (orig_audio is None)
    measured_orig_lufs = None if prev_result.get("mode") == "silent_foley" else _measure_original_lufs(orig_audio)
    original_lufs = measured_orig_lufs if measured_orig_lufs is not None else prev_result.get("original_lufs")
    lufs = _compute_effective_target_lufs(
        eff_loudness_mode, fixed_target, original_lufs, is_silent=is_silent_job
    )

    # シーンの基本必須フィールド・mix_settings を補完して保存
    scene = copy.deepcopy(updated_scene)
    recompute_pan = bool(scene.pop("recompute_dialog_pan", False))
    if recompute_pan:
        for obj in scene.get("objects") or []:
            if str(obj.get("role", "")).lower() == "dialog" and obj.get("raw_keyframes"):
                obj["keyframes"] = apply_dialog_pan_mode_to_keyframes(obj["raw_keyframes"], eff_pan_mode)

    scene.setdefault("sample_rate", SAMPLE_RATE)
    if not scene.get("duration_sec"):
        scene["duration_sec"] = float(prev_result.get("duration_sec") or 0.0)
    scene.setdefault("room", {"rt60_sec": 0.45, "size": "medium"})
    scene["mix_settings"] = {
        "dialog_pan_mode": eff_pan_mode,
        "enable_dialog_ducking": eff_ducking,
        "loudness_mode": eff_loudness_mode,
        "original_lufs": original_lufs,
        "target_lufs": lufs,
    }
    scene_path = jdir / "scene.json"
    _atomic_write_json(scene_path, scene)

    # 結合に使う元映像パスを特定
    working_video: Path | None = None
    for cand in (
        prev_result.get("working_video"),
        (prev_result.get("artifacts") or {}).get("original_video"),
        str(jdir / "preview_input.mp4"),
        req_data.get("video_path"),
    ):
        if cand and Path(cand).is_file():
            working_video = Path(cand).resolve()
            break
    if working_video is None:
        raise FileNotFoundError(f"再結合対象の元映像が見つかりません（job_dir={jdir}）。")

    _emit_progress(progress_cb, 15, "spatial", f"更新後の3D配置で再MixMaster中 ({norm_format})...")
    _emit_log(
        log_cb,
        f"再MixMasterを開始します（規格={norm_format}, 目標={lufs:.1f} LKFS [{eff_loudness_mode}], "
        f"定位={eff_pan_mode}, ダッキング={'ON' if eff_ducking else 'OFF'}, 音源数={len(scene['objects'])}）。",
    )

    sofa_path = _find_local_hrtf_sofa(req_data.get("hrtf_sofa_path"))
    spatial_out = _call_render_spatial_mix(
        scene=scene,
        spatial_dir=spatial_dir,
        spatial_format=norm_format,
        ambisonics_order=eff_order,
        target_lufs=lufs,
        hrtf_sofa_path=sofa_path,
        base_dir=jdir,
        enable_dialog_ducking=eff_ducking,
        progress_cb=_make_sub_progress(progress_cb, "spatial", 15.0, 82.0, unit_scale=True),
    )

    metrics = dict(spatial_out.get("metrics") or {})
    binaural_wav = Path(spatial_out["binaural_wav"]).resolve()
    master_spatial_wav = Path(spatial_out["master_spatial_wav"]).resolve()
    surround_714_wav = (
        Path(spatial_out["surround_714_wav"]).resolve()
        if spatial_out.get("surround_714_wav") and Path(spatial_out["surround_714_wav"]).is_file()
        else None
    )

    top_binaural_wav = jdir / "binaural.wav"
    if binaural_wav != top_binaural_wav:
        shutil.copy2(binaural_wav, top_binaural_wav)
        binaural_wav = top_binaural_wav

    _emit_progress(progress_cb, 86, "mux", "新しいバイノーラル音声を元映像へ再結合中...")
    binaural_mp4 = _mux_video_with_binaural(working_video, binaural_wav, jdir / "binaural_output.mp4", log_cb=log_cb)
    stems_zip = _build_stems_zip(jdir, stems_dir, scene_path, jdir / "stems.zip")

    elapsed = round(time.time() - t0, 2)
    models_used = dict(prev_result.get("models_used") or {})
    models_used["hrtf"] = metrics.get("hrtf_source", models_used.get("hrtf") or "physics_hrtf")
    models_used["spatial_format"] = norm_format

    result = {
        **prev_result,
        "ok": True,
        "spatial_format": norm_format,
        "ambisonics_order": eff_order if norm_format != "atmos" else None,
        "target_lufs": lufs,
        "effective_target_lufs": lufs,
        "loudness_mode": eff_loudness_mode,
        "original_lufs": original_lufs,
        "dialog_pan_mode": eff_pan_mode,
        "enable_dialog_ducking": eff_ducking,
        "remix_elapsed_sec": elapsed,
        "working_video": str(working_video.resolve()),
        "scene_path": str(scene_path.resolve()),
        "models_used": models_used,
        "metrics": metrics,
        "binaural_output_mp4": str(binaural_mp4.resolve()),
        "binaural_video": str(binaural_mp4.resolve()),
        "binaural_wav": str(binaural_wav.resolve()),
        "master_spatial_wav": str(master_spatial_wav.resolve()),
        "surround_714_wav": str(surround_714_wav.resolve()) if surround_714_wav else None,
        "original_preview_wav": orig_audio,
        "stems_zip": str(stems_zip.resolve()),
        "artifacts": {
            **{
                k: v
                for k, v in (prev_result.get("artifacts") or {}).items()
                if k in ("director_sheet", "director_context")
            },
            "original_video": str(working_video.resolve()),
            "binaural_video": str(binaural_mp4.resolve()),
            "original_audio": orig_audio,
            "binaural_audio": str(binaural_wav.resolve()),
            "master_spatial": str(master_spatial_wav.resolve()),
            "surround_714": str(surround_714_wav.resolve()) if surround_714_wav else None,
            "stems_zip": str(stems_zip.resolve()),
            "stem_files": _collect_stem_files(stems_dir),
        },
    }
    _atomic_write_json(jdir / "result.json", result)
    _emit_progress(progress_cb, 100, "done", f"再MixMasterが完了しました（{elapsed:.1f}秒）")
    _emit_log(log_cb, f"再MixMaster完了（{elapsed:.1f}秒）: LUFS={metrics.get('integrated_lufs')}, L/R={metrics.get('binaural_lr_balance_db')} dB")
    return result


def apply_director_plan_and_remix(
    job_dir: Path | str,
    director_plan: dict[str, Any],
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """既存の work/ 済みデータとベースシーンを再利用し、監督プランを再適用して高速リミックスする。"""
    jdir = Path(job_dir).resolve()
    work_dir = jdir / "work"
    prev_result = _read_json(jdir / "result.json", default={}) or {}
    req_data = _read_json(jdir / "request.json", default={}) or {}

    scene_json = _read_json(jdir / "scene.json", default=None)
    scene_base = _read_json(work_dir / "scene_base.json", default=None)

    source_scene: dict[str, Any] | None = None
    if isinstance(scene_json, dict) and scene_json.get("objects"):
        has_base = any(isinstance(o, dict) and "base_keyframes" in o for o in scene_json["objects"])
        if has_base or scene_base is None:
            source_scene = scene_json
        else:
            source_scene = scene_base
    elif isinstance(scene_base, dict) and scene_base.get("objects"):
        source_scene = scene_base

    if not isinstance(source_scene, dict) or not source_scene.get("objects"):
        raise FileNotFoundError(f"リミックス元の scene.json / scene_base.json が見つかりません（job_dir={jdir}）。")

    duration_sec = float(source_scene.get("duration_sec") or prev_result.get("duration_sec") or 1.0)
    validated_plan = validate_director_plan(director_plan, duration_sec)
    updated_scene = apply_director_plan_to_scene(source_scene, validated_plan)

    mix_cfg = dict(updated_scene.get("mix_settings") or {})
    mix_cfg["enable_director"] = True
    updated_scene["mix_settings"] = mix_cfg

    spatial_format = str(prev_result.get("spatial_format") or req_data.get("spatial_format") or "atmos")
    ambisonics_order = prev_result.get("ambisonics_order") or req_data.get("ambisonics_order")
    loudness_mode = prev_result.get("loudness_mode") or mix_cfg.get("loudness_mode") or req_data.get("loudness_mode")
    target_lufs: float | str | None = (
        None
        if loudness_mode == "match_original"
        else prev_result.get("target_lufs", mix_cfg.get("target_lufs", -18.0))
    )
    enable_dialog_ducking = mix_cfg.get("enable_dialog_ducking", prev_result.get("enable_dialog_ducking", True))
    dialog_pan_mode = mix_cfg.get("dialog_pan_mode", prev_result.get("dialog_pan_mode", "cinema"))

    _emit_log(log_cb, f"AIダビング監督プランを再適用してリミックスします: {validated_plan.get('summary_ja')}")
    remix_res = remix_job(
        jdir,
        updated_scene,
        spatial_format=spatial_format,
        target_lufs=target_lufs,
        ambisonics_order=ambisonics_order,
        loudness_mode=loudness_mode,
        enable_dialog_ducking=enable_dialog_ducking,
        dialog_pan_mode=dialog_pan_mode,
        progress_cb=progress_cb,
        log_cb=log_cb,
    )

    # director_context.json と result.json に最新の監督プランを同期
    ctx_path = work_dir / "director_context.json"
    ctx = _read_json(ctx_path, default={}) or {}
    if not ctx:
        vr = _read_json(work_dir / "vision_analysis.json", default={}) or {}
        job_id = str(req_data.get("job_id") or jdir.name)
        sheet_path = work_dir / "director_sheet.jpg"
        ctx = {
            "job_id": job_id,
            "duration_sec": round(duration_sec, 3),
            "enable_sfx_multilayer": bool(prev_result.get("enable_sfx_multilayer", True)),
            "enable_director": True,
            "intensity_track": vr.get("intensity_track") or [],
            "scene_segments": vr.get("scene_segments") or prev_result.get("scene_segments") or [],
            "scene_summary": vr.get("scene_summary") or prev_result.get("vision_summary") or {},
            "sfx_layers": prev_result.get("sfx_layers") or [],
            "sfx_events": prev_result.get("sfx_events") or [],
            "director_sheet_path": str(sheet_path.resolve()) if sheet_path.is_file() else None,
            "director_sheet_url": f"/api/jobs/{job_id}/media/director_sheet" if sheet_path.is_file() else None,
        }
    ctx["enable_director"] = True
    ctx["director_plan"] = validated_plan
    _atomic_write_json(ctx_path, ctx)

    models_used = dict(remix_res.get("models_used") or {})
    models_used["director"] = validated_plan.get("source", "custom_director")
    remix_res["models_used"] = models_used
    remix_res["enable_director"] = True
    remix_res["director_plan"] = validated_plan
    remix_res["director_context_path"] = str(ctx_path.resolve())
    _atomic_write_json(jdir / "result.json", remix_res)

    return remix_res

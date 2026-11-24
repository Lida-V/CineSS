"""CineSS 空間音響エンジン統一エントリポイント.

対応フォーマット（完全フリー＆オープン国際規格）:
- `"atmos"`: Dolby Atmos 互換（ITU-R BS.2076 準拠 ADM BWF + 7.1.4 Bed + Dynamic Objects + 内蔵HRTF直接バイノーラル）
  ※ DaVinci Resolve / EAR / Reaper 等と互換。TV 配信向け E-AC-3 JOC (`.ec3`) のみ外部対応ツールが必要。
- `"ambisonics_foa"`: 1次 Ambisonics FOA（4ch AmbiX: ACN + SN3D）
- `"ambisonics_hoa"`: 3次 Ambisonics HOA（16ch AmbiX 既定、`ambisonics_order` 1〜7 指定時はそちらを優先）
- `"ambisonics_hoa5"`: 5次 Ambisonics HOA（36ch AmbiX）
- `"ambisonics_hoa7"`: 7次 Ambisonics HOA（64ch AmbiX — IEM Plug-in Suite / SPARTA 上限準拠）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None  # type: ignore[assignment]

from scipy.io import wavfile as scipy_wavfile

from .adm_bwf import validate_adm_bwf, write_adm_bwf
from .ambisonics import (
    MAX_AMBISONICS_ORDER,
    MIN_AMBISONICS_ORDER,
    AmbisonicsRenderer,
    acn_sn3d_gains,
    ambisonics_channel_count,
    build_max_re_decoder_matrix,
)
from .atmos import ATMOS_714_SPEAKERS, AtmosRenderer, compute_714_panning_gains, spherical_to_atmos_xyz
from .hrtf import (
    HRTFEngine,
    RoomReverbEngine,
    SAMPLE_RATE,
    decorrelate_allpass,
    highpass_air,
    interpolate_keyframe_duck_exempt,
    interpolate_keyframe_gain_db,
    load_audio_48k,
    lowpass_rear_air,
    resample_audio,
    resolve_gain_linear_for_render,
    resolve_object_gain_db,
)
from .mastering import (
    LOUDNESS_PRESETS,
    compute_dialog_ducking_gain,
    linked_bus_compressor,
    master_audio,
    measure_audio_metrics,
    measure_lufs,
    measure_true_peak_dbtp,
    process_dialog_stem,
    true_peak_limiter,
)

# 3.5 GiB 超のマルチチャンネル WAV（例: 長尺 7次 HOA 64ch）は自動的に RF64 へ切替
RF64_AUTO_THRESHOLD_BYTES: int = int(3.5 * 1024 * 1024 * 1024)

__all__ = [
    "SAMPLE_RATE",
    "MIN_AMBISONICS_ORDER",
    "MAX_AMBISONICS_ORDER",
    "RF64_AUTO_THRESHOLD_BYTES",
    "ATMOS_714_SPEAKERS",
    "LOUDNESS_PRESETS",
    "HRTFEngine",
    "RoomReverbEngine",
    "AmbisonicsRenderer",
    "AtmosRenderer",
    "acn_sn3d_gains",
    "ambisonics_channel_count",
    "build_max_re_decoder_matrix",
    "compute_714_panning_gains",
    "spherical_to_atmos_xyz",
    "write_adm_bwf",
    "validate_adm_bwf",
    "process_dialog_stem",
    "compute_dialog_ducking_gain",
    "linked_bus_compressor",
    "true_peak_limiter",
    "measure_lufs",
    "measure_true_peak_dbtp",
    "measure_audio_metrics",
    "master_audio",
    "interpolate_keyframe_gain_db",
    "interpolate_keyframe_duck_exempt",
    "decorrelate_allpass",
    "highpass_air",
    "lowpass_rear_air",
    "load_audio_48k",
    "resample_audio",
    "resolve_spatial_format",
    "render_spatial_mix",
]


def _emit_progress(progress_cb: Callable[..., Any] | None, progress: float, message: str) -> None:
    """進捗コールバック（2引数または1引数）を安全に呼び出す。"""
    if progress_cb is None:
        return
    try:
        progress_cb(float(progress), message)
    except TypeError:
        try:
            progress_cb(message)
        except Exception:
            pass
    except Exception:
        pass


def resolve_spatial_format(spatial_format: str, ambisonics_order: int = 3) -> tuple[str, int | None]:
    """`spatial_format` と `ambisonics_order` を正規化し `(normalized_format, order_or_none)` を返す。"""
    fmt = str(spatial_format or "atmos").strip().lower()
    if fmt == "atmos":
        return "atmos", None
    if fmt == "ambisonics_foa":
        return "ambisonics_foa", 1
    if fmt == "ambisonics_hoa5":
        return "ambisonics_hoa5", 5
    if fmt == "ambisonics_hoa7":
        return "ambisonics_hoa7", 7
    if fmt in ("ambisonics_hoa", "ambisonics"):
        order = int(ambisonics_order)
        ambisonics_channel_count(order)
        if fmt == "ambisonics":
            if order == 1:
                return "ambisonics_foa", 1
            if order == 5:
                return "ambisonics_hoa5", 5
            if order == 7:
                return "ambisonics_hoa7", 7
            return "ambisonics_hoa", order
        return "ambisonics_hoa", order
    if fmt.startswith("ambisonics_hoa") and fmt[len("ambisonics_hoa") :].isdigit():
        order = int(fmt[len("ambisonics_hoa") :])
        ambisonics_channel_count(order)
        return fmt, order

    raise ValueError(
        f"未対応の spatial_format です: {spatial_format!r}"
        "（atmos / ambisonics_foa / ambisonics_hoa / ambisonics_hoa5 / ambisonics_hoa7 を指定してください）"
    )


def write_spatial_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    subtype: str = "PCM_24",
    rf64_threshold_bytes: int = RF64_AUTO_THRESHOLD_BYTES,
    force_rf64: bool = False,
) -> tuple[Path, str]:
    """音声配列を 48kHz WAV（推定サイズ 3.5GB 超または force_rf64 時は自動 RF64）として書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.float32)
    if subtype != "FLOAT":
        arr = np.clip(arr, -1.0, 1.0)

    n_samples = int(arr.shape[0])
    n_channels = 1 if arr.ndim == 1 else int(arr.shape[1])
    bytes_per_sample = 4 if subtype == "FLOAT" else 3
    estimated_bytes = n_samples * n_channels * bytes_per_sample + 4096
    use_rf64 = bool(force_rf64 or estimated_bytes >= int(rf64_threshold_bytes))
    wav_format = "RF64" if use_rf64 else "WAV"

    if sf is not None:
        sf.write(str(path), arr, int(sample_rate), subtype=subtype, format=wav_format)
    else:
        pcm16 = np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        scipy_wavfile.write(str(path), int(sample_rate), pcm16)
        wav_format = "WAV"

    return path, wav_format


def _prepare_stem_overrides(
    scene_dict: dict[str, Any],
    base_dir: str | None,
    sample_rate: int,
    enable_dialog_ducking: bool = True,
) -> tuple[dict[str, np.ndarray], bool]:
    """セリフ処理チェーンと発話連動サイドチェインダッキング（BGM -3dB / SFX・Foley -2dB）を適用する。"""
    overrides: dict[str, np.ndarray] = {}
    objects = scene_dict.get("objects") or []
    dialog_monos: list[np.ndarray] = []

    for idx, obj in enumerate(objects):
        if obj.get("muted", False) or not obj.get("enabled", True):
            continue
        if str(obj.get("role", "")).strip().lower() != "dialog":
            continue
        audio_path = obj.get("audio_path")
        if not audio_path:
            continue
        obj_id = str(obj.get("id", f"obj_{idx}"))
        audio, _ = load_audio_48k(audio_path, base_dir=base_dir, target_sr=sample_rate, mono=False)
        if obj.get("disable_dialog_protection", False):
            processed = audio
        else:
            processed = process_dialog_stem(audio, sample_rate=sample_rate)
            overrides[obj_id] = processed

        gain_lin = float(10.0 ** (float(obj.get("gain_db", 0.0)) / 20.0))
        if processed.ndim == 1:
            mono = processed.astype(np.float64)
        elif processed.shape[1] == 1:
            mono = processed[:, 0].astype(np.float64)
        else:
            mono = np.mean(processed, axis=1, dtype=np.float64)
        dialog_monos.append(mono * gain_lin)

    if not enable_dialog_ducking or not dialog_monos:
        return overrides, False

    max_len = max(m.shape[0] for m in dialog_monos)
    duration_sec = float(scene_dict.get("duration_sec") or 0.0)
    if duration_sec > 0.0:
        max_len = max(max_len, int(round(duration_sec * sample_rate)))
    if max_len <= 0:
        return overrides, False

    dialog_bus = np.zeros(max_len, dtype=np.float64)
    for m in dialog_monos:
        dialog_bus[: m.shape[0]] += m

    duck_music = compute_dialog_ducking_gain(dialog_bus, sample_rate=sample_rate, duck_db=-3.0)
    if float(np.min(duck_music)) >= 0.995:
        return overrides, False

    duck_ambience = compute_dialog_ducking_gain(dialog_bus, sample_rate=sample_rate, duck_db=-2.0)
    ducking_applied = False

    for idx, obj in enumerate(objects):
        if obj.get("muted", False) or not obj.get("enabled", True):
            continue
        role = str(obj.get("role", "")).strip().lower()
        if role == "sfx_sub" or bool(obj.get("duck_exempt", False)):
            continue
        if role not in ("music", "ambience", "sfx", "sfx_impact", "foley"):
            continue
        audio_path = obj.get("audio_path")
        if not audio_path:
            continue
        obj_id = str(obj.get("id", f"obj_{idx}"))
        audio, _ = load_audio_48k(audio_path, base_dir=base_dir, target_sr=sample_rate, mono=False)
        n_sig = audio.shape[0]
        if n_sig == 0:
            continue

        keyframes = obj.get("keyframes") or obj.get("gain_keyframes")
        if role in ("music", "ambience"):
            base_curve = duck_music if role == "music" else duck_ambience
            if base_curve.shape[0] >= n_sig:
                g = base_curve[:n_sig]
            else:
                g = np.pad(base_curve, (0, n_sig - base_curve.shape[0]), mode="constant", constant_values=1.0)
        else:
            raw_mono = (
                audio[:, 0].astype(np.float64)
                if audio.ndim == 1 or audio.shape[1] == 1
                else np.mean(audio, axis=1, dtype=np.float64)
            )
            eff_gain_db = resolve_object_gain_db(obj, n_sig, sample_rate=sample_rate)
            eff_gain_lin = resolve_gain_linear_for_render(
                n_sig, gain_db=eff_gain_db, keyframes=None, sample_rate=sample_rate
            )
            impact_mono = raw_mono * eff_gain_lin
            dlg_sig = (
                dialog_bus[:n_sig]
                if dialog_bus.shape[0] >= n_sig
                else np.pad(dialog_bus, (0, n_sig - dialog_bus.shape[0]), mode="constant")
            )
            imp_thr = float(obj.get("impact_threshold_db", -16.0))
            g = compute_dialog_ducking_gain(
                dlg_sig,
                sample_rate=sample_rate,
                duck_db=-2.0,
                impact_override_mono=impact_mono,
                impact_threshold_db=imp_thr,
            )

        if isinstance(keyframes, list) and any(isinstance(k, dict) and k.get("duck_exempt") for k in keyframes):
            sample_times = np.arange(n_sig, dtype=np.float64) / float(sample_rate)
            exempt_mask = interpolate_keyframe_duck_exempt(
                keyframes, sample_times, default_exempt=False, sample_rate=sample_rate
            )
            g_f64 = g.astype(np.float64)
            g = (g_f64 + (1.0 - g_f64) * exempt_mask).astype(np.float32)

        if float(np.min(g)) >= 0.9999:
            continue

        if audio.ndim == 1:
            overrides[obj_id] = (audio * g).astype(np.float32)
        else:
            overrides[obj_id] = (audio * g[:, np.newaxis]).astype(np.float32)
        if float(np.min(g)) < 0.995:
            ducking_applied = True

    return overrides, ducking_applied


def render_spatial_mix(
    scene_dict: dict[str, Any],
    out_dir: str,
    spatial_format: str = "atmos",
    ambisonics_order: int = 3,
    target_lufs: float = -18.0,
    hrtf_sofa_path: str | None = None,
    base_dir: str | None = None,
    progress_cb: Callable[..., Any] | None = None,
    enable_dialog_ducking: bool | None = None,
) -> dict[str, Any]:
    """CineSS 空間音響レンダリング＆マスタリング統一エントリ関数。

    Args:
        scene_dict: シーン構成辞書 (`duration_sec`, `sample_rate`, `room`, `objects`)。
        out_dir: 出力先ディレクトリ。
        spatial_format: `"atmos"` | `"ambisonics_foa"` | `"ambisonics_hoa"` | `"ambisonics_hoa5"` | `"ambisonics_hoa7"`。
        ambisonics_order: Ambisonics 次数 1〜7（`ambisonics_hoa` 時は既定 3・指定値を優先）。
        target_lufs: 目標統合ラウドネス（Cinema -24 / Atmos -18 / Web -14 LKFS）。
        hrtf_sofa_path: 実測 SOFA ファイルパス（省略または読込不可時は内蔵物理モデルへ自動フォールバック）。
        base_dir: `audio_path` が相対パスだった場合の基準ディレクトリ。
        progress_cb: 進捗コールバック `(progress: float, message: str)`。
        enable_dialog_ducking: 発話連動の BGM/SFX サイドチェインダッキング有効化（None 時は scene_dict["mix_settings"] 参照・既定 True）。

    Returns:
        `{"binaural_wav", "master_spatial_wav", "surround_714_wav", "metrics": {...}}`
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fmt, resolved_order = resolve_spatial_format(spatial_format, ambisonics_order=ambisonics_order)

    if enable_dialog_ducking is None:
        mix_settings = scene_dict.get("mix_settings")
        eff_ducking = bool(mix_settings.get("enable_dialog_ducking", True)) if isinstance(mix_settings, dict) else True
    else:
        eff_ducking = bool(enable_dialog_ducking)

    sr = SAMPLE_RATE
    _emit_progress(progress_cb, 0.05, "HRTF 空間音響エンジンを初期化しています")
    hrtf_engine = HRTFEngine(sofa_path=hrtf_sofa_path, sample_rate=sr)

    _emit_progress(progress_cb, 0.15, "ダイアログ処理とダッキング・ステム前処理を実行しています")
    stem_overrides, ducking_applied = _prepare_stem_overrides(
        scene_dict,
        base_dir=base_dir,
        sample_rate=sr,
        enable_dialog_ducking=eff_ducking,
    )

    binaural_wav = out_path / "binaural.wav"
    surround_714_wav: Path | None = None
    master_spatial_wav: Path
    master_channels: int
    master_container = "WAV"

    if fmt == "atmos":
        _emit_progress(
            progress_cb,
            0.30,
            "Dolby Atmos 互換（ITU-R BS.2076 7.1.4 Bed + Dynamic Objects）をレンダリングしています",
        )
        renderer = AtmosRenderer(hrtf_engine=hrtf_engine, sample_rate=sr)
        rendered = renderer.render_scene(scene_dict, base_dir=base_dir, audio_overrides=stem_overrides)

        _emit_progress(progress_cb, 0.65, "バイノーラル＆7.1.4ミックスをマスタリング・ラウドネス正規化しています")
        binaural_master, bin_metrics = master_audio(
            rendered["binaural"],
            target_lufs=float(target_lufs),
            ceiling_dbtp=-1.0,
            sample_rate=sr,
            enable_compressor=True,
        )
        surround_master, _ = master_audio(
            rendered["surround_714"],
            target_lufs=float(target_lufs),
            ceiling_dbtp=-1.0,
            sample_rate=sr,
            enable_compressor=True,
        )

        # ADM BWF 用の Bed と Dynamic Objects へ同一マスタリングゲイン＆安全リンク制限を適用
        applied_gain_lin = float(10.0 ** (float(bin_metrics.get("applied_gain_db", 0.0)) / 20.0))
        bed_scaled = (rendered["bed_714"].astype(np.float64) * applied_gain_lin).astype(np.float32)
        dyn_objs_scaled: list[dict[str, Any]] = []
        peak_all = float(np.max(np.abs(bed_scaled))) if bed_scaled.size else 0.0
        for dyn in rendered["dynamic_objects"]:
            sig_scaled = (dyn["audio"].astype(np.float64) * applied_gain_lin).astype(np.float32)
            if sig_scaled.size:
                peak_all = max(peak_all, float(np.max(np.abs(sig_scaled))))
            dyn_copy = dict(dyn)
            dyn_copy["audio"] = sig_scaled
            dyn_objs_scaled.append(dyn_copy)

        ceiling_lin = float(10.0 ** (-1.05 / 20.0))
        if peak_all > ceiling_lin and peak_all > 1e-12:
            safe_scale = ceiling_lin / peak_all
            bed_scaled = (bed_scaled * safe_scale).astype(np.float32)
            for dyn_copy in dyn_objs_scaled:
                dyn_copy["audio"] = (dyn_copy["audio"] * safe_scale).astype(np.float32)

        _emit_progress(progress_cb, 0.85, "バイノーラル WAV・7.1.4 WAV・ITU-R BS.2076 ADM BWF を書き出しています")
        write_spatial_wav(binaural_wav, binaural_master, sample_rate=sr, subtype="PCM_24")
        surround_714_wav, _ = write_spatial_wav(
            out_path / "surround_714.wav", surround_master, sample_rate=sr, subtype="PCM_24"
        )
        master_channels = 12 + len(dyn_objs_scaled)
        est_adm_bytes = int(bed_scaled.shape[0]) * master_channels * 3 + 65536
        force_rf64 = est_adm_bytes >= RF64_AUTO_THRESHOLD_BYTES
        master_spatial_wav = write_adm_bwf(
            out_path / "atmos_adm.wav",
            bed_audio=bed_scaled,
            dynamic_objects=dyn_objs_scaled,
            sample_rate=sr,
            bit_depth=24,
            programme_name=str(scene_dict.get("title") or "CineSS Dolby Atmos Compatible Mix"),
            integrated_lufs=float(bin_metrics["integrated_lufs"]),
            true_peak_dbtp=float(bin_metrics["true_peak_dbtp"]),
            force_rf64=force_rf64,
        )
        master_container = "RF64" if force_rf64 else "RIFF"

    else:
        assert resolved_order is not None
        order = resolved_order
        n_ch = ambisonics_channel_count(order)
        _emit_progress(progress_cb, 0.30, f"Ambisonics AmbiX ({order}次 / {n_ch}ch) をレンダリングしています")
        renderer = AmbisonicsRenderer(order=order, hrtf_engine=hrtf_engine, sample_rate=sr)
        rendered = renderer.render_scene(scene_dict, base_dir=base_dir, audio_overrides=stem_overrides)

        _emit_progress(progress_cb, 0.65, "バイノーラル出力と AmbiX マスターをマスタリングしています")
        binaural_master, bin_metrics = master_audio(
            rendered["binaural"],
            target_lufs=float(target_lufs),
            ceiling_dbtp=-1.0,
            sample_rate=sr,
            enable_compressor=True,
        )

        # AmbiX 音場は全チャンネル同一ゲイン＆リンク True Peak リミットで球面音場バランスを完全保持
        applied_gain_lin = float(10.0 ** (float(bin_metrics.get("applied_gain_db", 0.0)) / 20.0))
        ambix_scaled = (rendered["ambix"].astype(np.float64) * applied_gain_lin).astype(np.float32)
        ambix_master = true_peak_limiter(ambix_scaled, sample_rate=sr, ceiling_dbtp=-1.0)

        _emit_progress(progress_cb, 0.85, "バイノーラル WAV と AmbiX サラウンドマスター WAV を書き出しています")
        write_spatial_wav(binaural_wav, binaural_master, sample_rate=sr, subtype="PCM_24")
        fname = "ambisonics_foa_ambix.wav" if order == 1 else f"ambisonics_hoa{order}_ambix.wav"
        master_spatial_wav, master_container = write_spatial_wav(
            out_path / fname,
            ambix_master,
            sample_rate=sr,
            subtype="PCM_24",
        )
        master_channels = int(rendered["channels"])

    _emit_progress(progress_cb, 1.0, "空間音響 MixMaster が完了しました")

    metrics_out: dict[str, Any] = {
        "integrated_lufs": float(bin_metrics["integrated_lufs"]),
        "true_peak_dbtp": float(bin_metrics["true_peak_dbtp"]),
        "binaural_lr_balance_db": float(bin_metrics["lr_balance_db"]),
        "rms_l_db": float(bin_metrics["rms_l_db"]),
        "rms_r_db": float(bin_metrics["rms_r_db"]),
        "correlation": float(bin_metrics["correlation"]),
        "format": fmt,
        "hrtf_source": hrtf_engine.source_name,
        "diffuse_field_eq": bool(hrtf_engine.diffuse_field_eq_applied),
        "dialog_ducking_applied": bool(ducking_applied),
        "target_lufs": round(float(target_lufs), 2),
        "channels": int(master_channels),
        "container": master_container,
    }
    if resolved_order is not None:
        metrics_out["ambisonics_order"] = int(resolved_order)

    return {
        "binaural_wav": str(binaural_wav),
        "master_spatial_wav": str(master_spatial_wav),
        "surround_714_wav": str(surround_714_wav) if surround_714_wav is not None else None,
        "metrics": metrics_out,
    }

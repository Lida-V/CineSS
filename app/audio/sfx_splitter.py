# coding: utf-8
"""映画方式 4層ロスレス効果音（SFX）分解エンジン。

効果音ステム (`effect.wav`) を音響学的・位相整合的に 4つの役割へ相補分解する：
1. ``sfx_sub`` (Bed, 140Hz以下): LFE・重低音インパクト基盤（ゼロ位相 ``sosfiltfilt``）
2. ``sfx_ambience`` (Bed, >140Hz 持続成分): STFT + HPSS 調和/持続マスクによる空間環境成分
3. ``sfx_impact_left`` / ``sfx_impact_center`` / ``sfx_impact_right`` (Dynamic Objects, >140Hz 衝撃成分):
   STFT + HPSS パーカッシブマスクにステレオパン相補マスクを適用し、左・中央・右の衝撃成分へ分解。
   RMS が ``-62 dBFS`` 未満の実質無音な L/R チャンネルはファイル出力をスキップし、
   その成分を ``center`` へ畳み込むことで総和ロスレス性を厳密に維持する。
"""
from __future__ import annotations

import math
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, istft, resample_poly, sosfiltfilt, stft

from app.config import SAMPLE_RATE

SILENT_PAN_CHANNEL_THRESHOLD_DB = -62.0
SUB_CUTOFF_HZ = 140.0


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return round(20.0 * math.log10(rms), 2) if rms > 1e-10 else -120.0


def _peak_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    peak = float(np.max(np.abs(x)))
    return round(20.0 * math.log10(peak), 2) if peak > 1e-10 else -120.0


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float64)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(audio.astype(np.float64), int(dst_sr) // g, int(src_sr) // g, axis=0)


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - arr.shape[0]), mode="constant")


def _detect_impact_events(
    impact_band: np.ndarray,
    left_raw: np.ndarray,
    center_raw: np.ndarray,
    right_raw: np.ndarray,
    sr: int,
    max_events: int = 32,
) -> list[dict[str, Any]]:
    """衝撃成分から主要ピーク時刻 (t_sec, peak_db, dominant_side) を検出する。"""
    n_samples = impact_band.shape[0]
    if n_samples < max(64, int(0.02 * sr)):
        return []

    mono = 0.5 * (impact_band[:, 0] + impact_band[:, 1])
    overall_peak_db = _peak_db(mono)
    if overall_peak_db < -48.0:
        return []

    hop = max(1, int(round(0.010 * sr)))  # 10ms 刻み
    win = max(hop, int(round(0.025 * sr)))
    n_frames = max(1, (n_samples - win) // hop + 1)

    frame_peak = np.zeros(n_frames, dtype=np.float64)
    frame_rms = np.zeros(n_frames, dtype=np.float64)
    for idx in range(n_frames):
        s = idx * hop
        seg = mono[s : s + win]
        frame_peak[idx] = float(np.max(np.abs(seg)))
        frame_rms[idx] = float(np.sqrt(np.mean(seg * seg)))

    frame_peak_db = 20.0 * np.log10(np.maximum(frame_peak, 1e-10))
    frame_rms_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))

    active_rms = frame_rms_db[frame_rms_db > -80.0]
    baseline_db = float(np.median(active_rms)) if active_rms.size > 0 else -60.0
    height_thresh_db = max(-45.0, min(overall_peak_db - 1.5, max(baseline_db + 6.0, overall_peak_db - 18.0)))

    min_distance = max(1, int(round(0.22 / (hop / float(sr)))))
    peaks, _ = find_peaks(frame_peak_db, height=height_thresh_db, distance=min_distance, prominence=2.0)

    if len(peaks) == 0 and overall_peak_db >= -38.0:
        peaks = np.array([int(np.argmax(frame_peak_db))], dtype=np.int64)

    half_win = int(round(0.035 * sr))
    events: list[dict[str, Any]] = []
    for p_idx in peaks:
        center_sample = int(p_idx) * hop + win // 2
        t_sec = round(min(n_samples / float(sr), center_sample / float(sr)), 3)
        s = max(0, center_sample - half_win)
        e = min(n_samples, center_sample + half_win)

        # 原パンエネルギーとステレオバランスから優勢方向を判定
        e_l = float(np.mean(left_raw[s:e] ** 2)) if e > s else 0.0
        e_c = float(np.mean(center_raw[s:e] ** 2)) if e > s else 0.0
        e_r = float(np.mean(right_raw[s:e] ** 2)) if e > s else 0.0

        rms_l_ch = float(np.sqrt(np.mean(impact_band[s:e, 0] ** 2))) if e > s else 0.0
        rms_r_ch = float(np.sqrt(np.mean(impact_band[s:e, 1] ** 2))) if e > s else 0.0
        pan = (rms_r_ch - rms_l_ch) / max(rms_l_ch + rms_r_ch, 1e-12)

        if e_l > max(e_c, e_r) * 1.15 or pan < -0.22:
            side = "left"
        elif e_r > max(e_c, e_l) * 1.15 or pan > 0.22:
            side = "right"
        else:
            side = "center"

        events.append({
            "t_sec": t_sec,
            "peak_db": round(float(frame_peak_db[p_idx]), 2),
            "rms_db": round(float(frame_rms_db[p_idx]), 2),
            "dominant_side": side,
        })

    if len(events) > max_events:
        events = sorted(events, key=lambda ev: ev["peak_db"], reverse=True)[:max_events]
        events.sort(key=lambda ev: ev["t_sec"])

    return events


def split_sfx_layers(
    effect_wav: str | Path,
    out_dir: str | Path,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """効果音 WAV を映画方式の4層（Sub / Impact L・C・R / Ambience）へ厳密相補分解する。

    出力されるすべての層 WAV ファイルを時間サンプル単位で加算すると、
    浮動小数点誤差を除き元の効果音と一致する（総和ロスレス性保証）。
    """
    src_path = Path(effect_wav)
    if not src_path.is_file():
        raise FileNotFoundError(f"効果音 WAV が見つかりません: {src_path}")

    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    target_sr = int(sample_rate)

    audio, src_sr = sf.read(str(src_path), dtype="float32", always_2d=True)
    x_raw = audio.astype(np.float64)
    if int(src_sr) != target_sr:
        x_raw = _resample(x_raw, int(src_sr), target_sr)

    if x_raw.shape[1] == 1:
        x = np.repeat(x_raw, 2, axis=1)
    elif x_raw.shape[1] >= 2:
        x = x_raw[:, :2]
    else:
        x = np.zeros((x_raw.shape[0], 2), dtype=np.float64)

    n_samples = x.shape[0]
    duration_sec = n_samples / float(max(target_sr, 1))
    source_rms_db = _rms_db(x)

    # ------------------------------------------------------------------
    # Step A: 140Hz Butterworth ゼロ位相 LPF (sosfiltfilt) による Sub 層抽出
    # ------------------------------------------------------------------
    if n_samples > 6 and SUB_CUTOFF_HZ < 0.45 * target_sr:
        sos = butter(4, SUB_CUTOFF_HZ, btype="lowpass", fs=target_sr, output="sos")
        padlen = min(3 * (2 * sos.shape[0] + 1), max(0, n_samples - 1))
        low = sosfiltfilt(sos, x, axis=0, padlen=padlen)
    else:
        low = np.zeros_like(x, dtype=np.float64)
    high = x - low

    # ------------------------------------------------------------------
    # Step B: STFT + HPSS による Impact / Ambience 相補ソフトマスク分解
    # ------------------------------------------------------------------
    if n_samples >= 64:
        nperseg = min(2048, n_samples)
        if nperseg % 2 != 0:
            nperseg = max(32, nperseg - 1)
        noverlap = min(nperseg - 1, (nperseg * 3) // 4)

        _, _, z_l = stft(
            high[:, 0], fs=target_sr, window="hann",
            nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True,
        )
        _, _, z_r = stft(
            high[:, 1], fs=target_sr, window="hann",
            nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True,
        )
        mag_mix = np.sqrt(0.5 * (np.abs(z_l) ** 2 + np.abs(z_r) ** 2))

        n_freqs, n_frames = mag_mix.shape
        k_freq = min(17, n_freqs if n_freqs % 2 == 1 else max(1, n_freqs - 1))
        k_time = min(17, n_frames if n_frames % 2 == 1 else max(1, n_frames - 1))
        k_freq = max(3, k_freq) if n_freqs >= 3 else 1
        k_time = max(3, k_time) if n_frames >= 3 else 1

        # 周波数軸メディアン = Percussive (衝撃), 時間軸メディアン = Harmonic (環境・持続)
        p_spec = median_filter(mag_mix, size=(k_freq, 1), mode="reflect")
        h_spec = median_filter(mag_mix, size=(1, k_time), mode="reflect")

        eps = 1e-12
        p_pow = p_spec ** 2
        h_pow = h_spec ** 2
        m_p = p_pow / (p_pow + h_pow + eps)
        m_h = 1.0 - m_p

        _, imp_l = istft(
            z_l * m_p, fs=target_sr, window="hann",
            nperseg=nperseg, noverlap=noverlap, boundary=True,
        )
        _, imp_r = istft(
            z_r * m_p, fs=target_sr, window="hann",
            nperseg=nperseg, noverlap=noverlap, boundary=True,
        )
        _, amb_l = istft(
            z_l * m_h, fs=target_sr, window="hann",
            nperseg=nperseg, noverlap=noverlap, boundary=True,
        )
        _, amb_r = istft(
            z_r * m_h, fs=target_sr, window="hann",
            nperseg=nperseg, noverlap=noverlap, boundary=True,
        )

        impact_band = np.stack([_match_length(imp_l, n_samples), _match_length(imp_r, n_samples)], axis=1)
        ambience_init = np.stack([_match_length(amb_l, n_samples), _match_length(amb_r, n_samples)], axis=1)
        # 微小 ISTFT 境界誤差を残差保証で完全吸収（impact + ambience == high）
        stft_err = high - (impact_band + ambience_init)
        impact_band = impact_band + 0.5 * stft_err
        ambience_band = high - impact_band
    else:
        impact_band = high.copy()
        ambience_band = np.zeros_like(high, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step C: Impact 層のステレオパン相補マスク分解 (Left / Center / Right)
    # ------------------------------------------------------------------
    if n_samples >= 64:
        nperseg_pan = min(2048, n_samples)
        if nperseg_pan % 2 != 0:
            nperseg_pan = max(32, nperseg_pan - 1)
        noverlap_pan = min(nperseg_pan - 1, (nperseg_pan * 3) // 4)

        _, _, zi_l = stft(
            impact_band[:, 0], fs=target_sr, window="hann",
            nperseg=nperseg_pan, noverlap=noverlap_pan, boundary="zeros", padded=True,
        )
        _, _, zi_r = stft(
            impact_band[:, 1], fs=target_sr, window="hann",
            nperseg=nperseg_pan, noverlap=noverlap_pan, boundary="zeros", padded=True,
        )
        mag_l = np.abs(zi_l)
        mag_r = np.abs(zi_r)
        pan = (mag_r - mag_l) / (mag_l + mag_r + 1e-12)

        pan_thresh = 0.15
        pan_span = 0.45
        w_left = np.clip((-pan - pan_thresh) / pan_span, 0.0, 1.0)
        w_right = np.clip((pan - pan_thresh) / pan_span, 0.0, 1.0)

        _, l_ch0 = istft(zi_l * w_left, fs=target_sr, window="hann", nperseg=nperseg_pan, noverlap=noverlap_pan, boundary=True)
        _, l_ch1 = istft(zi_r * w_left, fs=target_sr, window="hann", nperseg=nperseg_pan, noverlap=noverlap_pan, boundary=True)
        _, r_ch0 = istft(zi_l * w_right, fs=target_sr, window="hann", nperseg=nperseg_pan, noverlap=noverlap_pan, boundary=True)
        _, r_ch1 = istft(zi_r * w_right, fs=target_sr, window="hann", nperseg=nperseg_pan, noverlap=noverlap_pan, boundary=True)

        left_sig = np.stack([_match_length(l_ch0, n_samples), _match_length(l_ch1, n_samples)], axis=1)
        right_sig = np.stack([_match_length(r_ch0, n_samples), _match_length(r_ch1, n_samples)], axis=1)
        center_sig = impact_band - left_sig - right_sig
    else:
        left_sig = np.zeros_like(impact_band, dtype=np.float64)
        right_sig = np.zeros_like(impact_band, dtype=np.float64)
        center_sig = impact_band.copy()

    impact_events = _detect_impact_events(
        impact_band=impact_band,
        left_raw=left_sig,
        center_raw=center_sig,
        right_raw=right_sig,
        sr=target_sr,
    )

    # -62 dBFS 未満の L/R はスキップして center に畳み込み、総和ロスレス性を維持
    folded_channels: list[str] = []
    emit_left = _rms_db(left_sig) >= SILENT_PAN_CHANNEL_THRESHOLD_DB
    if not emit_left:
        center_sig = center_sig + left_sig
        left_sig = np.zeros_like(left_sig)
        folded_channels.append("left")

    emit_right = _rms_db(right_sig) >= SILENT_PAN_CHANNEL_THRESHOLD_DB
    if not emit_right:
        center_sig = center_sig + right_sig
        right_sig = np.zeros_like(right_sig)
        folded_channels.append("right")

    # 書き出し仕様リスト（必ず総和 == x が成立するセットのみ出力）
    specs: list[dict[str, Any]] = [
        {
            "name": "sfx_sub",
            "filename": "sfx_sub.wav",
            "signal": low,
            "role": "sfx_sub",
            "is_bed": True,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "distance_m": 2.0,
            "spread": 0.5,
            "label": "重低音・衝撃基盤 (Sub)",
        },
        {
            "name": "sfx_ambience",
            "filename": "sfx_ambience.wav",
            "signal": ambience_band,
            "role": "ambience",
            "is_bed": True,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "distance_m": 4.5,
            "spread": 0.9,
            "label": "空間環境音 (Ambience)",
        },
    ]
    if emit_left:
        specs.append({
            "name": "sfx_impact_left",
            "filename": "sfx_impact_left.wav",
            "signal": left_sig,
            "role": "sfx_impact",
            "is_bed": False,
            "azimuth_deg": -35.0,
            "elevation_deg": 0.0,
            "distance_m": 2.2,
            "spread": 0.18,
            "label": "衝撃効果音 (左)",
        })
    specs.append({
        "name": "sfx_impact_center",
        "filename": "sfx_impact_center.wav",
        "signal": center_sig,
        "role": "sfx_impact",
        "is_bed": False,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "distance_m": 2.2,
        "spread": 0.25,
        "label": "衝撃効果音 (中央)",
    })
    if emit_right:
        specs.append({
            "name": "sfx_impact_right",
            "filename": "sfx_impact_right.wav",
            "signal": right_sig,
            "role": "sfx_impact",
            "is_bed": False,
            "azimuth_deg": 35.0,
            "elevation_deg": 0.0,
            "distance_m": 2.2,
            "spread": 0.18,
            "label": "衝撃効果音 (右)",
        })

    # 前回出力の残骸（例: 前回は L があり今回はスキップ）を混同しないよう対象名クリーンアップ
    for stale_name in ("sfx_impact_left.wav", "sfx_impact_right.wav"):
        if not any(sp["filename"] == stale_name for sp in specs):
            (dst_dir / stale_name).unlink(missing_ok=True)

    total_energy = float(np.sum(x * x)) + 1e-18
    layers: list[dict[str, Any]] = []
    layers_by_name: dict[str, dict[str, Any]] = {}
    reconstructed = np.zeros_like(x, dtype=np.float64)

    for sp in specs:
        sig_f32 = sp["signal"].astype(np.float32)
        out_path = dst_dir / sp["filename"]
        sf.write(str(out_path), sig_f32, target_sr, subtype="FLOAT")

        # 実際の書き出しファイル基準で再合成誤差も検証可能にする
        reconstructed += sig_f32.astype(np.float64)
        layer_energy = float(np.sum(sig_f32.astype(np.float64) ** 2))
        entry = {
            "name": sp["name"],
            "label": sp["label"],
            "path": str(out_path.resolve()),
            "filename": sp["filename"],
            "role": sp["role"],
            "is_bed": bool(sp["is_bed"]),
            "azimuth_deg": float(sp["azimuth_deg"]),
            "elevation_deg": float(sp["elevation_deg"]),
            "distance_m": float(sp["distance_m"]),
            "spread": float(sp["spread"]),
            "rms_db": _rms_db(sig_f32),
            "peak_db": _peak_db(sig_f32),
            "energy_ratio": round(layer_energy / total_energy, 4),
        }
        layers.append(entry)
        layers_by_name[sp["name"]] = entry

    recon_err_db = _rms_db(x - reconstructed)

    return {
        "source_wav": str(src_path.resolve()),
        "sample_rate": target_sr,
        "duration_sec": round(duration_sec, 3),
        "source_rms_db": source_rms_db,
        "reconstruction_error_db": recon_err_db,
        "folded_channels": folded_channels,
        "layers": layers,
        "layers_by_name": layers_by_name,
        "impact_events": impact_events,
    }

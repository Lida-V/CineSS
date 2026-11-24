"""Dolby Atmos 互換レンダリングエンジン（完全フリー・オープン国際規格 ITU-R BS.2076 準拠）.

商用有償エンコーダ（E-AC-3 JOC）に依存せず、7.1.4 Bed + Dynamic Audio Objects の構成で
ADM BWF (.wav)（DaVinci Resolve / EAR / Reaper 等互換）、7.1.4 12ch ミックスダウン、
および内蔵 HRTF 直接バイノーラル出力を生成する（TV向け .ec3 JOC のみ外部ツール対応）。

チャンネル配列（ITU / Dolby 7.1.4 標準 12ch）:
  0: L   (-30°,  0°)   [M+030]
  1: R   (+30°,  0°)   [M-030]
  2: C   (  0°,  0°)   [M+000]
  3: LFE (80Hz LPF)    [LFE1]
  4: Ls  (-90°,  0°)   [M+090]
  5: Rs  (+90°,  0°)   [M-090]
  6: Lrs (-135°, 0°)   [M+135]
  7: Rrs (+135°, 0°)   [M-135]
  8: Ltf (-45°, +45°)  [U+045]
  9: Rtf (+45°, +45°)  [U-045]
 10: Ltr (-135°,+45°)  [U+135]
 11: Rtr (+135°,+45°)  [U-135]

Atmos 正規化 Cartesian 座標:
- X ∈ [-1, 1]: 負=左 (-1), 正=右 (+1)
- Y ∈ [-1, 1]: 正=前 (+1), 負=後 (-1)
- Z ∈ [0, 1]:  0=耳高さ, +1=天井
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from .hrtf import (
    HRIR_BASE_DELAY_SAMPLES,
    HRTFEngine,
    RoomReverbEngine,
    SAMPLE_RATE,
    _deduplicate_sorted_keyframes,
    air_absorption_response,
    clamp_distance_m,
    clamp_elevation_deg,
    decorrelate_allpass,
    distance_direct_gain,
    distance_reverb_send,
    fit_length,
    highpass_air,
    interpolate_keyframes,
    load_audio_48k,
    lowpass_rear_air,
    resolve_gain_linear_for_render,
    resolve_object_gain_db,
    wrap_azimuth_deg,
)


@dataclass(frozen=True)
class Speaker714:
    """7.1.4 仮想スピーカー定義。"""

    index: int
    name: str
    azimuth_deg: float
    elevation_deg: float
    is_lfe: bool
    itu_label: str
    adm_x: float
    adm_y: float
    adm_z: float


ATMOS_714_SPEAKERS: tuple[Speaker714, ...] = (
    Speaker714(0, "L", -30.0, 0.0, False, "M+030", -1.0, 1.0, 0.0),
    Speaker714(1, "R", 30.0, 0.0, False, "M-030", 1.0, 1.0, 0.0),
    Speaker714(2, "C", 0.0, 0.0, False, "M+000", 0.0, 1.0, 0.0),
    Speaker714(3, "LFE", 0.0, 0.0, True, "LFE1", 0.0, 1.0, -1.0),
    Speaker714(4, "Ls", -90.0, 0.0, False, "M+090", -1.0, 0.0, 0.0),
    Speaker714(5, "Rs", 90.0, 0.0, False, "M-090", 1.0, 0.0, 0.0),
    Speaker714(6, "Lrs", -135.0, 0.0, False, "M+135", -1.0, -1.0, 0.0),
    Speaker714(7, "Rrs", 135.0, 0.0, False, "M-135", 1.0, -1.0, 0.0),
    Speaker714(8, "Ltf", -45.0, 45.0, False, "U+045", -1.0, 0.5, 1.0),
    Speaker714(9, "Rtf", 45.0, 45.0, False, "U-045", 1.0, 0.5, 1.0),
    Speaker714(10, "Ltr", -135.0, 45.0, False, "U+135", -1.0, -0.5, 1.0),
    Speaker714(11, "Rtr", 135.0, 45.0, False, "U-135", 1.0, -0.5, 1.0),
)

ATMOS_714_NON_LFE_INDICES: tuple[int, ...] = tuple(s.index for s in ATMOS_714_SPEAKERS if not s.is_lfe)

# 水平リング（方位角昇順: -135° .. +135°）とトップリング
_EAR_RING: tuple[tuple[int, float], ...] = (
    (6, -135.0),  # Lrs
    (4, -90.0),   # Ls
    (0, -30.0),   # L
    (2, 0.0),     # C
    (1, 30.0),    # R
    (5, 90.0),    # Rs
    (7, 135.0),   # Rrs
)

_TOP_RING: tuple[tuple[int, float], ...] = (
    (10, -135.0),  # Ltr
    (8, -45.0),    # Ltf
    (9, 45.0),     # Rtf
    (11, 135.0),   # Rtr
)


def spherical_to_atmos_xyz(
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float = 2.0,
) -> tuple[float, float, float]:
    """CineSS 球面座標 (azimuth_deg, elevation_deg, distance_m) を Atmos 正規化座標 (X, Y, Z) へ変換する。

    - X ∈ [-1, 1] (-1=左, +1=右)
    - Y ∈ [-1, 1] (+1=正面, -1=後方)
    - Z ∈ [0, 1]  (0=水平, +1=天井)
    """
    az = math.radians(float(wrap_azimuth_deg(azimuth_deg)))
    el = math.radians(float(clamp_elevation_deg(elevation_deg)))
    _ = clamp_distance_m(distance_m)

    c_el = math.cos(el)
    x = max(-1.0, min(1.0, c_el * math.sin(az)))
    y = max(-1.0, min(1.0, c_el * math.cos(az)))
    z = max(0.0, min(1.0, math.sin(el)))
    return float(x), float(y), float(z)


def _pairwise_ring_gains(azimuth_deg: float, ring: tuple[tuple[int, float], ...]) -> dict[int, float]:
    """円周配置スピーカーリング上で一定電力（sin/cos）ペアワイズパンニングゲインを算出する。"""
    az = float(wrap_azimuth_deg(azimuth_deg))
    n = len(ring)

    for i in range(n):
        idx_a, ang_a = ring[i]
        idx_b, ang_b = ring[(i + 1) % n]
        span = ang_b - ang_a
        if span <= 0.0:
            span += 360.0
        rel = az - ang_a
        if rel < 0.0:
            rel += 360.0
        if rel <= span + 1e-9:
            frac = max(0.0, min(1.0, rel / span))
            theta = frac * (0.5 * math.pi)
            return {idx_a: math.cos(theta), idx_b: math.sin(theta)}

    # フォールバック（最近傍）
    idx0 = ring[0][0]
    return {idx0: 1.0}


def compute_714_panning_gains(
    azimuth_deg: float,
    elevation_deg: float,
    spread: float = 0.0,
) -> np.ndarray:
    """7.1.4 12ch スピーカーへの一定電力 3D VBAP パンニングゲイン (12,) を計算する。"""
    el = float(clamp_elevation_deg(elevation_deg))
    sp = max(0.0, min(1.0, float(spread)))

    gains = np.zeros(12, dtype=np.float64)
    ear_g = _pairwise_ring_gains(azimuth_deg, _EAR_RING)
    top_g = _pairwise_ring_gains(azimuth_deg, _TOP_RING)

    if el <= 0.0:
        w_ear, w_top = 1.0, 0.0
    elif el < 45.0:
        phi = (el / 45.0) * (0.5 * math.pi)
        w_ear, w_top = math.cos(phi), math.sin(phi)
    else:
        w_ear, w_top = 0.0, 1.0

    for idx, g in ear_g.items():
        gains[idx] += w_ear * g

    if el <= 45.0:
        for idx, g in top_g.items():
            gains[idx] += w_top * g
    else:
        # 45° → 90° でトップ4ch均等（頭上）へ滑らかに遷移
        overhead_t = (el - 45.0) / 45.0
        for idx, g in top_g.items():
            gains[idx] += (1.0 - overhead_t) * g
        for idx, _ in _TOP_RING:
            gains[idx] += overhead_t * 0.5

    if sp > 1e-3:
        # spread に応じて全非LFEスピーカーへ緩やかに拡散ブレンド
        uniform = np.zeros(12, dtype=np.float64)
        for idx in ATMOS_714_NON_LFE_INDICES:
            uniform[idx] = 1.0 / math.sqrt(len(ATMOS_714_NON_LFE_INDICES))
        gains = (1.0 - 0.55 * sp) * gains + (0.55 * sp) * uniform

    # L2（一定電力）正規化
    norm = float(np.linalg.norm(gains))
    if norm > 1e-12:
        gains /= norm
    return gains


def _lfe_lowpass(mono: np.ndarray, sample_rate: int = SAMPLE_RATE, cutoff_hz: float = 80.0) -> np.ndarray:
    """LFE チャンネル用の 4次 Butterworth ローパスフィルタ（既定 80Hz）。"""
    if mono.size == 0 or float(np.max(np.abs(mono))) < 1e-10:
        return np.zeros_like(mono, dtype=np.float32)
    eff_cut = max(20.0, min(float(cutoff_hz), 0.45 * float(sample_rate)))
    sos = signal.butter(4, eff_cut, btype="lowpass", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, mono.astype(np.float64)).astype(np.float32)


def _delay_samples(x: np.ndarray, n_delay: int) -> np.ndarray:
    """サラウンド/ハイト空間展開用の安全なサンプル遅延。"""
    if n_delay <= 0 or n_delay >= x.shape[0]:
        return x.copy()
    out = np.zeros_like(x)
    out[n_delay:] = x[:-n_delay]
    return out


def _apply_distance_to_object_mono(
    mono_audio: np.ndarray,
    keyframes: list[dict[str, Any]] | None,
    gain_db: float | np.ndarray,
    rt60_sec: float,
    sample_rate: int = SAMPLE_RATE,
    block_size: int = 2048,
    role: str = "sfx",
) -> tuple[np.ndarray, np.ndarray]:
    """オブジェクト音声へ時間変化ゲイン・距離減衰・空気吸収LPFを滑らかに適用し、直接音とリバーブセンドを返す。"""
    x = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
    n_samples = x.shape[0]
    if n_samples == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    gain_lin = resolve_gain_linear_for_render(
        n_samples, gain_db=gain_db, keyframes=keyframes, sample_rate=sample_rate
    )
    x_scaled = x.astype(np.float64) * gain_lin
    if float(np.max(np.abs(x_scaled))) < 1e-9:
        return np.zeros((n_samples,), dtype=np.float32), np.zeros((n_samples,), dtype=np.float32)

    hop = block_size // 2
    window = np.hanning(block_size).astype(np.float64)
    x_pad = np.pad(x_scaled, (hop, hop + block_size), mode="constant")

    n_fft = block_size * 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    n_frames = int(math.ceil((n_samples + hop) / hop))
    frame_starts = np.arange(n_frames, dtype=np.int64) * hop
    frame_times = np.clip(frame_starts / float(sample_rate), 0.0, max(0.0, (n_samples - 1) / float(sample_rate)))
    _, _, dist_arr = interpolate_keyframes(keyframes, frame_times)

    out_pad = np.zeros(n_samples + hop * 2 + n_fft, dtype=np.float64)
    rev_pad = np.zeros(n_samples + hop * 2 + block_size, dtype=np.float64)

    for i in range(n_frames):
        start = int(frame_starts[i])
        seg = x_pad[start : start + block_size]
        if seg.shape[0] < block_size:
            seg = np.pad(seg, (0, block_size - seg.shape[0]), mode="constant")
        if float(np.max(np.abs(seg))) < 1e-11:
            continue

        seg_w = seg * window
        dist = float(dist_arr[i])
        g_dir = distance_direct_gain(dist, role=role)
        g_rev = distance_reverb_send(dist, rt60_sec=rt60_sec, role=role)
        h_air = air_absorption_response(freqs, dist, role=role)

        spec = np.fft.rfft(seg_w, n=n_fft)
        y = np.fft.irfft(spec * h_air * g_dir, n=n_fft)[: block_size + 128]
        out_pad[start : start + block_size + 128] += y
        if g_rev > 0.0:
            rev_pad[start : start + block_size] += seg_w * g_rev

    direct = out_pad[hop : hop + n_samples].astype(np.float32)
    reverb_send = rev_pad[hop : hop + n_samples].astype(np.float32)
    return direct, reverb_send


def _build_adm_keyframes(
    keyframes: list[dict[str, Any]] | None,
    duration_sec: float,
    step_sec: float = 0.05,
) -> list[dict[str, float]]:
    """オブジェクトのキーフレームを ADM audioBlockFormat 用の時間変化 Atmos (X, Y, Z) 列へ変換する。"""
    if not keyframes:
        x, y, z = spherical_to_atmos_xyz(0.0, 0.0, 2.0)
        return [{"t": 0.0, "x": x, "y": y, "z": z}]

    sorted_kf = _deduplicate_sorted_keyframes(keyframes)
    if not sorted_kf:
        x, y, z = spherical_to_atmos_xyz(0.0, 0.0, 2.0)
        return [{"t": 0.0, "x": x, "y": y, "z": z}]
    # 静的オブジェクト（1点のみ）なら 1 ブロックで十分
    if len(sorted_kf) == 1:
        k = sorted_kf[0]
        x, y, z = spherical_to_atmos_xyz(
            float(k.get("azimuth_deg", 0.0)),
            float(k.get("elevation_deg", 0.0)),
            float(k.get("distance_m", 2.0)),
        )
        return [{"t": 0.0, "x": x, "y": y, "z": z}]

    # 動的オブジェクトは滑らかな軌跡再現のため一定ステップでサンプリング（最大 1200 点程度に抑制）
    eff_step = max(step_sec, duration_sec / 1200.0) if duration_sec > 0 else step_sec
    times = np.arange(0.0, max(duration_sec, eff_step), eff_step, dtype=np.float64)
    if times.size == 0 or times[0] > 0.0:
        times = np.insert(times, 0, 0.0)
    if duration_sec > 0.0 and (duration_sec - times[-1]) > 1e-3:
        times = np.append(times, duration_sec)

    az_arr, el_arr, dist_arr = interpolate_keyframes(sorted_kf, times)
    blocks: list[dict[str, float]] = []
    for t_val, az, el, dist in zip(times, az_arr, el_arr, dist_arr):
        x, y, z = spherical_to_atmos_xyz(float(az), float(el), float(dist))
        blocks.append({"t": float(t_val), "x": x, "y": y, "z": z})
    return blocks


class AtmosRenderer:
    """Dolby Atmos（7.1.4 Bed + Dynamic Audio Objects）統合レンダラー。"""

    def __init__(
        self,
        hrtf_engine: HRTFEngine | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hrtf_engine = hrtf_engine or HRTFEngine(sample_rate=self.sample_rate)

    def render_bed_to_binaural(self, bed_714: np.ndarray) -> np.ndarray:
        """7.1.4 仮想スピーカー配置（ITU/Dolby 標準角度）の HRTF 畳み込みでバイノーラル (N, 2) を生成する。"""
        bed = np.asarray(bed_714, dtype=np.float32)
        if bed.ndim != 2 or bed.shape[1] != 12:
            raise ValueError(f"7.1.4 ベッドの形状が不正です: {bed.shape}")
        n = bed.shape[0]
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        out_l = np.zeros(n, dtype=np.float64)
        out_r = np.zeros(n, dtype=np.float64)
        delay_comp = 0 if self.hrtf_engine._use_sofa else int(round(HRIR_BASE_DELAY_SAMPLES))

        for spk in ATMOS_714_SPEAKERS:
            ch_sig = bed[:, spk.index]
            if float(np.max(np.abs(ch_sig))) < 1e-10:
                continue
            if spk.is_lfe:
                # LFE は耳介フィルタを通さず両耳へクリーンにモノ加算
                out_l += 0.5 * ch_sig.astype(np.float64)
                out_r += 0.5 * ch_sig.astype(np.float64)
                continue

            hrir_l, hrir_r = self.hrtf_engine.get_hrir(spk.azimuth_deg, spk.elevation_deg)
            conv_l = signal.fftconvolve(ch_sig, hrir_l, mode="full")
            conv_r = signal.fftconvolve(ch_sig, hrir_r, mode="full")
            end = delay_comp + n
            if end <= conv_l.shape[0]:
                out_l += conv_l[delay_comp:end]
                out_r += conv_r[delay_comp:end]
            else:
                avail = max(0, conv_l.shape[0] - delay_comp)
                if avail > 0:
                    out_l[:avail] += conv_l[delay_comp : delay_comp + avail]
                    out_r[:avail] += conv_r[delay_comp : delay_comp + avail]

        return np.stack([out_l, out_r], axis=1).astype(np.float32)

    def _route_bed_track_714(
        self,
        audio_2d: np.ndarray,
        gain_db: float | np.ndarray = 0.0,
        role: str = "music",
        keyframes: list[dict[str, Any]] | None = None,
        world_360: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """BGM・環境音・重低音ベッドトラックを 7.1.4 Bed (N, 12) へ 360° 無相関包囲配置する。"""
        arr = np.asarray(audio_2d, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n = arr.shape[0]
        bed = np.zeros((n, 12), dtype=np.float32)
        if n == 0 or float(np.max(np.abs(arr))) < 1e-10:
            return bed

        sr = self.sample_rate
        gain_lin = resolve_gain_linear_for_render(
            n, gain_db=gain_db, keyframes=keyframes, sample_rate=sr
        )
        role_norm = str(role or "music").strip().lower()

        w360 = world_360 if isinstance(world_360, dict) else {}
        w360_enabled = w360.get("enabled", True) is not False
        if w360_enabled:
            rear_scale = max(0.4, min(1.6, float(w360.get("rear_level", 1.0))))
            height_scale = max(0.3, min(1.6, float(w360.get("height_level", 1.0))))
        else:
            rear_scale = 0.65
            height_scale = 0.55

        if role_norm == "sfx_sub":
            left = (arr[:, 0] * gain_lin).astype(np.float32)
            right = (arr[:, 1] * gain_lin).astype(np.float32) if arr.shape[1] >= 2 else left
            mid = 0.5 * (left + right)
            if float(np.max(np.abs(left))) < 1e-10 and float(np.max(np.abs(right))) < 1e-10:
                return bed
            # 重低音ベッド: Front L/R (0.48) + LFE 140Hz LPF (0.88)
            bed[:, 0] += 0.48 * left
            bed[:, 1] += 0.48 * right
            bed[:, 3] += 0.88 * _lfe_lowpass(mid, sample_rate=sr, cutoff_hz=140.0)
            return bed

        is_ambience = role_norm == "ambience"

        if arr.shape[1] >= 2:
            left = (arr[:, 0] * gain_lin).astype(np.float32)
            right = (arr[:, 1] * gain_lin).astype(np.float32)
            if float(np.max(np.abs(left))) < 1e-10 and float(np.max(np.abs(right))) < 1e-10:
                return bed
            mid = 0.5 * (left + right)
            if is_ambience:
                # 環境音: 素数サンプルオールパス無相関化＋後方空気吸収LPF＋頭上HPFで真の360°包囲を生成
                rear_l = lowpass_rear_air(left, sample_rate=sr, cutoff_hz=7500.0)
                rear_r = lowpass_rear_air(right, sample_rate=sr, cutoff_hz=7500.0)
                hi_l = highpass_air(left, sample_rate=sr, cutoff_hz=320.0)
                hi_r = highpass_air(right, sample_rate=sr, cutoff_hz=320.0)
                bed[:, 0] += 0.38 * left
                bed[:, 1] += 0.38 * right
                bed[:, 4] += 0.56 * decorrelate_allpass(decorrelate_allpass(left, 211, coeff=0.50), 313, coeff=0.48)
                bed[:, 5] += 0.56 * decorrelate_allpass(decorrelate_allpass(right, 223, coeff=0.50), 331, coeff=-0.48)
                bed[:, 6] += (0.54 * rear_scale) * decorrelate_allpass(decorrelate_allpass(rear_l, 389, coeff=0.52), 503, coeff=-0.50)
                bed[:, 7] += (0.54 * rear_scale) * decorrelate_allpass(decorrelate_allpass(rear_r, 401, coeff=0.52), 521, coeff=0.50)
                bed[:, 8] += (0.42 * height_scale) * decorrelate_allpass(decorrelate_allpass(hi_l, 293, coeff=0.48), 431, coeff=0.48)
                bed[:, 9] += (0.42 * height_scale) * decorrelate_allpass(decorrelate_allpass(hi_r, 307, coeff=0.48), 443, coeff=-0.48)
                bed[:, 10] += (0.38 * height_scale * rear_scale) * decorrelate_allpass(decorrelate_allpass(hi_l, 467, coeff=0.50), 601, coeff=-0.50)
                bed[:, 11] += (0.38 * height_scale * rear_scale) * decorrelate_allpass(decorrelate_allpass(hi_r, 479, coeff=0.50), 613, coeff=0.50)
                bed[:, 3] += 0.18 * _lfe_lowpass(mid, sample_rate=sr)
            else:
                # BGM はフロントL/Rを芯に、オールパス無相関化で側後方・頭上へ自然な広がり
                rear_l = lowpass_rear_air(left, sample_rate=sr, cutoff_hz=7500.0)
                rear_r = lowpass_rear_air(right, sample_rate=sr, cutoff_hz=7500.0)
                hi_l = highpass_air(left, sample_rate=sr, cutoff_hz=320.0)
                hi_r = highpass_air(right, sample_rate=sr, cutoff_hz=320.0)
                bed[:, 0] += 0.76 * left
                bed[:, 1] += 0.76 * right
                bed[:, 2] += 0.14 * mid
                bed[:, 4] += 0.36 * decorrelate_allpass(left, 241, coeff=0.45)
                bed[:, 5] += 0.36 * decorrelate_allpass(right, 251, coeff=0.45)
                bed[:, 6] += (0.24 * rear_scale) * decorrelate_allpass(rear_l, 409, coeff=0.48)
                bed[:, 7] += (0.24 * rear_scale) * decorrelate_allpass(rear_r, 419, coeff=0.48)
                bed[:, 8] += (0.20 * height_scale) * decorrelate_allpass(hi_l, 311, coeff=0.45)
                bed[:, 9] += (0.20 * height_scale) * decorrelate_allpass(hi_r, 317, coeff=0.45)
                bed[:, 3] += 0.34 * _lfe_lowpass(mid, sample_rate=sr)
        else:
            mono = (arr[:, 0] * gain_lin).astype(np.float32)
            if float(np.max(np.abs(mono))) < 1e-10:
                return bed
            rear_mono = lowpass_rear_air(mono, sample_rate=sr, cutoff_hz=7500.0)
            hi_mono = highpass_air(mono, sample_rate=sr, cutoff_hz=320.0)
            if is_ambience:
                bed[:, 0] += 0.38 * mono
                bed[:, 1] += 0.38 * mono
                bed[:, 4] += 0.56 * decorrelate_allpass(decorrelate_allpass(mono, 211, coeff=0.50), 313, coeff=0.48)
                bed[:, 5] += 0.56 * decorrelate_allpass(decorrelate_allpass(mono, 223, coeff=0.50), 331, coeff=-0.48)
                bed[:, 6] += (0.54 * rear_scale) * decorrelate_allpass(decorrelate_allpass(rear_mono, 389, coeff=0.52), 503, coeff=-0.50)
                bed[:, 7] += (0.54 * rear_scale) * decorrelate_allpass(decorrelate_allpass(rear_mono, 401, coeff=0.52), 521, coeff=0.50)
                bed[:, 8] += (0.42 * height_scale) * decorrelate_allpass(decorrelate_allpass(hi_mono, 293, coeff=0.48), 431, coeff=0.48)
                bed[:, 9] += (0.42 * height_scale) * decorrelate_allpass(decorrelate_allpass(hi_mono, 307, coeff=0.48), 443, coeff=-0.48)
                bed[:, 10] += (0.38 * height_scale * rear_scale) * decorrelate_allpass(decorrelate_allpass(hi_mono, 467, coeff=0.50), 601, coeff=-0.50)
                bed[:, 11] += (0.38 * height_scale * rear_scale) * decorrelate_allpass(decorrelate_allpass(hi_mono, 479, coeff=0.50), 613, coeff=0.50)
                bed[:, 3] += 0.18 * _lfe_lowpass(mono, sample_rate=sr)
            else:
                bed[:, 0] += 0.52 * mono
                bed[:, 1] += 0.52 * mono
                bed[:, 4] += 0.38 * decorrelate_allpass(mono, 241, coeff=0.45)
                bed[:, 5] += 0.38 * decorrelate_allpass(mono, 251, coeff=0.45)
                bed[:, 6] += (0.28 * rear_scale) * decorrelate_allpass(rear_mono, 409, coeff=0.48)
                bed[:, 7] += (0.28 * rear_scale) * decorrelate_allpass(rear_mono, 419, coeff=0.48)
                bed[:, 8] += (0.22 * height_scale) * decorrelate_allpass(hi_mono, 311, coeff=0.45)
                bed[:, 9] += (0.22 * height_scale) * decorrelate_allpass(hi_mono, 317, coeff=0.45)
                bed[:, 3] += 0.28 * _lfe_lowpass(mono, sample_rate=sr)

        return bed

    def _pan_dynamic_object_714(
        self,
        mono_audio: np.ndarray,
        keyframes: list[dict[str, Any]] | None,
        spread: float = 0.0,
        role: str = "sfx",
        control_step: int = 64,
    ) -> np.ndarray:
        """動的オブジェクト音声を 7.1.4 ミックスダウン (N, 12) へ滑らかに 3D VBAP パンニングする。"""
        x = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        out = np.zeros((n, 12), dtype=np.float32)
        if n == 0 or float(np.max(np.abs(x))) < 1e-10:
            return out

        ctrl_idx = np.arange(0, n, control_step, dtype=np.int64)
        if ctrl_idx[-1] != n - 1:
            ctrl_idx = np.append(ctrl_idx, n - 1)
        ctrl_times = ctrl_idx.astype(np.float64) / float(self.sample_rate)
        az_c, el_c, _ = interpolate_keyframes(keyframes, ctrl_times)

        gains_ctrl = np.empty((ctrl_idx.shape[0], 12), dtype=np.float64)
        for i in range(ctrl_idx.shape[0]):
            gains_ctrl[i] = compute_714_panning_gains(float(az_c[i]), float(el_c[i]), spread=spread)

        sample_idx = np.arange(n, dtype=np.float64)
        ctrl_idx_f = ctrl_idx.astype(np.float64)
        for ch in ATMOS_714_NON_LFE_INDICES:
            if float(np.max(np.abs(gains_ctrl[:, ch]))) < 1e-8:
                continue
            g_ch = np.interp(sample_idx, ctrl_idx_f, gains_ctrl[:, ch])
            out[:, ch] = (x * g_ch).astype(np.float32)

        # SFX / Foley のみ LFE 補助（sfx_impact は 0.28、セリフは LFE に送らない）
        role_norm = str(role or "sfx").strip().lower()
        if role_norm == "sfx_impact":
            out[:, 3] = 0.28 * _lfe_lowpass(x, sample_rate=self.sample_rate)
        elif role_norm in ("sfx", "foley"):
            out[:, 3] = 0.20 * _lfe_lowpass(x, sample_rate=self.sample_rate)

        return out

    def render_scene(
        self,
        scene_dict: dict[str, Any],
        base_dir: str | None = None,
        audio_overrides: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """シーンから Dolby Atmos 7.1.4 Bed + Dynamic Objects + バイノーラル + 7.1.4 ミックスダウンを同時生成する。"""
        room_cfg = scene_dict.get("room") or {}
        rt60_sec = float(room_cfg.get("rt60_sec", 0.45))
        room_size = str(room_cfg.get("size", "medium"))

        objects = scene_dict.get("objects") or []
        loaded_tracks: list[tuple[dict[str, Any], np.ndarray]] = []
        max_len = 0

        for idx, obj in enumerate(objects):
            if obj.get("muted", False) or not obj.get("enabled", True):
                continue
            obj_id = str(obj.get("id", f"obj_{idx}"))
            if audio_overrides and obj_id in audio_overrides:
                audio = np.asarray(audio_overrides[obj_id], dtype=np.float32)
                if audio.ndim == 1:
                    audio = audio[:, np.newaxis]
            else:
                audio_path = obj.get("audio_path")
                if not audio_path:
                    continue
                audio, _ = load_audio_48k(audio_path, base_dir=base_dir, target_sr=self.sample_rate, mono=False)
            max_len = max(max_len, audio.shape[0])
            loaded_tracks.append((obj, audio))

        duration_sec = float(scene_dict.get("duration_sec") or 0.0)
        n_samples = int(round(duration_sec * self.sample_rate)) if duration_sec > 0.0 else max_len
        if n_samples <= 0:
            n_samples = self.sample_rate
        actual_duration = n_samples / float(self.sample_rate)

        bed_714 = np.zeros((n_samples, 12), dtype=np.float32)
        binaural_direct = np.zeros((n_samples, 2), dtype=np.float32)
        reverb_bus_mono = np.zeros((n_samples,), dtype=np.float32)
        dynamic_objects: list[dict[str, Any]] = []
        world_360 = scene_dict.get("world_360") or {}

        for idx, (obj, audio) in enumerate(loaded_tracks):
            audio_fit = fit_length(audio, n_samples)
            is_bed = bool(obj.get("is_bed", False))
            gain_db = resolve_object_gain_db(obj, n_samples, sample_rate=self.sample_rate)
            role = str(obj.get("role", "sfx"))
            spread = float(obj.get("spread", 0.1))

            if is_bed:
                bed_714 += self._route_bed_track_714(
                    audio_fit,
                    gain_db=gain_db,
                    role=role,
                    keyframes=obj.get("keyframes"),
                    world_360=world_360,
                )
                continue

            mono = (
                audio_fit[:, 0]
                if audio_fit.shape[1] == 1
                else np.mean(audio_fit, axis=1, dtype=np.float64).astype(np.float32)
            )
            keyframes = obj.get("keyframes")

            # 1) ダイナミックオブジェクト単体音声（距離減衰＆空気吸収適用済、ADM BWF / 7.1.4 パン用）
            direct_mono, rev_send = _apply_distance_to_object_mono(
                mono,
                keyframes=keyframes,
                gain_db=gain_db,
                rt60_sec=rt60_sec,
                sample_rate=self.sample_rate,
                role=role,
            )
            reverb_bus_mono += rev_send

            # 2) 高精細バイノーラル直接レンダリング（連続 HRTF オーバーラップ加算、スピーカー量子化なし）
            obj_bin, _ = self.hrtf_engine.render_mono_dynamic(
                mono,
                keyframes=keyframes,
                gain_db=gain_db,
                rt60_sec=rt60_sec,
                block_size=2048,
                apply_distance=True,
                role=role,
            )
            binaural_direct += obj_bin

            # 3) Atmos Dynamic Object メタデータ記録
            dynamic_objects.append(
                {
                    "id": str(obj.get("id", f"object_{idx}")),
                    "label": str(obj.get("label", obj.get("id", f"Object {idx + 1}"))),
                    "role": role,
                    "spread": spread,
                    "audio": direct_mono,
                    "keyframes": keyframes,
                    "adm_keyframes": _build_adm_keyframes(keyframes, duration_sec=actual_duration),
                }
            )

        # 7.1.4 拡散室内リバーブを 11ch（LFE除く）ベッドへ加算（映画ダビングステージ方式＋360°重み）
        if float(np.max(np.abs(reverb_bus_mono))) > 1e-8:
            reverb_engine = RoomReverbEngine(
                rt60_sec=rt60_sec,
                room_size=room_size,
                sample_rate=self.sample_rate,
            )
            rev_11ch = reverb_engine.process_multichannel(reverb_bus_mono, num_channels=11)
            w360 = world_360 if isinstance(world_360, dict) else {}
            w360_enabled = w360.get("enabled", True) is not False
            rear_rev_w = max(0.75, min(1.35, float(w360.get("rear_level", 1.0)))) if w360_enabled else 0.75
            height_rev_w = max(0.75, min(1.35, float(w360.get("height_level", 1.0)))) if w360_enabled else 0.75
            rev_weights = []
            for spk_idx in ATMOS_714_NON_LFE_INDICES:
                if spk_idx in (6, 7):
                    rev_weights.append(rear_rev_w)
                elif spk_idx in (8, 9):
                    rev_weights.append(height_rev_w)
                elif spk_idx in (10, 11):
                    rev_weights.append(0.5 * (rear_rev_w + height_rev_w))
                else:
                    rev_weights.append(1.0)
            w_arr = np.asarray(rev_weights, dtype=np.float64)
            rev_norm = math.sqrt(float(np.sum(w_arr * w_arr))) or math.sqrt(11.0)
            for k, spk_idx in enumerate(ATMOS_714_NON_LFE_INDICES):
                bed_714[:, spk_idx] += rev_11ch[:, k] * float(w_arr[k] / rev_norm)

        # バイノーラル統合: 7.1.4 Bed (仮想スピーカーHRTF) + Dynamic Objects (直接連続HRTF)
        binaural_bed = self.render_bed_to_binaural(bed_714)
        binaural = (binaural_bed + binaural_direct).astype(np.float32)

        # 7.1.4 12ch 再生確認用ミックスダウン（Bed + 全 Dynamic Objects の 3D VBAP 畳み込み）
        surround_714 = bed_714.copy()
        for dyn in dynamic_objects:
            surround_714 += self._pan_dynamic_object_714(
                dyn["audio"],
                keyframes=dyn["keyframes"],
                spread=dyn["spread"],
                role=dyn["role"],
            )

        return {
            "bed_714": bed_714,
            "dynamic_objects": dynamic_objects,
            "binaural": binaural,
            "surround_714": surround_714,
            "duration_sec": actual_duration,
            "hrtf_source": self.hrtf_engine.source_name,
        }

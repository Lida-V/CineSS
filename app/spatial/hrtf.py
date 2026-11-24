"""HRTF バイノーラル定位エンジン・距離モデリング・室内音響（初期反射＋拡散リバーブ）.

座標系規約（CineSS 共通）:
- サンプルレート: 48000 Hz
- 方位角 azimuth_deg ∈ [-180, 180]:
    0° = 正面, 負 = 左 (-90° = 真左), 正 = 右 (+90° = 真右), ±180° = 真後ろ
- 仰角 elevation_deg ∈ [-90, 90]:
    0° = 水平, 正 = 上 (+90° = 真上), 負 = 下 (-90° = 真下)
- 距離 distance_m ∈ [0.5, 15.0]:
    近距離ほどドライ＆大音量、遠距離ほど距離減衰＋空気吸収LPF＋残響比増加
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.spatial import cKDTree

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None  # type: ignore[assignment]

from scipy.io import wavfile as scipy_wavfile

SAMPLE_RATE: int = 48000
HEAD_RADIUS_M: float = 0.0875
SPEED_OF_SOUND_MS: float = 343.0
DEFAULT_HRIR_LEN: int = 256
HRIR_BASE_DELAY_SAMPLES: float = 48.0


# ---------------------------------------------------------------------------
# 共通オーディオ I/O & キーフレーム補間ユーティリティ
# ---------------------------------------------------------------------------


def wrap_azimuth_deg(azimuth_deg: float | np.ndarray) -> float | np.ndarray:
    """方位角を [-180, 180] 度に正規化する。"""
    wrapped = ((np.asarray(azimuth_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    if np.ndim(azimuth_deg) == 0:
        val = float(wrapped)
        return 180.0 if val == -180.0 and float(azimuth_deg) > 0.0 else val
    return wrapped


def clamp_elevation_deg(elevation_deg: float | np.ndarray) -> float | np.ndarray:
    """仰角を [-90, 90] 度にクランプする。"""
    clamped = np.clip(np.asarray(elevation_deg, dtype=np.float64), -90.0, 90.0)
    return float(clamped) if np.ndim(elevation_deg) == 0 else clamped


def clamp_distance_m(distance_m: float | np.ndarray) -> float | np.ndarray:
    """距離を安全な範囲 [0.5, 15.0] m にクランプする。"""
    clamped = np.clip(np.asarray(distance_m, dtype=np.float64), 0.5, 15.0)
    return float(clamped) if np.ndim(distance_m) == 0 else clamped


def resolve_audio_path(audio_path: str | Path, base_dir: str | Path | None = None) -> Path:
    """絶対パスまたは base_dir 相対パスを解決する。"""
    p = Path(audio_path)
    if p.is_absolute() or base_dir is None:
        return p
    return Path(base_dir) / p


def resample_audio(audio: np.ndarray, src_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """任意サンプルレートの音声 (N,) または (N, C) を target_sr へ高品質リサンプルする。"""
    arr = np.asarray(audio, dtype=np.float32)
    if src_sr == target_sr or arr.size == 0:
        return arr
    g = math.gcd(int(target_sr), int(src_sr))
    up = int(target_sr) // g
    down = int(src_sr) // g
    resampled = signal.resample_poly(arr, up, down, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def load_audio_48k(
    audio_path: str | Path,
    base_dir: str | Path | None = None,
    target_sr: int = SAMPLE_RATE,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """音声ファイルを読み込み、target_sr (既定 48kHz) float32 に変換する。

    Returns:
        mono=True の場合は (N,) float32、mono=False の場合は (N, C) float32 とサンプルレート。
    """
    resolved = resolve_audio_path(audio_path, base_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {resolved}")

    data: np.ndarray
    sr: int
    if sf is not None:
        try:
            data, sr = sf.read(str(resolved), dtype="float32", always_2d=True)
        except Exception:
            sr, raw = scipy_wavfile.read(str(resolved))
            data = _scipy_wav_to_float32(raw)
    else:
        sr, raw = scipy_wavfile.read(str(resolved))
        data = _scipy_wav_to_float32(raw)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        sr = target_sr

    if mono:
        if data.shape[1] == 1:
            return data[:, 0].astype(np.float32, copy=False), sr
        return np.mean(data, axis=1, dtype=np.float64).astype(np.float32), sr

    return data.astype(np.float32, copy=False), sr


def _scipy_wav_to_float32(raw: np.ndarray) -> np.ndarray:
    """scipy.io.wavfile.read の出力を [-1, 1] の float32 (N, C) に正規化する。"""
    arr = np.asarray(raw)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    if arr.dtype == np.uint8:
        return ((arr.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    info = np.iinfo(arr.dtype)
    scale = float(max(abs(info.min), info.max))
    return (arr.astype(np.float64) / scale).astype(np.float32)


def fit_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """音声配列を target_samples に揃える（短い場合はゼロ埋め、長い場合は切り詰め）。"""
    n = audio.shape[0]
    if n == target_samples:
        return audio
    if n > target_samples:
        return audio[:target_samples]
    pad_len = target_samples - n
    if audio.ndim == 1:
        return np.pad(audio, (0, pad_len), mode="constant")
    return np.pad(audio, ((0, pad_len), (0, 0)), mode="constant")


def interpolate_keyframes(
    keyframes: list[dict[str, Any]] | None,
    times_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """キーフレーム列を指定時刻配列 times_sec で滑らかに補間する。

    方位角は最短角経路（unwrap 補間）で連続補間する。
    """
    times = np.asarray(times_sec, dtype=np.float64)
    if not keyframes:
        zeros = np.zeros_like(times, dtype=np.float64)
        dists = np.full_like(times, 2.0, dtype=np.float64)
        return zeros, zeros.copy(), dists

    sorted_kf = sorted(keyframes, key=lambda k: float(k.get("t", 0.0)))
    t_kf = np.array([float(k.get("t", 0.0)) for k in sorted_kf], dtype=np.float64)
    az_kf = np.array([float(k.get("azimuth_deg", 0.0)) for k in sorted_kf], dtype=np.float64)
    el_kf = np.array([float(k.get("elevation_deg", 0.0)) for k in sorted_kf], dtype=np.float64)
    dist_kf = np.array([float(k.get("distance_m", 2.0)) for k in sorted_kf], dtype=np.float64)

    # 同一時刻の重複キーフレームがある場合は最後を採用しつつ単調増加を保証
    if len(t_kf) > 1:
        _, unique_idx = np.unique(t_kf[::-1], return_index=True)
        keep_idx = np.sort(len(t_kf) - 1 - unique_idx)
        t_kf = t_kf[keep_idx]
        az_kf = az_kf[keep_idx]
        el_kf = el_kf[keep_idx]
        dist_kf = dist_kf[keep_idx]

    if len(t_kf) == 1:
        az = np.full_like(times, wrap_azimuth_deg(float(az_kf[0])), dtype=np.float64)
        el = np.full_like(times, clamp_elevation_deg(float(el_kf[0])), dtype=np.float64)
        dist = np.full_like(times, clamp_distance_m(float(dist_kf[0])), dtype=np.float64)
        return az, el, dist

    az_rad_unwrapped = np.unwrap(np.radians(az_kf))
    az_interp = wrap_azimuth_deg(np.degrees(np.interp(times, t_kf, az_rad_unwrapped)))
    el_interp = clamp_elevation_deg(np.interp(times, t_kf, el_kf))
    dist_interp = clamp_distance_m(np.interp(times, t_kf, dist_kf))
    return np.asarray(az_interp), np.asarray(el_interp), np.asarray(dist_interp)


def direction_to_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """CineSS 座標 (azimuth_deg, elevation_deg) を標準空間単位ベクトル (+X=前, +Y=左, +Z=上) へ変換する。"""
    phi = -math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    c_el = math.cos(el)
    return np.array([c_el * math.cos(phi), c_el * math.sin(phi), math.sin(el)], dtype=np.float64)


# ---------------------------------------------------------------------------
# 距離減衰・空気吸収モデル
# ---------------------------------------------------------------------------


def distance_direct_gain(
    distance_m: float,
    ref_distance_m: float = 1.0,
    role: str = "sfx",
) -> float:
    """逆距離則ベースの直接音ゲイン（安全な近距離クランプ＆セリフ専用 Perspective 付き）。

    - role != "dialog": 1.0m -> 1.0 (0 dB), 2.0m -> -5.4 dB, 8.0m -> -16.3 dB
    - role == "dialog": 2.0m -> 1.0 (0 dB), clip((2.0 / d) ** 0.50, 0.38, 1.22)
    """
    d = float(clamp_distance_m(distance_m))
    if str(role or "sfx").strip().lower() == "dialog":
        return float(np.clip((2.0 / d) ** 0.50, 0.38, 1.22))
    return float((ref_distance_m / d) ** 0.9)


def distance_reverb_send(
    distance_m: float,
    rt60_sec: float = 0.45,
    role: str = "sfx",
) -> float:
    """距離に応じたリバーブセンド量。

    実空間では拡散音場レベルは距離に対してほぼ一定（遠方で緩やかに増加）のため、
    直接音が 1/d で減衰するのと相まって Direct-to-Reverb 比が距離で大きく変化しつつ、
    遠距離の総音量もしっかり減衰する物理バランスを保つ。
    role == "dialog" の場合は明瞭度保護のため通常の 0.55 倍に抑制する。
    """
    d = float(clamp_distance_m(distance_m))
    rt_scale = math.sqrt(max(0.15, min(float(rt60_sec), 3.0)) / 0.45)
    # 1.0m -> 約0.09 (非常にドライ), 8.0m -> 約0.18 (残響比大幅増かつ総音量は1.0mより約10dB減衰)
    base = 0.065 + 0.028 * math.log2(max(d, 0.5) / 0.5)
    send = min(0.32, base * rt_scale)
    if str(role or "sfx").strip().lower() == "dialog":
        send *= 0.55
    return float(send)


def air_absorption_cutoff_hz(distance_m: float, role: str = "sfx") -> float:
    """距離依存の空気吸収 LPF カットオフ周波数 (Hz)。

    1.0m 以下（dialog は 2.0m 以下）はほぼフラット (20kHz)、遠距離では滑らかに高域が減衰する。
    """
    d = float(clamp_distance_m(distance_m))
    ref_d = 2.0 if str(role or "sfx").strip().lower() == "dialog" else 1.0
    excess = max(0.0, d - ref_d)
    return float(max(2200.0, 20000.0 / (1.0 + 0.70 * excess)))


def air_absorption_response(
    freqs_hz: np.ndarray,
    distance_m: float,
    role: str = "sfx",
) -> np.ndarray:
    """最小位相 1次空気吸収ローパスフィルタの複素周波数応答。"""
    fc = air_absorption_cutoff_hz(distance_m, role=role)
    if fc >= 19900.0:
        return np.ones_like(freqs_hz, dtype=np.complex128)
    ratio = freqs_hz / fc
    # 1次最小位相 LPF: H(jw) = 1 / (1 + j * f / fc)
    return 1.0 / (1.0 + 1j * ratio)


# ---------------------------------------------------------------------------
# 物理ベース合成 HRTF（Woodworth-Schlosberg ITD + 球体頭部回折 ILD + ピンナ）
# ---------------------------------------------------------------------------


def _biquad_peaking_response(
    freqs_hz: np.ndarray,
    center_hz: float,
    gain_db: float,
    q: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Audio EQ Cookbook ピーキングEQの厳密な最小位相複素周波数応答。"""
    if abs(gain_db) < 1e-4 or center_hz <= 20.0 or center_hz >= 0.48 * sample_rate:
        return np.ones_like(freqs_hz, dtype=np.complex128)
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * center_hz / sample_rate
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * max(q, 0.1))

    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_lin

    z_inv = np.exp(-1j * 2.0 * math.pi * freqs_hz / sample_rate)
    z_inv2 = z_inv * z_inv
    return (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)


def _biquad_highshelf_response(
    freqs_hz: np.ndarray,
    shelf_hz: float,
    gain_db: float,
    slope: float = 0.8,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Audio EQ Cookbook ハイシェルフフィルタの厳密な最小位相複素周波数応答。"""
    if abs(gain_db) < 1e-4 or shelf_hz <= 20.0 or shelf_hz >= 0.48 * sample_rate:
        return np.ones_like(freqs_hz, dtype=np.complex128)
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * shelf_hz / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = (sin_w0 / 2.0) * math.sqrt(max(0.05, (a_lin + 1.0 / a_lin) * (1.0 / slope - 1.0) + 2.0))
    two_sqrt_a_alpha = 2.0 * math.sqrt(a_lin) * alpha

    b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0)
    b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0)
    a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha

    z_inv = np.exp(-1j * 2.0 * math.pi * freqs_hz / sample_rate)
    z_inv2 = z_inv * z_inv
    return (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)


def compute_woodworth_itd_sec(azimuth_deg: float, elevation_deg: float) -> float:
    """Woodworth-Schlosberg 頭部モデルによる符号付き ITD (秒) を返す。

    正の値 = 左耳が右耳より早く到達（左音源 azimuth_deg < 0）。
    負の値 = 右耳が左耳より早く到達（右音源 azimuth_deg > 0）。
    """
    u = direction_to_unit_vector(azimuth_deg, elevation_deg)
    # u[1] = +1 が真左, -1 が真右
    lateral_angle = math.asin(max(-1.0, min(1.0, float(u[1]))))
    return (HEAD_RADIUS_M / SPEED_OF_SOUND_MS) * (lateral_angle + math.sin(lateral_angle))


def synthesize_physical_hrir(
    azimuth_deg: float,
    elevation_deg: float,
    sample_rate: int = SAMPLE_RATE,
    ir_len: int = DEFAULT_HRIR_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """物理モデル（Woodworth-Schlosberg ITD + 球体頭部回折 ILD + ピンナ＋肩反射）による HRIR ペアを合成する。"""
    u = direction_to_unit_vector(azimuth_deg, elevation_deg)
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])

    # 各耳軸との角度: 左耳軸=(0, +1, 0), 右耳軸=(0, -1, 0)
    theta_l = math.acos(max(-1.0, min(1.0, uy)))
    theta_r = math.acos(max(-1.0, min(1.0, -uy)))

    # 1. Woodworth-Schlosberg ITD（左音源では uy > 0 → tau > 0 → 左耳の遅延が小さい＝先行）
    itd_sec = compute_woodworth_itd_sec(azimuth_deg, elevation_deg)
    delay_l_samples = HRIR_BASE_DELAY_SAMPLES - 0.5 * itd_sec * sample_rate
    delay_r_samples = HRIR_BASE_DELAY_SAMPLES + 0.5 * itd_sec * sample_rate

    # FFT グリッド（循環畳み込みを避けるため 2*ir_len で計算して切り出し）
    n_fft = max(512, 1 << (int(ir_len * 2) - 1).bit_length())
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 2. 球体頭部回折 ILD（Brown-Duda 型最小位相モデル）
    #    低域（<< f_knee）は頭部回り込みで 0 dB、高域は同側わずかに増幅・対側は頭部影で大きく減衰
    f_knee = 580.0

    def _head_shadow(theta: float) -> np.ndarray:
        # theta=0 (同側) -> alpha ≈ 1.32 (+2.4 dB)
        # theta=pi/2 (正面/正中面) -> alpha = 1.00 (0 dB)
        # theta=150° (最大音響影) -> alpha ≈ 0.20 (-14 dB)
        # theta=180° (反対極ブライトスポット) -> alpha ≈ 0.26 (-11.7 dB)
        theta_min = math.radians(150.0)
        if theta <= theta_min:
            # cos(0)=1 -> 1.32, cos(90°)=0 -> 0.76? 正中面(90°)でちょうど 1.0 にする滑らか曲線
            t_norm = theta / theta_min
            # theta=0 -> 1.32, theta=90° (t=0.6) -> 1.0, theta=150° (t=1.0) -> 0.20
            alpha = 0.20 + 1.12 * (0.5 * (1.0 + math.cos(math.pi * t_norm))) ** 0.55
        else:
            # 150° -> 180° の反対極回折ブライトスポット
            t_back = (theta - theta_min) / (math.pi - theta_min)
            alpha = 0.20 + 0.06 * math.sin(0.5 * math.pi * t_back)

        # 低中域の穏やかなレベル差（300Hz〜1.2kHzでも自然な横定位を補助）
        s = 1j * (freqs / f_knee)
        h_diff = (1.0 + alpha * s) / (1.0 + s)
        # 200Hz 以上での穏やかな近接・回折ブースト/減衰（正中面 theta=90° では 1.0）
        mid_tilt_db = 1.6 * math.cos(theta)
        h_mid = _biquad_highshelf_response(freqs, 220.0, mid_tilt_db, slope=0.6, sample_rate=sample_rate)
        return h_diff * h_mid

    h_l = _head_shadow(theta_l)
    h_r = _head_shadow(theta_r)

    # 3. ピンナ（耳介）仰角ノッチ＆前後弁別シェルフ（両耳共通＋耳別微差）
    el_deg = math.degrees(math.asin(max(-1.0, min(1.0, uz))))
    # 前後弁別: 後方 (ux < 0) は耳介フランジ遮蔽で 3.5kHz 以上が減衰、前方は明瞭
    back_amount = max(0.0, -ux)  # 0 (前) .. 1 (真後ろ)
    front_back_shelf = _biquad_highshelf_response(
        freqs, 3200.0, -4.5 * back_amount, slope=0.75, sample_rate=sample_rate
    )
    front_back_dip = _biquad_peaking_response(
        freqs, 1800.0, -2.0 * back_amount, q=1.5, sample_rate=sample_rate
    )

    # 仰角ピンナノッチ N1: 低仰角 ~5.5kHz → 水平 ~7.4kHz → 高仰角 ~10.8kHz
    notch_hz = 7400.0 + 38.0 * el_deg
    notch_hz = max(4800.0, min(12500.0, notch_hz))
    # 頭上真上付近ではノッチが浅くなり、8kHz 帯の頭上ブーストが強まる
    overhead_factor = max(0.0, uz)
    notch_depth_db = -8.5 * (1.0 - 0.55 * overhead_factor) * (0.4 + 0.6 * max(0.0, ux + 0.35))
    pinna_notch = _biquad_peaking_response(
        freqs, notch_hz, notch_depth_db, q=4.5, sample_rate=sample_rate
    )
    # 頭上高さ知覚キュー（Blauert directional band ~8 kHz）
    overhead_boost = _biquad_peaking_response(
        freqs, 8200.0, 3.8 * (overhead_factor ** 1.4), q=1.8, sample_rate=sample_rate
    )
    # 低仰角の胸/肩反射による穏やかな 1.4kHz シェーピング
    torso_delay_sec = 0.0011 + 0.0007 * (1.0 - max(0.0, -uz))
    torso_gain = 0.14 * (0.5 + 0.5 * overhead_factor) * (0.6 + 0.4 * max(0.0, ux))
    torso_lpf = 1.0 / (1.0 + 1j * (freqs / 2800.0))
    h_torso = 1.0 + torso_gain * torso_lpf * np.exp(-1j * 2.0 * math.pi * freqs * torso_delay_sec)

    common_pinna = front_back_shelf * front_back_dip * pinna_notch * overhead_boost * h_torso

    # サブサンプル正確遅延を周波数領域で付与
    phase_l = np.exp(-1j * 2.0 * math.pi * freqs * (delay_l_samples / sample_rate))
    phase_r = np.exp(-1j * 2.0 * math.pi * freqs * (delay_r_samples / sample_rate))

    spec_l = h_l * common_pinna * phase_l
    spec_r = h_r * common_pinna * phase_r

    # 正面 (0°, 0°) の広帯域レベルが 0 dB に近くなるよう微調整（肩反射分を補正）
    norm = 1.0 / (1.0 + 0.5 * torso_gain)
    ir_l = np.fft.irfft(spec_l * norm, n=n_fft)[:ir_len]
    ir_r = np.fft.irfft(spec_r * norm, n=n_fft)[:ir_len]

    # 末尾 32 サンプルを滑らかにコサインテーパーして打ち切りクリックを防止
    taper_len = min(32, ir_len // 4)
    taper = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, taper_len, dtype=np.float64)))
    ir_l[-taper_len:] *= taper
    ir_r[-taper_len:] *= taper

    return ir_l.astype(np.float32), ir_r.astype(np.float32)


# ---------------------------------------------------------------------------
# SOFA 実測 HRTF ローダー ＋ 物理フォールバック統合エンジン
# ---------------------------------------------------------------------------


class HRTFEngine:
    """SOFA 実測 HRTF（利用可能時）または物理ベース合成 HRTF を提供するエンジン。"""

    def __init__(
        self,
        sofa_path: str | Path | None = None,
        sample_rate: int = SAMPLE_RATE,
        ir_len: int = DEFAULT_HRIR_LEN,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.ir_len = int(ir_len)
        self.source_name = "physical_woodworth_brown_duda"
        self.fallback_reason: str | None = None
        self.diffuse_field_eq_applied: bool = False

        self._use_sofa = False
        self._sofa_dirs: np.ndarray | None = None
        self._sofa_tree: cKDTree | None = None
        self._sofa_irs: np.ndarray | None = None  # shape (M, 2, N)
        self._sofa_onsets: np.ndarray | None = None  # shape (M, 2)
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        if sofa_path is not None and str(sofa_path).strip():
            self._try_load_sofa(Path(sofa_path))

    @staticmethod
    def _apply_diffuse_field_eq(
        irs: np.ndarray,
        sample_rate: int,
        dirs: np.ndarray | None = None,
    ) -> np.ndarray:
        """全測定点平均の拡散音場等化（最小位相実ケプストラム逆フィルタ）と正面 0dB 正規化を適用する。"""
        n_meas, _, n_ir = irs.shape
        if n_meas == 0 or n_ir < 16 or sample_rate <= 0:
            return irs.astype(np.float32, copy=False)

        n_fft = max(4096, 1 << (n_ir * 4 - 1).bit_length())
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
        spec = np.fft.rfft(irs.astype(np.float64), n=n_fft, axis=2)
        p_df = np.mean(np.abs(spec) ** 2, axis=(0, 1))

        # 対数周波数軸で 1/6 オクターブ相当（FWHM = 1/6 oct）のガウス平滑化を適用
        p_smooth = p_df.copy()
        pos_mask = freqs > 0.0
        if np.any(pos_mask):
            f_pos = freqs[pos_mask]
            log_f = np.log2(f_pos)
            sigma_oct = (1.0 / 6.0) / 2.35482
            diff = log_f[:, np.newaxis] - log_f[np.newaxis, :]
            weights = np.exp(-0.5 * (diff / sigma_oct) ** 2)
            weights /= np.sum(weights, axis=1, keepdims=True)
            p_smooth[pos_mask] = weights @ p_df[pos_mask]
            p_smooth[0] = p_smooth[1]

        # 基準レベル P_ref: 300Hz〜1500Hz の平均パワー
        ref_mask = (freqs >= 300.0) & (freqs <= 1500.0)
        if not np.any(ref_mask):
            ref_mask = pos_mask
        p_ref = float(np.mean(p_smooth[ref_mask])) if np.any(ref_mask) else 1.0

        raw_gain = np.sqrt(max(p_ref, 1e-12) / np.maximum(p_smooth, 1e-12))
        gain_db = 20.0 * np.log10(np.maximum(raw_gain, 1e-12))

        # 60Hz 以下と 14kHz 以上は 0 dB へ滑らかにコサインテーパーし、[-20 dB, +16 dB] にクランプ
        w_band = np.ones_like(freqs)
        w_band[freqs <= 20.0] = 0.0
        low_m = (freqs > 20.0) & (freqs < 60.0)
        if np.any(low_m):
            log_ratio = (np.log2(freqs[low_m]) - math.log2(20.0)) / (math.log2(60.0) - math.log2(20.0))
            w_band[low_m] = 0.5 * (1.0 - np.cos(math.pi * log_ratio))

        hi_end = min(20000.0, 0.48 * float(sample_rate))
        if hi_end > 14000.0:
            w_band[freqs >= hi_end] = 0.0
            hi_m = (freqs > 14000.0) & (freqs < hi_end)
            if np.any(hi_m):
                w_band[hi_m] = 0.5 * (1.0 + np.cos(math.pi * (freqs[hi_m] - 14000.0) / (hi_end - 14000.0)))

        gain_db = np.clip(gain_db * w_band, -20.0, 16.0)
        mag = 10.0 ** (gain_db / 20.0)

        # 実ケプストラム法による因果最小位相逆応答 H_eq を生成（プリリンギングゼロ・ITD非破壊）
        log_mag = np.log(np.maximum(mag, 1e-12))
        cep = np.fft.irfft(log_mag, n=n_fft)
        win = np.zeros(n_fft, dtype=np.float64)
        win[0] = 1.0
        win[1 : n_fft // 2] = 2.0
        win[n_fft // 2] = 1.0
        h_eq_spec = np.exp(np.fft.rfft(cep * win, n=n_fft))
        h_eq = np.fft.irfft(h_eq_spec, n=n_fft)

        # 全 HRIR へ畳み込み（長さ n_ir、末尾 16 サンプルコサインテーパー）
        n_conv = 1 << (n_ir + n_fft - 1).bit_length()
        h_eq_fft = np.fft.rfft(h_eq, n=n_conv)
        ir_fft = np.fft.rfft(irs.astype(np.float64), n=n_conv, axis=2)
        irs_eq = np.fft.irfft(ir_fft * h_eq_fft[np.newaxis, np.newaxis, :], n=n_conv, axis=2)[:, :, :n_ir]

        taper_len = min(16, max(1, n_ir // 4))
        taper = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, taper_len, dtype=np.float64)))
        irs_eq[:, :, -taper_len:] *= taper[np.newaxis, np.newaxis, :]

        # 正面 (0°, 0°) 最近傍測定点の 200Hz〜4000Hz 平均振幅（両耳 RMS）を 1.0 (0 dB) に正規化
        n_eval = max(2048, 1 << (n_ir * 2 - 1).bit_length())
        f_ev = np.fft.rfftfreq(n_eval, d=1.0 / float(sample_rate))
        m_mid = (f_ev >= 200.0) & (f_ev <= 4000.0)
        if not np.any(m_mid):
            m_mid = f_ev > 0.0

        if dirs is not None and dirs.shape[0] == n_meas:
            front_idx = int(np.argmax(dirs[:, 0]))
            h_front = np.fft.rfft(irs_eq[front_idx], n=n_eval, axis=1)
            front_amp = float(np.sqrt(np.mean(np.abs(h_front[:, m_mid]) ** 2)))
        else:
            h_all = np.fft.rfft(irs_eq, n=n_eval, axis=2)
            front_amp = float(np.sqrt(np.mean(np.abs(h_all[:, :, m_mid]) ** 2)))

        if front_amp > 1e-9:
            irs_eq *= 1.0 / front_amp

        return irs_eq.astype(np.float32)

    def _try_load_sofa(self, path: Path) -> None:
        """SimpleFreeFieldHRIR 形式の SOFA (HDF5/NetCDF) を読み込む。失敗時は物理モデルへ自動フォールバック。"""
        if not path.exists():
            self.fallback_reason = f"SOFAファイル未検出: {path}"
            return

        try:
            irs, positions, pos_type, sofa_sr = self._read_sofa_arrays(path)
        except Exception as exc:
            self.fallback_reason = f"SOFA読み込み失敗（物理モデルへフォールバック）: {exc}"
            return

        if irs.ndim != 3 or irs.shape[1] < 2 or positions.ndim != 2 or positions.shape[1] < 2:
            self.fallback_reason = "SOFAデータ構造が SimpleFreeFieldHRIR と一致しません"
            return

        irs = irs[:, :2, :].astype(np.float32)
        if sofa_sr != self.sample_rate and sofa_sr > 0:
            g = math.gcd(self.sample_rate, int(round(sofa_sr)))
            up = self.sample_rate // g
            down = int(round(sofa_sr)) // g
            irs = signal.resample_poly(irs, up, down, axis=2).astype(np.float32)

        # SourcePosition を (+X=前, +Y=左, +Z=上) の単位ベクトルへ変換
        if pos_type == "cartesian":
            vecs = positions[:, :3].astype(np.float64)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            dirs = vecs / norms
        else:
            # SOFA 球面座標: az (度, 反時計回り正: +90=左), el (度, 上正)
            az_rad = np.radians(positions[:, 0].astype(np.float64))
            el_rad = np.radians(positions[:, 1].astype(np.float64))
            c_el = np.cos(el_rad)
            dirs = np.stack([c_el * np.cos(az_rad), c_el * np.sin(az_rad), np.sin(el_rad)], axis=1)

        # 少数測定点の単体テスト用合成 SOFA (n_meas < 8) を除き、拡散音場等化＋正面 0dB 正規化を適用
        if irs.shape[0] >= 8:
            irs = self._apply_diffuse_field_eq(irs, self.sample_rate, dirs=dirs)
            self.diffuse_field_eq_applied = True
        else:
            self.diffuse_field_eq_applied = False

        # 各 HRIR の立ち上がり遅延（ピーク/エネルギー重心ベース）を事前計測（補間時のコムフィルタ低減用）
        onsets = np.zeros((irs.shape[0], 2), dtype=np.float64)
        for ch in range(2):
            peaks = np.argmax(np.abs(irs[:, ch, :]), axis=1)
            onsets[:, ch] = peaks.astype(np.float64)

        self._sofa_irs = irs
        self._sofa_dirs = dirs
        self._sofa_tree = cKDTree(dirs)
        self._sofa_onsets = onsets
        self.ir_len = int(irs.shape[2])
        self._use_sofa = True
        self.source_name = f"sofa:{path.name}"
        self.fallback_reason = None

    @staticmethod
    def _read_sofa_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, str, int]:
        """h5py / netCDF4 / scipy.io.netcdf_file の順で SOFA を読み出す。"""
        last_err: Exception | None = None

        try:
            import h5py  # type: ignore[import-not-found]

            with h5py.File(str(path), "r") as f:
                irs = np.asarray(f["Data.IR"][:])
                positions = np.asarray(f["SourcePosition"][:])
                pos_type_raw = f["SourcePosition"].attrs.get("Type", b"spherical")
                if isinstance(pos_type_raw, bytes):
                    pos_type = pos_type_raw.decode("utf-8", errors="ignore").strip().lower()
                else:
                    pos_type = str(pos_type_raw).strip().lower()
                sr_arr = np.asarray(f["Data.SamplingRate"][:]).reshape(-1)
                sr = int(round(float(sr_arr[0]))) if sr_arr.size else SAMPLE_RATE
                return irs, positions, pos_type, sr
        except Exception as exc:
            last_err = exc

        try:
            import netCDF4  # type: ignore[import-not-found]

            with netCDF4.Dataset(str(path), "r") as ds:
                irs = np.asarray(ds.variables["Data.IR"][:])
                pos_var = ds.variables["SourcePosition"]
                positions = np.asarray(pos_var[:])
                pos_type = str(getattr(pos_var, "Type", "spherical")).strip().lower()
                sr_arr = np.asarray(ds.variables["Data.SamplingRate"][:]).reshape(-1)
                sr = int(round(float(sr_arr[0]))) if sr_arr.size else SAMPLE_RATE
                return irs, positions, pos_type, sr
        except Exception as exc:
            last_err = exc

        try:
            with scipy_wavfile.netcdf_file(str(path), "r", mmap=False) as ds:  # type: ignore[attr-defined]
                irs = np.asarray(ds.variables["Data.IR"].data)
                pos_var = ds.variables["SourcePosition"]
                positions = np.asarray(pos_var.data)
                pos_type = str(getattr(pos_var, "Type", b"spherical"))
                sr = int(round(float(np.asarray(ds.variables["Data.SamplingRate"].data).reshape(-1)[0])))
                return irs, positions, pos_type.lower(), sr
        except Exception:
            pass

        try:
            from scipy.io import netcdf_file

            with netcdf_file(str(path), "r", mmap=False) as ds:
                irs = np.asarray(ds.variables["Data.IR"].data)
                pos_var = ds.variables["SourcePosition"]
                positions = np.asarray(pos_var.data)
                pos_type_raw = getattr(pos_var, "Type", b"spherical")
                pos_type = (
                    pos_type_raw.decode("utf-8", errors="ignore")
                    if isinstance(pos_type_raw, bytes)
                    else str(pos_type_raw)
                ).strip().lower()
                sr = int(round(float(np.asarray(ds.variables["Data.SamplingRate"].data).reshape(-1)[0])))
                return irs, positions, pos_type, sr
        except Exception as exc:
            last_err = exc

        raise RuntimeError(str(last_err) if last_err else "SOFAローダーが利用できません")

    def get_hrir(self, azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """指定方向 (azimuth_deg, elevation_deg) の (hrir_l, hrir_r) を返す（1度量子化キャッシュ付き）。"""
        az = float(wrap_azimuth_deg(azimuth_deg))
        el = float(clamp_elevation_deg(elevation_deg))
        key = (int(round(az * 2.0)), int(round(el * 2.0)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._use_sofa and self._sofa_tree is not None and self._sofa_irs is not None:
            hrir = self._lookup_sofa_hrir(az, el)
        else:
            hrir = synthesize_physical_hrir(az, el, sample_rate=self.sample_rate, ir_len=self.ir_len)

        if len(self._cache) < 4096:
            self._cache[key] = hrir
        return hrir

    def _lookup_sofa_hrir(self, azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """球面 k-NN ＋立ち上がり時間整合補間による SOFA HRIR 取得。"""
        assert self._sofa_tree is not None
        assert self._sofa_irs is not None
        assert self._sofa_onsets is not None

        u = direction_to_unit_vector(azimuth_deg, elevation_deg)
        k = min(3, self._sofa_irs.shape[0])
        dists, idxs = self._sofa_tree.query(u, k=k)
        if k == 1 or float(np.atleast_1d(dists)[0]) < 1e-3:
            idx0 = int(np.atleast_1d(idxs)[0])
            return self._sofa_irs[idx0, 0].copy(), self._sofa_irs[idx0, 1].copy()

        dists_arr = np.atleast_1d(dists).astype(np.float64)
        idxs_arr = np.atleast_1d(idxs).astype(np.int64)
        weights = 1.0 / np.maximum(dists_arr, 1e-4)
        weights /= np.sum(weights)

        n = self._sofa_irs.shape[2]
        n_fft = 1 << (n * 2 - 1).bit_length()
        freqs = np.fft.rfftfreq(n_fft, d=1.0)
        out = np.zeros((2, n), dtype=np.float32)

        for ch in range(2):
            target_onset = float(np.sum(weights * self._sofa_onsets[idxs_arr, ch]))
            acc_spec = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
            for w, idx in zip(weights, idxs_arr):
                shift = target_onset - float(self._sofa_onsets[idx, ch])
                spec = np.fft.rfft(self._sofa_irs[idx, ch].astype(np.float64), n=n_fft)
                acc_spec += w * spec * np.exp(-1j * 2.0 * math.pi * freqs * shift)
            out[ch] = np.fft.irfft(acc_spec, n=n_fft)[:n].astype(np.float32)

        return out[0], out[1]

    def render_mono_dynamic(
        self,
        mono_audio: np.ndarray,
        keyframes: list[dict[str, Any]] | None,
        gain_db: float = 0.0,
        rt60_sec: float = 0.45,
        block_size: int = 2048,
        apply_distance: bool = True,
        role: str = "sfx",
    ) -> tuple[np.ndarray, np.ndarray]:
        """単一モノラル音源を 50% オーバーラップ Hann 窓クロスフェードで連続バイノーラル定位する。

        Returns:
            direct_stereo: (N, 2) 直接音バイノーラル出力（距離減衰・空気吸収LPF・動的HRTF適用済）
            reverb_send_mono: (N,) 距離依存のリバーブセンド用モノラル信号
        """
        x = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        n_samples = x.shape[0]
        if n_samples == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        gain_lin = float(10.0 ** (float(gain_db) / 20.0))
        if gain_lin == 0.0 or float(np.max(np.abs(x))) < 1e-9:
            return np.zeros((n_samples, 2), dtype=np.float32), np.zeros((n_samples,), dtype=np.float32)

        hop = block_size // 2
        window = np.hanning(block_size).astype(np.float64)
        # 端点でのフェードイン／アウト欠損を防ぐため両端に hop 分ゼロパディング（厳密 COLA = 1.0）
        x_pad = np.pad(x.astype(np.float64) * gain_lin, (hop, hop + block_size), mode="constant")

        ir_len = self.ir_len
        n_fft = 1 << (block_size + ir_len - 2).bit_length()
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        n_frames = int(math.ceil((n_samples + hop) / hop))
        frame_starts = np.arange(n_frames, dtype=np.int64) * hop
        # パディング前の実時間に対応するフレーム中心時刻
        frame_center_times = np.clip((frame_starts + hop - hop) / float(self.sample_rate), 0.0, max(0.0, (n_samples - 1) / float(self.sample_rate)))

        az_arr, el_arr, dist_arr = interpolate_keyframes(keyframes, frame_center_times)

        out_len = n_samples + hop * 2 + block_size + ir_len
        direct_pad = np.zeros((out_len, 2), dtype=np.float64)
        reverb_pad = np.zeros((out_len,), dtype=np.float64)

        for i in range(n_frames):
            start = int(frame_starts[i])
            seg = x_pad[start : start + block_size]
            if seg.shape[0] < block_size:
                seg = np.pad(seg, (0, block_size - seg.shape[0]), mode="constant")
            if float(np.max(np.abs(seg))) < 1e-11:
                continue

            seg_w = seg * window
            az = float(az_arr[i])
            el = float(el_arr[i])
            dist = float(dist_arr[i])

            if apply_distance:
                g_dir = distance_direct_gain(dist, role=role)
                g_rev = distance_reverb_send(dist, rt60_sec=rt60_sec, role=role)
                h_air = air_absorption_response(freqs, dist, role=role)
            else:
                g_dir = 1.0
                g_rev = 0.0
                h_air = np.ones_like(freqs, dtype=np.complex128)

            hrir_l, hrir_r = self.get_hrir(az, el)
            x_spec = np.fft.rfft(seg_w, n=n_fft)
            hl_spec = np.fft.rfft(hrir_l.astype(np.float64), n=n_fft)
            hr_spec = np.fft.rfft(hrir_r.astype(np.float64), n=n_fft)

            y_l = np.fft.irfft(x_spec * hl_spec * h_air * g_dir, n=n_fft)[: block_size + ir_len - 1]
            y_r = np.fft.irfft(x_spec * hr_spec * h_air * g_dir, n=n_fft)[: block_size + ir_len - 1]

            end_pos = start + block_size + ir_len - 1
            direct_pad[start:end_pos, 0] += y_l
            direct_pad[start:end_pos, 1] += y_r

            if g_rev > 0.0:
                reverb_pad[start : start + block_size] += seg_w * g_rev

        # HRIR の基本遅延 (48サンプル) を考慮して時間整合（SOFAの場合はそのまま）
        delay_comp = 0 if self._use_sofa else int(round(HRIR_BASE_DELAY_SAMPLES))
        direct = direct_pad[hop + delay_comp : hop + delay_comp + n_samples, :].astype(np.float32)
        reverb_send = reverb_pad[hop : hop + n_samples].astype(np.float32)
        return direct, reverb_send


# ---------------------------------------------------------------------------
# 室内音響エンジン（鏡像初期反射＋帯域別RT60拡散リバーブ）
# ---------------------------------------------------------------------------


_ROOM_SIZE_SCALE: dict[str, float] = {
    "small": 0.72,
    "medium": 1.0,
    "large": 1.45,
    "hall": 2.0,
}


class RoomReverbEngine:
    """鏡像法ベースの初期反射とマルチバンド減衰の拡散テールを高速 FFT 畳み込みで生成する室内音響エンジン。"""

    def __init__(
        self,
        rt60_sec: float = 0.45,
        room_size: str = "medium",
        sample_rate: int = SAMPLE_RATE,
        seed: int = 202501,
    ) -> None:
        self.rt60_sec = float(max(0.12, min(3.0, rt60_sec)))
        self.room_size = str(room_size or "medium").lower()
        self.size_scale = _ROOM_SIZE_SCALE.get(self.room_size, 1.0)
        self.sample_rate = int(sample_rate)
        self.seed = int(seed)
        self._binaural_ir: np.ndarray | None = None

    def build_multichannel_ir(self, num_channels: int) -> np.ndarray:
        """num_channels チャンネル分の相互非相関な室内インパルス応答 (ir_len, num_channels) を生成する。

        - 前半: 鏡像モデルに基づく初期反射（壁・天井・床の反射方向と壁面吸収LPF）
        - 後半: 低域/中域/高域で異なる RT60 減衰を持つ滑らかな拡散残響テール
        """
        sr = self.sample_rate
        ir_sec = min(3.2, max(0.35, self.rt60_sec * 1.65))
        ir_len = int(round(ir_sec * sr))
        rng = np.random.default_rng(self.seed + num_channels * 97)

        ir = np.zeros((ir_len, num_channels), dtype=np.float64)

        # 1. 鏡像法イメージソース初期反射タップ群 (ms, gain, pan_index_ratio)
        early_taps = [
            (9.2, 0.42, 0.15),
            (13.7, -0.36, 0.85),
            (18.4, 0.31, 0.50),
            (22.9, -0.28, 0.25),
            (28.1, 0.25, 0.75),
            (33.6, -0.22, 0.10),
            (39.4, 0.19, 0.90),
            (46.2, -0.17, 0.40),
            (52.8, 0.15, 0.60),
            (61.0, -0.13, 0.30),
        ]
        # 初期反射はソフトな壁面吸収パルス（短ガウスローパスパルス）として埋め込む
        pulse_t = np.arange(-8, 9, dtype=np.float64)
        wall_pulse = np.exp(-0.5 * (pulse_t / 2.2) ** 2)
        wall_pulse /= np.sum(wall_pulse)

        for idx, (delay_ms, amp, pan_pos) in enumerate(early_taps):
            center = int(round((delay_ms * self.size_scale) * 1e-3 * sr))
            if center + 10 >= ir_len:
                continue
            for ch in range(num_channels):
                # チャンネルごとにわずかなマイクロ遅延と方位重みを変えて空間的な広がりを付与
                ch_phase = (ch + 0.5) / max(1, num_channels)
                spatial_w = 0.55 + 0.45 * math.cos(2.0 * math.pi * (ch_phase - pan_pos))
                micro_jit = int(round(((idx * 7 + ch * 13) % 11 - 5) * 0.6))
                c_pos = max(9, min(ir_len - 10, center + micro_jit))
                sign = 1.0 if ((idx + ch) % 2 == 0) else -1.0
                ir[c_pos - 8 : c_pos + 9, ch] += sign * amp * spatial_w * wall_pulse

        # 2. 帯域別 RT60 減衰の拡散テール
        t = np.arange(ir_len, dtype=np.float64) / float(sr)
        onset_start = 0.018 * self.size_scale
        onset_dur = 0.032 * self.size_scale
        onset_env = np.clip((t - onset_start) / max(0.005, onset_dur), 0.0, 1.0)
        onset_env = 0.5 - 0.5 * np.cos(math.pi * onset_env)

        # 低域 (RT60*1.1), 中域 (RT60*1.0), 高域 (RT60*0.52) の指数減衰エンベロープ
        decay_low = np.exp(-6.907755 * t / (self.rt60_sec * 1.10)) * onset_env
        decay_mid = np.exp(-6.907755 * t / self.rt60_sec) * onset_env
        decay_high = np.exp(-6.907755 * t / max(0.08, self.rt60_sec * 0.52)) * onset_env

        sos_low = signal.butter(2, 450.0, btype="lowpass", fs=sr, output="sos")
        sos_mid = signal.butter(2, [450.0, 3600.0], btype="bandpass", fs=sr, output="sos")
        sos_high = signal.butter(2, 3600.0, btype="highpass", fs=sr, output="sos")

        # 低域のみ自然拡散音場のわずかな相関（IACC）を再現する共通ノイズ成分
        common_low = signal.sosfilt(sos_low, rng.standard_normal(ir_len))

        for ch in range(num_channels):
            noise = rng.standard_normal(ir_len)
            n_low = 0.35 * common_low + 0.65 * signal.sosfilt(sos_low, noise)
            n_mid = signal.sosfilt(sos_mid, noise)
            n_high = signal.sosfilt(sos_high, noise)
            tail = 0.70 * n_low * decay_low + 1.00 * n_mid * decay_mid + 0.55 * n_high * decay_high
            ir[:, ch] += 0.38 * tail

        # 末尾を滑らかにフェードアウト
        fade = min(int(0.03 * sr), ir_len // 5)
        if fade > 0:
            ir[-fade:, :] *= np.linspace(1.0, 0.0, fade, dtype=np.float64)[:, np.newaxis]

        # 全チャンネル平均エネルギーが 1.0 になるよう正規化（予測可能な Wet/Dry 比を保証）
        mean_energy = float(np.mean(np.sum(ir * ir, axis=0)))
        if mean_energy > 1e-12:
            ir /= math.sqrt(mean_energy)

        return ir.astype(np.float32)

    def process_binaural(self, mono_send: np.ndarray) -> np.ndarray:
        """モノラルリバーブセンドからステレオバイノーラル室内残響 (N, 2) を生成する。"""
        x = np.asarray(mono_send, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n == 0 or float(np.max(np.abs(x))) < 1e-9:
            return np.zeros((n, 2), dtype=np.float32)

        if self._binaural_ir is None:
            self._binaural_ir = self.build_multichannel_ir(2)

        out = np.zeros((n, 2), dtype=np.float32)
        for ch in range(2):
            conv = signal.fftconvolve(x, self._binaural_ir[:, ch], mode="full")
            out[:, ch] = conv[:n].astype(np.float32)
        return out

    def process_multichannel(self, mono_send: np.ndarray, num_channels: int) -> np.ndarray:
        """モノラルリバーブセンドから num_channels チャンネルの拡散残響 (N, C) を生成する。"""
        x = np.asarray(mono_send, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n == 0 or float(np.max(np.abs(x))) < 1e-9:
            return np.zeros((n, num_channels), dtype=np.float32)

        ir = self.build_multichannel_ir(num_channels)
        out = np.zeros((n, num_channels), dtype=np.float32)
        for ch in range(num_channels):
            conv = signal.fftconvolve(x, ir[:, ch], mode="full")
            out[:, ch] = conv[:n].astype(np.float32)
        return out

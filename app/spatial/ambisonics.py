"""Ambisonics (AmbiX: ACN チャンネル順 + SN3D 正規化) 1〜7次（4ch〜64ch）エンコーダー＆バイノーラルデコーダー.

対応次数（IEM Plug-in Suite / SPARTA 準拠 1〜7次）:
- 1次 FOA:  4ch (W, Y, Z, X)
- 2次 HOA:  9ch
- 3次 HOA: 16ch（標準推奨既定）
- 4次 HOA: 25ch
- 5次 HOA: 36ch
- 6次 HOA: 49ch
- 7次 HOA: 64ch（上限）

座標変換規約:
- CineSS 入力: azimuth_deg ∈ [-180, 180]（0°=正面, 負=左 -90°, 正=右 +90°）, elevation_deg ∈ [-90, 90]（正=上）
- AmbiX 標準: +X=前, +Y=左, +Z=上、数学方位角 phi = -radians(azimuth_deg)
  → 左音源 (azimuth_deg = -90°) は phi = +90° → ACN 1 (Y) = +1.0（正）
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal
from scipy.special import eval_legendre, roots_legendre

from .hrtf import (
    HRTFEngine,
    RoomReverbEngine,
    SAMPLE_RATE,
    air_absorption_response,
    distance_direct_gain,
    distance_reverb_send,
    fit_length,
    interpolate_keyframes,
    load_audio_48k,
)

MIN_AMBISONICS_ORDER: int = 1
MAX_AMBISONICS_ORDER: int = 7

# ---------------------------------------------------------------------------
# AmbiX (ACN + SN3D) 一般次数 N=1..7 実球面調和関数
# ---------------------------------------------------------------------------


def ambisonics_channel_count(order: int) -> int:
    """Ambisonics 次数 (1〜7) に対応するチャンネル数 (order + 1)^2 を返す。"""
    order_int = int(order)
    if order_int < MIN_AMBISONICS_ORDER or order_int > MAX_AMBISONICS_ORDER:
        raise ValueError(
            f"サポートされる Ambisonics 次数は {MIN_AMBISONICS_ORDER}〜{MAX_AMBISONICS_ORDER} です（指定: {order}）"
        )
    return (order_int + 1) ** 2


def acn_sn3d_gains(
    azimuth_deg: float | np.ndarray,
    elevation_deg: float | np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """AmbiX (ACN 順, SN3D 正規化, Condon-Shortley 位相なし) の一般次数 (N=1..7) 実球面調和関数を計算する。

    漸化式（Condon-Shortley 位相なしの陪 Legendre 多項式 P_l^m(sin δ)）:
      - P_0^0 = 1
      - P_m^m = (2m - 1) * cos(δ) * P_{m-1}^{m-1}
      - P_{m+1}^m = (2m + 1) * sin(δ) * P_m^m
      - (l - m) P_l^m = (2l - 1) sin(δ) P_{l-1}^m - (l + m - 1) P_{l-2}^m
    SN3D 正規化係数:
      - N_l^m = sqrt((2 - δ_{m,0}) * (l - m)! / (l + m)!)

    Args:
        azimuth_deg: 方位角（度）。負=左, 正=右。スカラーまたは 1D 配列 (T,)。
        elevation_deg: 仰角（度）。正=上。スカラーまたは 1D 配列 (T,)。
        order: Ambisonics 次数（1〜7）。

    Returns:
        スカラー入力時は shape (C,)、配列入力時は shape (T, C) の float64 配列。
    """
    order_int = int(order)
    n_ch = ambisonics_channel_count(order_int)
    scalar_in = np.ndim(azimuth_deg) == 0 and np.ndim(elevation_deg) == 0

    az = np.atleast_1d(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevation_deg, dtype=np.float64))
    if az.shape != el.shape:
        az, el = np.broadcast_arrays(az, el)

    n_pts = az.shape[0]
    # CineSS 規約（左=負）から AmbiX 標準（+Y=左, 反時計回り正）への厳密変換
    phi = -np.radians(az)
    delta = np.radians(el)

    c = np.cos(delta)  # >= 0 for delta in [-pi/2, pi/2]
    s = np.sin(delta)

    # cos(m*phi), sin(m*phi) を m=0..order 分事前計算
    cos_mphi = [np.ones(n_pts, dtype=np.float64)]
    sin_mphi = [np.zeros(n_pts, dtype=np.float64)]
    for m in range(1, order_int + 1):
        cos_mphi.append(np.cos(m * phi))
        sin_mphi.append(np.sin(m * phi))

    # P_l^m(s) テーブルを漸化式で構築（Condon-Shortley 符号 (-1)^m なし）
    p_table: dict[tuple[int, int], np.ndarray] = {}
    p_table[(0, 0)] = np.ones(n_pts, dtype=np.float64)
    for m in range(1, order_int + 1):
        p_table[(m, m)] = (2 * m - 1) * c * p_table[(m - 1, m - 1)]
    for m in range(0, order_int):
        p_table[(m + 1, m)] = (2 * m + 1) * s * p_table[(m, m)]
    for m in range(0, order_int - 1):
        for l in range(m + 2, order_int + 1):
            p_table[(l, m)] = (
                (2 * l - 1) * s * p_table[(l - 1, m)] - (l + m - 1) * p_table[(l - 2, m)]
            ) / float(l - m)

    # ACN 順（acn = l*(l+1) + m）に SN3D 正規化して格納
    y = np.empty((n_pts, n_ch), dtype=np.float64)
    for l in range(order_int + 1):
        center_acn = l * (l + 1)
        # m = 0 (SN3D 係数 = 1.0)
        y[:, center_acn] = p_table[(l, 0)]
        for m in range(1, l + 1):
            norm_lm = math.sqrt(2.0 * math.factorial(l - m) / float(math.factorial(l + m)))
            plm = norm_lm * p_table[(l, m)]
            y[:, center_acn + m] = plm * cos_mphi[m]
            y[:, center_acn - m] = plm * sin_mphi[m]

    return y[0] if scalar_in else y


def max_re_weights(order: int) -> np.ndarray:
    """N次 (1〜7) 3D Ambisonics の厳密な Max-rE 次数別重み a_l (l=0..order) を計算する。

    3D 球面上の最大エネルギーベクトル r_E を実現する厳密解:
    a_l = P_l(x_max), ただし x_max は Legendre 多項式 P_{order+1}(x) = 0 の最大根。
    """
    order_int = int(order)
    ambisonics_channel_count(order_int)
    roots, _ = roots_legendre(order_int + 1)
    x_max = float(np.max(roots))
    return np.array([float(eval_legendre(l, x_max)) for l in range(order_int + 1)], dtype=np.float64)


# ---------------------------------------------------------------------------
# 球面仮想スピーカーレイアウト（全次数で厳密左右対称）
# ---------------------------------------------------------------------------


def _unit_vecs_to_ciness_angles(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AmbiX 単位ベクトル (+X=前, +Y=左, +Z=上) を CineSS (azimuth_deg, elevation_deg) に変換する。"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    u = vecs / np.maximum(norms, 1e-12)
    phi = np.arctan2(u[:, 1], u[:, 0])
    az_deg = -np.degrees(phi)  # +Y (phi=+90°) -> az=-90° (左)
    el_deg = np.degrees(np.arcsin(np.clip(u[:, 2], -1.0, 1.0)))
    return az_deg, el_deg


def _symmetric_fibonacci_sphere(num_points: int) -> np.ndarray:
    """Y軸（左右軸）に関して厳密に鏡面対称な等面積 Fibonacci 球面点群 (num_points, 3) を生成する。

    左半球 (y > 0) に num_points // 2 点を等面積らせん配置し、右半球へ (x, -y, z) で厳密鏡映する。
    これにより正面 0° 音源の左右バランス誤差 0.00 dB と極めて低い条件数を両立する。
    """
    half = max(6, int(num_points) // 2)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    pts = np.empty((half * 2, 3), dtype=np.float64)
    for i in range(half):
        y_val = (i + 0.5) / float(half)  # (0, 1) 等間隔 → 球面帯の面積が完全均等
        r_xz = math.sqrt(max(0.0, 1.0 - y_val * y_val))
        theta = i * golden_angle
        x_val = r_xz * math.cos(theta)
        z_val = r_xz * math.sin(theta)
        pts[2 * i] = (x_val, y_val, z_val)
        pts[2 * i + 1] = (x_val, -y_val, z_val)
    return pts


def get_virtual_speaker_layout(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """次数 (1〜7) に応じた十分密かつ厳密左右対称な球面仮想スピーカー配置 (az_deg, el_deg, weights) を返す。

    - 1次 (4ch):  12点 正二十面体（厳密球面5-デザイン）
    - 2〜3次 (9/16ch): 32点 Pentakis 十二面体（正二十面体12頂点＋正十二面体20頂点、完全対称）
    - 4〜5次 (25/36ch): 64点 左右対称 Fibonacci 球面
    - 6〜7次 (49/64ch): 110点 左右対称 Fibonacci 球面
    """
    order_int = int(order)
    ambisonics_channel_count(order_int)

    if order_int == 1:
        gr = (1.0 + math.sqrt(5.0)) * 0.5
        raw = []
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                raw.append([0.0, s1, s2 * gr])
                raw.append([s1, s2 * gr, 0.0])
                raw.append([s2 * gr, 0.0, s1])
        vecs = np.array(raw, dtype=np.float64)
        az_deg, el_deg = _unit_vecs_to_ciness_angles(vecs)
        weights = np.full(len(vecs), 1.0 / len(vecs), dtype=np.float64)
        return az_deg, el_deg, weights

    if order_int <= 3:
        # 32点 Pentakis 十二面体（正二十面体 12点 + 正十二面体 20点、全軸鏡面対称）
        gr = (1.0 + math.sqrt(5.0)) * 0.5
        inv_gr = 1.0 / gr
        raw = []
        # 12 正二十面体頂点
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                raw.append([0.0, s1, s2 * gr])
                raw.append([s1, s2 * gr, 0.0])
                raw.append([s2 * gr, 0.0, s1])
        # 8 立方体頂点 + 12 黄金長方形頂点 = 20 正十二面体頂点
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    raw.append([sx, sy, sz])
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                raw.append([0.0, s1 * inv_gr, s2 * gr])
                raw.append([s1 * inv_gr, s2 * gr, 0.0])
                raw.append([s2 * gr, 0.0, s1 * inv_gr])
        vecs = np.array(raw, dtype=np.float64)
        az_deg, el_deg = _unit_vecs_to_ciness_angles(vecs)
        weights = np.full(len(vecs), 1.0 / len(vecs), dtype=np.float64)
        return az_deg, el_deg, weights

    if order_int <= 5:
        vecs = _symmetric_fibonacci_sphere(64)
    else:
        vecs = _symmetric_fibonacci_sphere(110)

    az_deg, el_deg = _unit_vecs_to_ciness_angles(vecs)
    weights = np.full(len(vecs), 1.0 / len(vecs), dtype=np.float64)
    return az_deg, el_deg, weights


def build_max_re_decoder_matrix(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """任意次数 N=1..7 の Max-rE 重み付き厳密モード整合デコード行列 D (K, C) と仮想スピーカー座標を返す。

    Moore-Penrose 擬似逆行列 `D_basic = pinv(Y_spk.T)` により `Y_spk.T @ D_basic = I_C` を
    機械精度で保証し、各次数 l に厳密な Max-rE 重み `a_l` を乗じる。
    """
    order_int = int(order)
    az_deg, el_deg, _weights = get_virtual_speaker_layout(order_int)
    y_spk = acn_sn3d_gains(az_deg, el_deg, order=order_int)  # (K, C)
    a_l = max_re_weights(order_int)

    n_ch = ambisonics_channel_count(order_int)
    a_per_ch = np.empty(n_ch, dtype=np.float64)
    for l in range(order_int + 1):
        a_per_ch[l * l : (l + 1) * (l + 1)] = a_l[l]

    # モード整合最小ノルム解 (K, C): Y_spk.T @ d_basic = I_C
    d_basic = np.linalg.pinv(y_spk.T)
    decoder = d_basic * a_per_ch[np.newaxis, :]
    return decoder, az_deg, el_deg


# ---------------------------------------------------------------------------
# Ambisonics レンダラー（エンコード＋3D室内残響＋バイノーラルデコード）
# ---------------------------------------------------------------------------


def _apply_dynamic_distance_mono(
    mono_audio: np.ndarray,
    keyframes: list[dict[str, Any]] | None,
    gain_db: float,
    rt60_sec: float,
    sample_rate: int = SAMPLE_RATE,
    block_size: int = 2048,
    role: str = "sfx",
) -> tuple[np.ndarray, np.ndarray]:
    """距離減衰＋距離依存空気吸収 LPF を Hann 窓 50% オーバーラップでモノラル信号へ滑らかに適用する。"""
    x = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
    n_samples = x.shape[0]
    gain_lin = float(10.0 ** (float(gain_db) / 20.0))
    if n_samples == 0 or gain_lin == 0.0 or float(np.max(np.abs(x))) < 1e-9:
        return np.zeros((n_samples,), dtype=np.float32), np.zeros((n_samples,), dtype=np.float32)

    hop = block_size // 2
    window = np.hanning(block_size).astype(np.float64)
    x_pad = np.pad(x.astype(np.float64) * gain_lin, (hop, hop + block_size), mode="constant")

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


def _decorrelate_delay(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """微小遅延による簡易非相関化（ベッド音源の自然な空間幅展開用）。"""
    if delay_samples <= 0 or delay_samples >= x.shape[0]:
        return x.copy()
    out = np.zeros_like(x)
    out[delay_samples:] = x[:-delay_samples]
    return out


class AmbisonicsRenderer:
    """一般次数 1〜7 (AmbiX: ACN + SN3D) エンコーダー＆Max-rE HRTF バイノーラルデコーダー。"""

    def __init__(
        self,
        order: int = 3,
        hrtf_engine: HRTFEngine | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.order = int(order)
        self.num_channels = ambisonics_channel_count(self.order)
        self.sample_rate = int(sample_rate)
        self.hrtf_engine = hrtf_engine or HRTFEngine(sample_rate=self.sample_rate)
        self._binaural_filters: tuple[np.ndarray, np.ndarray] | None = None

    def encode_point_source(
        self,
        mono_audio: np.ndarray,
        keyframes: list[dict[str, Any]] | None,
        control_step: int = 64,
    ) -> np.ndarray:
        """動的キーフレーム補間によりモノラル音源を AmbiX (N, C) へ滑らかにエンコードする。"""
        x = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n == 0 or float(np.max(np.abs(x))) < 1e-10:
            return np.zeros((n, self.num_channels), dtype=np.float32)

        # 制御点間隔 control_step (既定 64サンプル ≈ 1.33ms) で球面調和係数を計算し線形補間
        ctrl_idx = np.arange(0, n, control_step, dtype=np.int64)
        if ctrl_idx[-1] != n - 1:
            ctrl_idx = np.append(ctrl_idx, n - 1)
        ctrl_times = ctrl_idx.astype(np.float64) / float(self.sample_rate)
        az_c, el_c, _ = interpolate_keyframes(keyframes, ctrl_times)
        sh_ctrl = acn_sn3d_gains(az_c, el_c, order=self.order)  # (M, C)

        sample_idx = np.arange(n, dtype=np.float64)
        ctrl_idx_f = ctrl_idx.astype(np.float64)

        b_field = np.empty((n, self.num_channels), dtype=np.float32)
        for ch in range(self.num_channels):
            g_ch = np.interp(sample_idx, ctrl_idx_f, sh_ctrl[:, ch])
            b_field[:, ch] = (x * g_ch).astype(np.float32)
        return b_field

    def encode_bed_source(
        self,
        audio_2d: np.ndarray,
        gain_db: float = 0.0,
    ) -> np.ndarray:
        """BGM・環境音などのベッド音源を自然な空間幅（±45°〜±110°仮想音源展開）で AmbiX エンコードする。"""
        arr = np.asarray(audio_2d, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n = arr.shape[0]
        gain_lin = float(10.0 ** (float(gain_db) / 20.0))
        if n == 0 or gain_lin == 0.0 or float(np.max(np.abs(arr))) < 1e-10:
            return np.zeros((n, self.num_channels), dtype=np.float32)

        b_field = np.zeros((n, self.num_channels), dtype=np.float32)

        if arr.shape[1] >= 2:
            left = arr[:, 0] * gain_lin
            right = arr[:, 1] * gain_lin
            virtual_feeds = [
                (left * 0.72, -45.0, 0.0),
                (right * 0.72, 45.0, 0.0),
                (_decorrelate_delay(left, int(0.011 * self.sample_rate)) * 0.42, -110.0, 15.0),
                (_decorrelate_delay(right, int(0.014 * self.sample_rate)) * 0.42, 110.0, 15.0),
            ]
        else:
            mono = arr[:, 0] * gain_lin
            virtual_feeds = [
                (mono * 0.45, -45.0, 0.0),
                (_decorrelate_delay(mono, int(0.008 * self.sample_rate)) * 0.45, 45.0, 0.0),
                (_decorrelate_delay(mono, int(0.016 * self.sample_rate)) * 0.32, -110.0, 15.0),
                (_decorrelate_delay(mono, int(0.023 * self.sample_rate)) * 0.32, 110.0, 15.0),
            ]

        # 高次 (4〜7次) のベッド音源が点状に尖りすぎないよう緩やかに高次テーパーを掛ける
        order_taper = np.ones(self.num_channels, dtype=np.float32)
        for l in range(2, self.order + 1):
            order_taper[l * l : (l + 1) * (l + 1)] = float(0.85 ** (l - 1))

        for sig, az, el in virtual_feeds:
            sh = acn_sn3d_gains(az, el, order=self.order).astype(np.float32) * order_taper
            b_field += sig[:, np.newaxis] * sh[np.newaxis, :]

        return b_field

    def _get_binaural_ambi_filters(self) -> tuple[np.ndarray, np.ndarray]:
        """Max-rE 仮想スピーカーデコード行列と各方向 HRIR を合成した (C, ir_len) フィルタ対を返す。

        線形性により Σ_k (Σ_j D_{k,j} B_j) * h_k == Σ_j B_j * (Σ_k D_{k,j} h_k) が厳密に成立。
        """
        if self._binaural_filters is not None:
            return self._binaural_filters

        decoder, az_deg, el_deg = build_max_re_decoder_matrix(self.order)  # (K, C)
        k_spk = len(az_deg)
        ir_len = self.hrtf_engine.ir_len

        h_l_spk = np.empty((k_spk, ir_len), dtype=np.float64)
        h_r_spk = np.empty((k_spk, ir_len), dtype=np.float64)
        for k in range(k_spk):
            hl, hr = self.hrtf_engine.get_hrir(float(az_deg[k]), float(el_deg[k]))
            h_l_spk[k] = hl
            h_r_spk[k] = hr

        # 合成バイノーラルフィルタ: shape (C, ir_len)
        filt_l = decoder.T @ h_l_spk
        filt_r = decoder.T @ h_r_spk

        # 正面 (0°, 0°) 点音源の再現ゲインを広帯域でおよそ 0 dB に正規化
        sh_front = acn_sn3d_gains(0.0, 0.0, order=self.order)
        front_l = sh_front @ filt_l
        front_r = sh_front @ filt_r
        ref_l, ref_r = self.hrtf_engine.get_hrir(0.0, 0.0)
        e_decoded = float(np.sum(front_l * front_l) + np.sum(front_r * front_r))
        e_direct = float(np.sum(ref_l.astype(np.float64) ** 2) + np.sum(ref_r.astype(np.float64) ** 2))
        if e_decoded > 1e-12 and e_direct > 1e-12:
            norm = math.sqrt(e_direct / e_decoded)
            filt_l *= norm
            filt_r *= norm

        self._binaural_filters = (filt_l.astype(np.float32), filt_r.astype(np.float32))
        return self._binaural_filters

    def decode_to_binaural(self, ambix_audio: np.ndarray) -> np.ndarray:
        """AmbiX 音場 (N, C) を Max-rE 仮想スピーカー＋HRTF 畳み込みによりバイノーラル (N, 2) へデコードする。"""
        b = np.asarray(ambix_audio, dtype=np.float32)
        if b.ndim != 2 or b.shape[1] != self.num_channels:
            raise ValueError(f"AmbiX 配列のチャンネル数が一致しません: 期待 {self.num_channels}, 実際 {b.shape}")
        n = b.shape[0]
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        filt_l, filt_r = self._get_binaural_ambi_filters()
        out_l = np.zeros(n, dtype=np.float64)
        out_r = np.zeros(n, dtype=np.float64)

        for ch in range(self.num_channels):
            sig = b[:, ch]
            if float(np.max(np.abs(sig))) < 1e-10:
                continue
            out_l += signal.fftconvolve(sig, filt_l[ch], mode="full")[:n]
            out_r += signal.fftconvolve(sig, filt_r[ch], mode="full")[:n]

        return np.stack([out_l, out_r], axis=1).astype(np.float32)

    def render_scene(
        self,
        scene_dict: dict[str, Any],
        base_dir: str | None = None,
        audio_overrides: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """シーン定義から AmbiX マスター (N, C) とバイノーラル出力 (N, 2) をレンダリングする。"""
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

        ambix_mix = np.zeros((n_samples, self.num_channels), dtype=np.float32)
        reverb_bus_mono = np.zeros((n_samples,), dtype=np.float32)

        for obj, audio in loaded_tracks:
            audio_fit = fit_length(audio, n_samples)
            is_bed = bool(obj.get("is_bed", False))
            gain_db = float(obj.get("gain_db", 0.0))

            if is_bed:
                ambix_mix += self.encode_bed_source(audio_fit, gain_db=gain_db)
            else:
                mono = (
                    audio_fit[:, 0]
                    if audio_fit.shape[1] == 1
                    else np.mean(audio_fit, axis=1, dtype=np.float64).astype(np.float32)
                )
                keyframes = obj.get("keyframes")
                role = str(obj.get("role", "sfx"))
                direct_mono, rev_send = _apply_dynamic_distance_mono(
                    mono,
                    keyframes=keyframes,
                    gain_db=gain_db,
                    rt60_sec=rt60_sec,
                    sample_rate=self.sample_rate,
                    role=role,
                )
                ambix_mix += self.encode_point_source(direct_mono, keyframes=keyframes)
                reverb_bus_mono += rev_send

        # 3D 拡散室内残響（4方向非相関残響を低次中心の拡散音場として加算 → 自然な包囲感）
        if float(np.max(np.abs(reverb_bus_mono))) > 1e-8:
            reverb_engine = RoomReverbEngine(
                rt60_sec=rt60_sec,
                room_size=room_size,
                sample_rate=self.sample_rate,
            )
            rev_4ch = reverb_engine.process_multichannel(reverb_bus_mono, num_channels=4)
            rev_dirs = [(-50.0, 12.0), (50.0, -10.0), (-135.0, 22.0), (135.0, 18.0)]
            # 拡散音場は 1次以下を主体とし、高次は空間エイリアシング防止のため急速減衰
            rev_taper = np.zeros(self.num_channels, dtype=np.float32)
            rev_taper[0:4] = 1.0
            if self.order >= 2:
                rev_taper[4:9] = 0.45
            if self.order >= 3:
                rev_taper[9:16] = 0.18
            for k, (az, el) in enumerate(rev_dirs):
                sh = acn_sn3d_gains(az, el, order=self.order).astype(np.float32) * rev_taper
                # 非ゼロチャンネルのみ加算して 7次 64ch 時の演算も高速化
                nz = np.nonzero(rev_taper)[0]
                ambix_mix[:, nz] += (rev_4ch[:, k : k + 1] * 0.5) * sh[np.newaxis, nz]

        binaural = self.decode_to_binaural(ambix_mix)
        return {
            "ambix": ambix_mix,
            "binaural": binaural,
            "channels": self.num_channels,
            "order": self.order,
            "hrtf_source": self.hrtf_engine.source_name,
        }

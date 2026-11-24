"""マスタリング・ダイアログ保護・ITU-R BS.1770-4 ラウドネス正規化・4xオーバーサンプリング True Peak リミッター.

特徴:
- ダイアログ保護: 80Hz HPF + 軽微なプレゼンス明瞭化 (3.2kHz)
- 空間イメージ保持リンク・コンプレッサー（全チャンネル共通サイドチェーンにより ILD / 音像バランスを一切崩さない）
- 4x オーバーサンプリング True Peak 計測＆先読みリンク・リミッター（≤ -1.0 dBTP 保証）
- ITU-R BS.1770-4 ラウドネス計測（pyloudnorm 優先、未導入・多ch・短尺時は厳密 K-weighting ゲーティングを内蔵実行）
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage, signal

from .hrtf import SAMPLE_RATE

try:
    import pyloudnorm as pyln  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pyln = None

LOUDNESS_PRESETS: dict[str, float] = {
    "cinema": -24.0,
    "atmos": -18.0,
    "web": -14.0,
}

# ITU-R BS.1770-4 48kHz 厳密 K-weighting 2段バイカッド係数
_BS1770_STAGE1_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
_BS1770_STAGE1_A = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
_BS1770_STAGE2_B = np.array([1.0, -2.0, 1.0], dtype=np.float64)
_BS1770_STAGE2_A = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)


# ---------------------------------------------------------------------------
# ダイアログ保護処理
# ---------------------------------------------------------------------------


def _peaking_sos(center_hz: float, gain_db: float, q: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Audio EQ Cookbook ピーキングEQの 1段 SOS 係数を返す。"""
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
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _apply_voice_compressor(y: np.ndarray, sample_rate: int) -> np.ndarray:
    """映画ダビング準拠のセリフ用ボイスコンプレッサー（2ms RMS, -20dB, 2.5:1, ゲート-55dB, 自動メイクアップ）。"""
    n = y.shape[0]
    if n == 0 or float(np.max(np.abs(y))) < 1e-9:
        return y

    blk = max(1, int(round(0.002 * sample_rate)))
    n_blks = int(math.ceil(n / blk))
    pad = n_blks * blk - n
    linked_sq = np.mean(y * y, axis=1)
    if pad > 0:
        linked_sq = np.pad(linked_sq, (0, pad), mode="edge")
    rms_blk = np.sqrt(np.mean(linked_sq.reshape(n_blks, blk), axis=1))

    att_a = math.exp(-float(blk) / max(1.0, 0.008 * sample_rate))
    rel_a = math.exp(-float(blk) / max(1.0, 0.090 * sample_rate))
    env = np.empty_like(rms_blk)
    cur = float(rms_blk[0])
    for i in range(n_blks):
        v = float(rms_blk[i])
        coeff = att_a if v > cur else rel_a
        cur = coeff * cur + (1.0 - coeff) * v
        env[i] = cur

    env_db = 20.0 * np.log10(np.maximum(env, 1e-10))
    gate_db = -55.0
    voiced_mask = env_db > gate_db
    if not np.any(voiced_mask):
        return y

    threshold_db = -20.0
    ratio = 2.5
    knee_db = 4.0
    half_k = 0.5 * knee_db
    slope = (1.0 / ratio) - 1.0

    gr_db = np.zeros_like(env_db)
    above = env_db >= (threshold_db + half_k)
    in_knee = (env_db > (threshold_db - half_k)) & (~above)
    gr_db[above] = slope * (env_db[above] - threshold_db)
    gr_db[in_knee] = slope * ((env_db[in_knee] - threshold_db + half_k) ** 2) / (2.0 * knee_db)
    gr_db[~voiced_mask] = 0.0

    mean_gr_db = float(-np.mean(gr_db[voiced_mask]))
    makeup_db = min(4.0, max(0.0, 0.65 * mean_gr_db))
    total_db = np.where(voiced_mask, gr_db + makeup_db, 0.0)

    gain_blk = 10.0 ** (total_db / 20.0)
    blk_centers = (np.arange(n_blks, dtype=np.float64) + 0.5) * blk
    gain_sample = np.interp(np.arange(n, dtype=np.float64), blk_centers, gain_blk)
    return y * gain_sample[:, np.newaxis]


def _apply_deesser(y: np.ndarray, sample_rate: int) -> np.ndarray:
    """5.0〜9.5 kHz 歯擦音検出＆5.0〜10.0 kHz 相補 LR4 動的減衰ディエッサー（声の芯 100Hz〜4kHz 無劣化）。"""
    n = y.shape[0]
    if n == 0 or sample_rate < 22000 or float(np.max(np.abs(y))) < 1e-9:
        return y

    hi_det = min(9500.0, 0.45 * float(sample_rate))
    if hi_det <= 5200.0:
        return y

    sos_det = signal.butter(2, [5000.0, hi_det], btype="bandpass", fs=sample_rate, output="sos")
    sib_sig = signal.sosfilt(sos_det, y, axis=0)

    blk = max(1, int(round(0.001 * sample_rate)))
    n_blks = int(math.ceil(n / blk))
    pad = n_blks * blk - n

    sib_sq = np.mean(sib_sig * sib_sig, axis=1)
    full_sq = np.mean(y * y, axis=1)
    if pad > 0:
        sib_sq = np.pad(sib_sq, (0, pad), mode="edge")
        full_sq = np.pad(full_sq, (0, pad), mode="edge")

    sib_rms = np.sqrt(np.mean(sib_sq.reshape(n_blks, blk), axis=1))
    full_rms = np.sqrt(np.mean(full_sq.reshape(n_blks, blk), axis=1))

    sib_db = 20.0 * np.log10(np.maximum(sib_rms, 1e-10))
    sib_ratio = sib_rms / np.maximum(full_rms, 1e-10)

    excess_db = np.maximum(0.0, sib_db - (-22.0))
    ratio_weight = np.clip((sib_ratio - 0.25) / 0.20, 0.0, 1.0)
    target_atten_db = np.minimum(6.0, excess_db * 0.75 * ratio_weight)

    att_a = math.exp(-float(blk) / max(1.0, 0.003 * sample_rate))
    rel_a = math.exp(-float(blk) / max(1.0, 0.045 * sample_rate))
    atten_env = np.empty_like(target_atten_db)
    cur = 0.0
    for i in range(n_blks):
        v = float(target_atten_db[i])
        coeff = att_a if v > cur else rel_a
        cur = coeff * cur + (1.0 - coeff) * v
        atten_env[i] = cur

    if float(np.max(atten_env)) < 1e-3:
        return y

    # 位相整合済み LR4 相補3バンド分割（5kHz・10kHz）で 5.0〜10.0kHz のみを動的減衰
    hi_cut = min(10000.0, 0.45 * float(sample_rate))
    sos_5k_lp = signal.butter(2, 5000.0, btype="lowpass", fs=sample_rate, output="sos")
    sos_5k_hp = signal.butter(2, 5000.0, btype="highpass", fs=sample_rate, output="sos")
    sos_10k_lp = signal.butter(2, hi_cut, btype="lowpass", fs=sample_rate, output="sos")
    sos_10k_hp = signal.butter(2, hi_cut, btype="highpass", fs=sample_rate, output="sos")

    y_low5 = signal.sosfilt(sos_5k_lp, signal.sosfilt(sos_5k_lp, y, axis=0), axis=0)
    y_hi5 = signal.sosfilt(sos_5k_hp, signal.sosfilt(sos_5k_hp, y, axis=0), axis=0)

    # 低域側にも 10kHz オールパスを通すことで 5kHz/10kHz クロス境界の位相を完全一致させる
    y_low5_aligned = signal.sosfilt(sos_10k_lp, signal.sosfilt(sos_10k_lp, y_low5, axis=0), axis=0) + signal.sosfilt(
        sos_10k_hp, signal.sosfilt(sos_10k_hp, y_low5, axis=0), axis=0
    )
    y_5_10k = signal.sosfilt(sos_10k_lp, signal.sosfilt(sos_10k_lp, y_hi5, axis=0), axis=0)
    y_above_10k = signal.sosfilt(sos_10k_hp, signal.sosfilt(sos_10k_hp, y_hi5, axis=0), axis=0)

    gain_blk = 10.0 ** (-atten_env / 20.0)
    blk_centers = (np.arange(n_blks, dtype=np.float64) + 0.5) * blk
    gain_sample = np.interp(np.arange(n, dtype=np.float64), blk_centers, gain_blk)

    return y_low5_aligned + (y_5_10k * gain_sample[:, np.newaxis]) + y_above_10k


def process_dialog_stem(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hpf_hz: float = 80.0,
    presence_db: float = 1.2,
    presence_hz: float = 3200.0,
    enable_compressor: bool = True,
    enable_deesser: bool = True,
) -> np.ndarray:
    """映画ダビング準拠セリフ処理チェーン（80Hz HPF → ボイスコンプ → ディエッサー → 3.2kHz プレゼンス）。"""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or float(np.max(np.abs(arr))) < 1e-9:
        return arr.copy()

    orig_1d = arr.ndim == 1
    x = arr[:, np.newaxis].astype(np.float64) if orig_1d else arr.astype(np.float64)

    if hpf_hz > 0.0 and hpf_hz < 0.45 * sample_rate:
        sos_hpf = signal.butter(2, hpf_hz, btype="highpass", fs=sample_rate, output="sos")
        y = signal.sosfilt(sos_hpf, x, axis=0)
    else:
        y = x.copy()

    if enable_compressor:
        y = _apply_voice_compressor(y, sample_rate=sample_rate)

    if enable_deesser:
        y = _apply_deesser(y, sample_rate=sample_rate)

    if abs(presence_db) > 1e-3 and presence_hz < 0.45 * sample_rate:
        sos_pres = _peaking_sos(presence_hz, presence_db, q=1.2, sample_rate=sample_rate)
        y = signal.sosfilt(sos_pres, y, axis=0)

    out = y.astype(np.float32)
    return out[:, 0] if orig_1d else out


def compute_dialog_ducking_gain(
    dialog_mono: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_db: float = -3.0,
    threshold_db: float = -34.0,
    attack_ms: float = 15.0,
    release_ms: float = 220.0,
    hold_ms: float = 40.0,
    impact_override_mono: np.ndarray | None = None,
    impact_threshold_db: float = -16.0,
) -> np.ndarray:
    """セリフ合算モノラルから発話エンベロープを検出し、発話中のみ滑らかに duck_db 分減衰するゲイン曲線 (N,) を返す。

    `impact_override_mono` が指定された場合、その約 2ms RMS が `impact_threshold_db` を超える瞬間は
    スマート・インパクト免除としてゲインを滑らかに `1.0` 側へ引き戻す。
    """
    x = np.asarray(dialog_mono, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return np.ones((0,), dtype=np.float32)

    overall_rms = float(np.sqrt(np.mean(x * x)))
    if overall_rms < 10.0 ** (-60.0 / 20.0) or duck_db >= -1e-4:
        return np.ones((n,), dtype=np.float32)

    blk = max(1, int(round(0.005 * sample_rate)))
    n_blks = int(math.ceil(n / blk))
    pad = n_blks * blk - n
    sq = x * x
    if pad > 0:
        sq = np.pad(sq, (0, pad), mode="edge")
    rms_blk = np.sqrt(np.mean(sq.reshape(n_blks, blk), axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms_blk, 1e-10))

    active = rms_db > float(threshold_db)
    if not np.any(active):
        return np.ones((n,), dtype=np.float32)

    hold_blks = max(0, int(round(float(hold_ms) * 1e-3 * sample_rate / blk)))
    held_active = active.copy()
    if hold_blks > 0:
        rem = 0
        for i in range(n_blks):
            if active[i]:
                rem = hold_blks
                held_active[i] = True
            elif rem > 0:
                held_active[i] = True
                rem -= 1

    target_db = np.where(held_active, float(duck_db), 0.0)
    att_a = math.exp(-float(blk) / max(1.0, float(attack_ms) * 1e-3 * sample_rate))
    rel_a = math.exp(-float(blk) / max(1.0, float(release_ms) * 1e-3 * sample_rate))

    env_db = np.empty_like(target_db)
    cur_db = 0.0
    for i in range(n_blks):
        t_val = float(target_db[i])
        coeff = att_a if t_val < cur_db else rel_a
        cur_db = coeff * cur_db + (1.0 - coeff) * t_val
        env_db[i] = cur_db

    gain_blk = 10.0 ** (env_db / 20.0)
    blk_centers = (np.arange(n_blks, dtype=np.float64) + 0.5) * blk
    gain_sample = np.interp(np.arange(n, dtype=np.float64), blk_centers, gain_blk)

    # スマート・インパクト免除: 約 2ms RMS が impact_threshold_db を超える瞬間は 1.0 へ引き戻す
    if impact_override_mono is not None:
        imp = np.asarray(impact_override_mono, dtype=np.float64).reshape(-1)
        if imp.size > 0 and float(np.max(np.abs(imp))) >= 1e-8:
            if imp.shape[0] > n:
                imp = imp[:n]
            elif imp.shape[0] < n:
                imp = np.pad(imp, (0, n - imp.shape[0]), mode="constant")

            blk_imp = max(1, int(round(0.002 * sample_rate)))
            n_blks_imp = int(math.ceil(n / blk_imp))
            pad_imp = n_blks_imp * blk_imp - n
            imp_sq = imp * imp
            if pad_imp > 0:
                imp_sq = np.pad(imp_sq, (0, pad_imp), mode="edge")
            imp_rms_blk = np.sqrt(np.mean(imp_sq.reshape(n_blks_imp, blk_imp), axis=1))
            imp_db = 20.0 * np.log10(np.maximum(imp_rms_blk, 1e-10))

            imp_active = imp_db > float(impact_threshold_db)
            if np.any(imp_active):
                hold_imp_blks = max(0, int(round(0.020 * sample_rate / blk_imp)))
                held_imp = imp_active.copy()
                if hold_imp_blks > 0:
                    rem_i = 0
                    for i in range(n_blks_imp):
                        if imp_active[i]:
                            rem_i = hold_imp_blks
                            held_imp[i] = True
                        elif rem_i > 0:
                            held_imp[i] = True
                            rem_i -= 1

                att_imp = math.exp(-float(blk_imp) / max(1.0, 0.002 * sample_rate))
                rel_imp = math.exp(-float(blk_imp) / max(1.0, 0.110 * sample_rate))
                exempt_env = np.empty(n_blks_imp, dtype=np.float64)
                cur_ex = 0.0
                for i in range(n_blks_imp):
                    tgt_ex = 1.0 if held_imp[i] else 0.0
                    c_ex = att_imp if tgt_ex > cur_ex else rel_imp
                    cur_ex = c_ex * cur_ex + (1.0 - c_ex) * tgt_ex
                    exempt_env[i] = cur_ex

                imp_centers = (np.arange(n_blks_imp, dtype=np.float64) + 0.5) * blk_imp
                exempt_sample = np.clip(
                    np.interp(np.arange(n, dtype=np.float64), imp_centers, exempt_env),
                    0.0,
                    1.0,
                )
                gain_sample = gain_sample + (1.0 - gain_sample) * exempt_sample

    min_gain = float(10.0 ** (float(duck_db) / 20.0))
    return np.clip(gain_sample, min_gain, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 K-weighting & ラウドネス計測
# ---------------------------------------------------------------------------


def _k_weight_filter(audio_2d: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """ITU-R BS.1770-4 K-weighting フィルタを (N, C) 音声へ適用する。"""
    if sample_rate == 48000:
        sos = np.vstack(
            [
                np.concatenate([_BS1770_STAGE1_B, _BS1770_STAGE1_A]),
                np.concatenate([_BS1770_STAGE2_B, _BS1770_STAGE2_A]),
            ]
        )
        return signal.sosfilt(sos, audio_2d.astype(np.float64), axis=0)

    # 48kHz 以外のサンプルレート向け双一次変換（BS.1770-4 アナログ原型準拠）
    f0 = 1681.974450955533
    g_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10.0 ** (g_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b0 = (vh + vb * k / q + k * k) / a0
    b1 = 2.0 * (k * k - vh) / a0
    b2 = (vh - vb * k / q + k * k) / a0
    a1 = 2.0 * (k * k - 1.0) / a0
    a2 = (1.0 - k / q + k * k) / a0

    f_hp = 38.13547087602444
    q_hp = 0.5003270373238773
    k2 = math.tan(math.pi * f_hp / sample_rate)
    a0_2 = 1.0 + k2 / q_hp + k2 * k2
    b0_2 = 1.0 / a0_2
    b1_2 = -2.0 / a0_2
    b2_2 = 1.0 / a0_2
    a1_2 = 2.0 * (k2 * k2 - 1.0) / a0_2
    a2_2 = (1.0 - k2 / q_hp + k2 * k2) / a0_2

    sos = np.array(
        [
            [b0, b1, b2, 1.0, a1, a2],
            [b0_2, b1_2, b2_2, 1.0, a1_2, a2_2],
        ],
        dtype=np.float64,
    )
    return signal.sosfilt(sos, audio_2d.astype(np.float64), axis=0)


def _channel_loudness_weights(num_channels: int) -> np.ndarray:
    """ITU-R BS.1770-4 / BS.2051 チャンネル重み G_i を返す。"""
    if num_channels == 1:
        return np.array([1.0], dtype=np.float64)
    if num_channels == 2:
        return np.array([1.0, 1.0], dtype=np.float64)
    if num_channels == 6:
        # 5.1: L, R, C, LFE(0), Ls(1.41), Rs(1.41)
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41], dtype=np.float64)
    if num_channels == 12:
        # 7.1.4: L, R, C, LFE(0), Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltr, Rtr
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    return np.ones(num_channels, dtype=np.float64)


def _measure_lufs_builtin(audio_2d: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """ITU-R BS.1770-4 K-weighting + 絶対(-70 LUFS)/相対(-10 LU)ゲーティング統合ラウドネス計測。"""
    n, ch = audio_2d.shape
    if n == 0 or float(np.max(np.abs(audio_2d))) < 1e-10:
        return -70.0

    filtered = _k_weight_filter(audio_2d, sample_rate=sample_rate)
    weights = _channel_loudness_weights(ch)

    block_len = int(round(0.4 * sample_rate))
    hop_len = int(round(0.1 * sample_rate))

    if n < block_len:
        ms = np.mean(filtered * filtered, axis=0)
        weighted_sum = float(np.sum(weights * ms))
        if weighted_sum <= 1e-15:
            return -70.0
        return float(max(-70.0, -0.691 + 10.0 * math.log10(weighted_sum)))

    # 400ms ブロック (75% オーバーラップ = 100ms ステップ) の高速累積和計算
    sq = filtered * filtered
    csum = np.vstack([np.zeros((1, ch), dtype=np.float64), np.cumsum(sq, axis=0)])
    starts = np.arange(0, n - block_len + 1, hop_len, dtype=np.int64)
    block_ms = (csum[starts + block_len] - csum[starts]) / float(block_len)  # (B, C)

    weighted_power = block_ms @ weights  # (B,)
    block_lufs = np.where(weighted_power > 1e-15, -0.691 + 10.0 * np.log10(np.maximum(weighted_power, 1e-15)), -120.0)

    # 第1段階: 絶対ゲート (-70 LKFS)
    abs_mask = block_lufs > -70.0
    if not np.any(abs_mask):
        return -70.0

    mean_abs = float(np.mean(weighted_power[abs_mask]))
    gamma_r = -0.691 + 10.0 * math.log10(max(mean_abs, 1e-15)) - 10.0

    # 第2段階: 相対ゲート (Gamma_r)
    rel_mask = abs_mask & (block_lufs > gamma_r)
    if not np.any(rel_mask):
        return float(max(-70.0, -0.691 + 10.0 * math.log10(max(mean_abs, 1e-15))))

    mean_rel = float(np.mean(weighted_power[rel_mask]))
    return float(-0.691 + 10.0 * math.log10(max(mean_rel, 1e-15)))


def measure_lufs(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """ITU-R BS.1770-4 統合ラウドネス (LUFS / LKFS) を計測する。"""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0 or float(np.max(np.abs(arr))) < 1e-10:
        return -70.0
    audio_2d = arr[:, np.newaxis] if arr.ndim == 1 else arr

    # pyloudnorm が導入済みかつ標準ステレオ/モノで 0.45秒以上の場合は pyloudnorm を優先使用
    if pyln is not None and audio_2d.shape[1] <= 2 and audio_2d.shape[0] >= int(0.45 * sample_rate):
        try:
            meter = pyln.Meter(sample_rate)
            val = float(meter.integrated_loudness(audio_2d))
            if np.isfinite(val):
                return val
        except Exception:
            pass

    return _measure_lufs_builtin(audio_2d, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# 4x オーバーサンプリング True Peak 計測＆先読みリンク・リミッター
# ---------------------------------------------------------------------------


def measure_true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float:
    """4x オーバーサンプリングによる True Peak レベル (dBTP) を計測する。"""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return -120.0
    sample_peak = float(np.max(np.abs(arr)))
    if sample_peak < 1e-12:
        return -120.0

    arr_2d = arr[:, np.newaxis] if arr.ndim == 1 else arr

    if arr_2d.shape[0] < 8 or oversample <= 1:
        tp = sample_peak
    else:
        # 多チャンネル時 (16〜64ch) の高速化: 最大チャンネルの 45% 未満しかない静かなチャンネルは
        # インターサンプルピークでも最大チャンネルを超え得ないため除外して厳密かつ高速に計測
        ch_peaks = np.max(np.abs(arr_2d), axis=0)
        hot_mask = ch_peaks >= (0.45 * sample_peak)
        sub = arr_2d[:, hot_mask] if np.any(hot_mask) else arr_2d
        up = signal.resample_poly(sub, int(oversample), 1, axis=0)
        tp = max(sample_peak, float(np.max(np.abs(up))))

    return float(20.0 * math.log10(max(tp, 1e-12)))


def true_peak_limiter(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 55.0,
    oversample: int = 4,
) -> np.ndarray:
    """空間イメージを完全保持するリンク先読み 4x True Peak リミッター（ceiling_dbtp 以下を保証）。"""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    orig_1d = arr.ndim == 1
    x = arr[:, np.newaxis].astype(np.float32) if orig_1d else arr.astype(np.float32).copy()
    n = x.shape[0]
    raw_max = float(np.max(np.abs(x)))
    if raw_max < 1e-10:
        return arr.copy()

    # わずかな安全マージン (-0.06 dB) を設けて 4x インターサンプルピークの超過を厳密防止
    internal_ceiling_lin = 10.0 ** ((ceiling_dbtp - 0.06) / 20.0)
    hard_ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)

    # 十分余裕がある場合（サンプルピークが天井の 55% 未満）はインターサンプル超過不可のため即座に返却
    if raw_max < 0.55 * hard_ceiling_lin:
        return arr.copy()

    look_n = max(4, int(round(lookahead_ms * 1e-3 * sample_rate)))
    rel_n = max(8, int(round(release_ms * 1e-3 * sample_rate)))

    for _pass in range(3):
        ch_peaks = np.max(np.abs(x), axis=0)
        sample_env = np.max(np.abs(x), axis=1).astype(np.float64)
        # 天井付近に迫るチャンネルのみ 4x オーバーサンプリング（64ch HOA でも高速＆厳密）
        hot_mask = ch_peaks >= (0.55 * hard_ceiling_lin)
        if n >= 8 and oversample > 1 and np.any(hot_mask):
            up = signal.resample_poly(x[:, hot_mask], int(oversample), 1, axis=0)
            tp_4x = np.max(np.abs(up), axis=1)
            pad_rem = (-tp_4x.shape[0]) % oversample
            if pad_rem:
                tp_4x = np.pad(tp_4x, (0, pad_rem), mode="edge")
            peak_env = np.maximum(sample_env, np.max(tp_4x.reshape(-1, oversample), axis=1)[:n])
        else:
            peak_env = sample_env

        max_tp = float(np.max(peak_env))
        if max_tp <= hard_ceiling_lin:
            break

        # 先読みピークホールド（対称幅 2*look_n + 1 → ピーク前後 look_n サンプルで最小ゲインを維持）
        held = ndimage.maximum_filter1d(peak_env, size=2 * look_n + 1, mode="nearest")
        g_raw = np.where(held > internal_ceiling_lin, internal_ceiling_lin / np.maximum(held, 1e-12), 1.0)

        # リリース側の滑らかな回復（ブロック間引き→1次追従→補間で高速＆ポンピング低減）
        step = max(1, look_n // 2)
        g_ds = g_raw[::step].copy()
        alpha_rel = math.exp(-float(step) / float(rel_n))
        for i in range(1, g_ds.shape[0]):
            if g_ds[i] > g_ds[i - 1]:
                g_ds[i] = min(g_ds[i], alpha_rel * g_ds[i - 1] + (1.0 - alpha_rel) * g_ds[i])
        ds_idx = np.arange(g_ds.shape[0], dtype=np.float64) * step
        g_rel = np.minimum(g_raw, np.interp(np.arange(n, dtype=np.float64), ds_idx, g_ds))

        # Hann 窓による滑らかなゲインスムージング（look_n 幅以下なのでピーク点での抑制保証を維持）
        win_len = max(3, (look_n // 2) * 2 + 1)
        hann_k = np.hanning(win_len + 2)[1:-1]
        hann_k /= np.sum(hann_k)
        g_smooth = signal.fftconvolve(np.pad(g_rel, (win_len // 2, win_len // 2), mode="edge"), hann_k, mode="valid")[:n]
        g_final = np.minimum(g_smooth, g_raw).astype(np.float32)

        # 全チャンネルへ同一ゲインを乗算（L/R バランス・空間フィールドを完全維持）
        x *= g_final[:, np.newaxis]

    # 最終確認: 万が一インターサンプルピークが ceiling_dbtp を上回る場合のみ一律微減衰
    if float(np.max(np.abs(x))) >= 0.55 * hard_ceiling_lin:
        final_tp_db = measure_true_peak_dbtp(x, oversample=oversample)
        if final_tp_db > ceiling_dbtp:
            trim = np.float32(10.0 ** ((ceiling_dbtp - 0.02 - final_tp_db) / 20.0))
            x *= trim

    return x[:, 0] if orig_1d else x


# ---------------------------------------------------------------------------
# 空間イメージ保持リンク・バスコンプレッサー
# ---------------------------------------------------------------------------


def linked_bus_compressor(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -14.0,
    ratio: float = 2.0,
    knee_db: float = 4.0,
    attack_ms: float = 12.0,
    release_ms: float = 120.0,
) -> np.ndarray:
    """全チャンネル共通サイドチェーンによるソフトニー・リンクコンプレッサー（空間定位・ILD完全保持）。"""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or float(np.max(np.abs(arr))) < 1e-9:
        return arr.copy()

    orig_1d = arr.ndim == 1
    x = arr[:, np.newaxis].astype(np.float64) if orig_1d else arr.astype(np.float64)
    n = x.shape[0]

    # 全チャンネル共通のリンク検出信号（2ms ブロック RMS で高速＆滑らかに包絡検出）
    blk = max(1, int(round(0.002 * sample_rate)))
    n_blks = int(math.ceil(n / blk))
    pad = n_blks * blk - n
    linked_sq = np.mean(x * x, axis=1)
    if pad > 0:
        linked_sq = np.pad(linked_sq, (0, pad), mode="edge")
    rms_blk = np.sqrt(np.mean(linked_sq.reshape(n_blks, blk), axis=1))

    att_a = math.exp(-float(blk) / max(1.0, attack_ms * 1e-3 * sample_rate))
    rel_a = math.exp(-float(blk) / max(1.0, release_ms * 1e-3 * sample_rate))
    env = np.empty_like(rms_blk)
    cur = float(rms_blk[0])
    for i in range(n_blks):
        v = float(rms_blk[i])
        coeff = att_a if v > cur else rel_a
        cur = coeff * cur + (1.0 - coeff) * v
        env[i] = cur

    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    half_k = 0.5 * knee_db
    slope = (1.0 / max(1.0, ratio)) - 1.0

    gr_db = np.zeros_like(env_db)
    above = env_db >= (threshold_db + half_k)
    in_knee = (env_db > (threshold_db - half_k)) & (~above)
    gr_db[above] = slope * (env_db[above] - threshold_db)
    if knee_db > 1e-4:
        gr_db[in_knee] = slope * ((env_db[in_knee] - threshold_db + half_k) ** 2) / (2.0 * knee_db)

    gain_blk = 10.0 ** (gr_db / 20.0)
    blk_centers = (np.arange(n_blks, dtype=np.float64) + 0.5) * blk
    gain_sample = np.interp(np.arange(n, dtype=np.float64), blk_centers, gain_blk)

    y = (x * gain_sample[:, np.newaxis]).astype(np.float32)
    return y[:, 0] if orig_1d else y


# ---------------------------------------------------------------------------
# 出力品質メトリクス計測＆総合マスタリング
# ---------------------------------------------------------------------------


def measure_audio_metrics(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, float]:
    """ステレオまたはマルチチャンネル音声の品質メトリクスを計測する。

    Note:
        `lr_balance_db = rms_r_db - rms_l_db`（負=L優勢, 正=R優勢, 0=中央均衡 — 方位角規約と符号一致）。
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr_2d = np.stack([arr, arr], axis=1)
    else:
        arr_2d = arr

    lufs = measure_lufs(arr_2d, sample_rate=sample_rate)
    tp_dbtp = measure_true_peak_dbtp(arr_2d, oversample=4)

    left = arr_2d[:, 0]
    right = arr_2d[:, 1] if arr_2d.shape[1] >= 2 else arr_2d[:, 0]

    rms_l = float(np.sqrt(np.mean(left * left))) if left.size else 0.0
    rms_r = float(np.sqrt(np.mean(right * right))) if right.size else 0.0
    rms_l_db = float(20.0 * math.log10(max(rms_l, 1e-12)))
    rms_r_db = float(20.0 * math.log10(max(rms_r, 1e-12)))
    lr_balance_db = float(rms_r_db - rms_l_db)

    denom = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
    corr = float(np.sum(left * right) / denom) if denom > 1e-15 else 0.0
    corr = max(-1.0, min(1.0, corr))

    return {
        "integrated_lufs": round(lufs, 2),
        "true_peak_dbtp": round(tp_dbtp, 2),
        "rms_l_db": round(rms_l_db, 2),
        "rms_r_db": round(rms_r_db, 2),
        "lr_balance_db": round(lr_balance_db, 2),
        "correlation": round(corr, 4),
    }


def master_audio(
    audio: np.ndarray,
    target_lufs: float = -18.0,
    ceiling_dbtp: float = -1.0,
    sample_rate: int = SAMPLE_RATE,
    enable_compressor: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """リンクコンプレッサー → BS.1770-4 ラウドネス正規化 → 4x True Peak リミッターの一貫マスタリングを実行する。"""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or float(np.max(np.abs(arr))) < 1e-9:
        metrics = measure_audio_metrics(arr, sample_rate=sample_rate)
        metrics["applied_gain_db"] = 0.0
        return arr.copy(), metrics

    y = arr.copy()
    if enable_compressor:
        y = linked_bus_compressor(y, sample_rate=sample_rate, threshold_db=-14.0, ratio=1.8, knee_db=5.0)

    total_gain_db = 0.0
    for _iter in range(3):
        cur_lufs = measure_lufs(y, sample_rate=sample_rate)
        if cur_lufs <= -69.5:
            break
        diff_db = float(target_lufs) - cur_lufs
        if abs(diff_db) <= 0.15 and measure_true_peak_dbtp(y) <= ceiling_dbtp:
            break
        # 過度なゲイン発散を防ぐ安全クランプ
        step_db = max(-36.0, min(36.0, diff_db))
        y = (y.astype(np.float64) * (10.0 ** (step_db / 20.0))).astype(np.float32)
        total_gain_db += step_db
        y = true_peak_limiter(y, sample_rate=sample_rate, ceiling_dbtp=ceiling_dbtp)

    # 最終安全リミット
    if measure_true_peak_dbtp(y) > ceiling_dbtp:
        y = true_peak_limiter(y, sample_rate=sample_rate, ceiling_dbtp=ceiling_dbtp)

    metrics = measure_audio_metrics(y, sample_rate=sample_rate)
    metrics["applied_gain_db"] = round(total_gain_db, 2)
    return y, metrics

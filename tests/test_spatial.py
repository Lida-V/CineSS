"""CineSS 空間音響エンジン数値検証テストスイート.

検証項目:
1. 左右定位検証（-60° -> L>R +3dB以上 & ITD先行、+60° -> R>L +3dB以上）
   - Dolby Atmos 互換 / Ambisonics 1次 FOA (4ch) / 3次 HOA (16ch) / 5次 HOA (36ch) / 7次 HOA (64ch) すべて
2. 球面調和関数直交性・加算定理・Max-rE デコード厳密性（1〜7次全次数）
3. 距離モデリング（1.0m vs 8.0m で音量差・高域減衰差・残響比差が物理的に正しいこと）
4. 動的移動（左→右）でクリックノイズなく滑らかに前半 L 優勢 → 後半 R 優勢へ遷移すること
5. SOFA 実測読み込み（h5py 合成 SOFA）と未検出時フォールバック、48kHz 自動リサンプル
6. ITU-R BS.2076 ADM BWF 書き出し・`validate_adm_bwf` 合格・`soundfile` 読込互換・RF64 自動出力
7. マスタリング後の LUFS 誤差 ±1.0 LU 以内、True Peak ≤ -0.9 dBTP、左右バランス非破壊性
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from app.spatial import (
    HRTFEngine,
    SAMPLE_RATE,
    _prepare_stem_overrides,
    acn_sn3d_gains,
    ambisonics_channel_count,
    build_max_re_decoder_matrix,
    compute_dialog_ducking_gain,
    interpolate_keyframe_duck_exempt,
    interpolate_keyframe_gain_db,
    master_audio,
    measure_audio_metrics,
    measure_lufs,
    measure_true_peak_dbtp,
    process_dialog_stem,
    render_spatial_mix,
    validate_adm_bwf,
    write_adm_bwf,
    write_spatial_wav,
)
from app.spatial.ambisonics import AmbisonicsRenderer, max_re_weights
from app.spatial.atmos import AtmosRenderer
from app.spatial.hrtf import (
    air_absorption_cutoff_hz,
    distance_direct_gain,
    distance_reverb_send,
)
from app.spatial.mastering import _measure_lufs_builtin


def _make_speech_like_signal(duration_sec: float, sr: int = SAMPLE_RATE, seed: int = 42) -> np.ndarray:
    """音声帯域（250Hz〜6kHz）の広帯域ピンク風テスト信号を生成する。"""
    n = int(round(duration_sec * sr))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    sos = signal.butter(3, [250.0, 6000.0], btype="bandpass", fs=sr, output="sos")
    sig = signal.sosfilt(sos, white)
    # わずかな音節変調を加えてリアルな音声包絡に近づける
    t = np.arange(n, dtype=np.float64) / float(sr)
    env = 0.65 + 0.35 * np.sin(2.0 * math.pi * 3.5 * t)
    sig *= env
    sig /= max(1e-9, float(np.max(np.abs(sig))))
    return (0.45 * sig).astype(np.float32)


def _measure_lr_diff_db(stereo: np.ndarray) -> float:
    """L - R の RMS レベル差 (dB) を返す（正なら L > R, 負なら R > L）。"""
    rms_l = float(np.sqrt(np.mean(stereo[:, 0].astype(np.float64) ** 2)))
    rms_r = float(np.sqrt(np.mean(stereo[:, 1].astype(np.float64) ** 2)))
    return float(20.0 * math.log10(max(rms_l, 1e-12) / max(rms_r, 1e-12)))


def _estimate_itd_lead_samples(stereo: np.ndarray, max_lag: int = 64) -> int:
    """相互相関により L チャンネルが R チャンネルより何サンプル先行しているか推定する。

    正の値: L が R より先行（左音源）。
    負の値: R が L より先行（右音源）。
    """
    # 低中域 (200Hz〜1800Hz) に絞ることで明確な ITD ピークを抽出
    sos = signal.butter(3, [200.0, 1800.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    l_f = signal.sosfilt(sos, stereo[:, 0].astype(np.float64))
    r_f = signal.sosfilt(sos, stereo[:, 1].astype(np.float64))

    # 相関評価窓（過渡部を除く中央部）
    n = len(l_f)
    s0 = min(n // 5, 4800)
    s1 = max(s0 + 4096, n - s0)
    seg_l = l_f[s0:s1]
    seg_r = r_f[s0:s1]

    best_lag = 0
    best_corr = -1e18
    for lag in range(-max_lag, max_lag + 1):
        # lag > 0: r(t + lag) ≈ l(t) → L が R より lag サンプル早く到達
        if lag > 0:
            c = float(np.dot(seg_l[:-lag], seg_r[lag:]))
        elif lag < 0:
            k = -lag
            c = float(np.dot(seg_l[k:], seg_r[:-k]))
        else:
            c = float(np.dot(seg_l, seg_r))
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag


class TestSpatialEngine(unittest.TestCase):
    """CineSS 空間音響エンジンの包括的数値検証テスト。"""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _write_wav(self, name: str, data: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
        p = self.tmpdir / name
        sf.write(str(p), data, sr, subtype="PCM_24")
        return p

    # ------------------------------------------------------------------
    # 1. Ambisonics 1〜7次 数学的厳密性（SN3D 加算定理 & Max-rE デコード）
    # ------------------------------------------------------------------
    def test_ambisonics_sh_addition_theorem_and_decoder_orders_1_to_7(self) -> None:
        rng = np.random.default_rng(123)
        az = rng.uniform(-180.0, 180.0, size=32)
        el = rng.uniform(-90.0, 90.0, size=32)

        y7 = acn_sn3d_gains(az, el, order=7)
        self.assertEqual(y7.shape, (32, 64))

        # 各次数 l=0..7 で Σ_{m=-l}^l (Y_l^m)^2 == 1.0 を検証
        for l in range(8):
            sl = y7[:, l * l : (l + 1) * (l + 1)]
            norm_sq = np.sum(sl * sl, axis=1)
            np.testing.assert_allclose(norm_sq, np.ones(32), atol=1e-12)

        # 左真横 (-90°) は AmbiX +Y (ACN 1) == +1.0, 右真横 (+90°) は -1.0
        y_left = acn_sn3d_gains(-90.0, 0.0, order=1)
        y_right = acn_sn3d_gains(90.0, 0.0, order=1)
        self.assertAlmostEqual(float(y_left[1]), 1.0, places=12)
        self.assertAlmostEqual(float(y_right[1]), -1.0, places=12)

        # 全次数 1..7 でチャンネル数と Max-rE モード整合性を検証
        for order in range(1, 8):
            n_ch = ambisonics_channel_count(order)
            self.assertEqual(n_ch, (order + 1) ** 2)
            dec, az_s, el_s = build_max_re_decoder_matrix(order)
            y_spk = acn_sn3d_gains(az_s, el_s, order=order)
            reconstructed = y_spk.T @ dec  # (C, C)
            a_l = max_re_weights(order)
            expected_diag = np.concatenate([np.full(2 * l + 1, a_l[l]) for l in range(order + 1)])
            np.testing.assert_allclose(reconstructed, np.diag(expected_diag), atol=1e-10)

    # ------------------------------------------------------------------
    # 2. 左右定位検証（Atmos / FOA / HOA3 / HOA5 / HOA7 すべて）
    # ------------------------------------------------------------------
    def test_left_right_localization_all_formats(self) -> None:
        sig = _make_speech_like_signal(1.2, seed=7)
        src_wav = self._write_wav("voice.wav", sig)

        formats = [
            ("atmos", 3, 13),            # 12ch Bed + 1 Dynamic Object
            ("ambisonics_foa", 1, 4),    # 1次 4ch
            ("ambisonics_hoa", 3, 16),   # 3次 16ch
            ("ambisonics_hoa5", 5, 36),  # 5次 36ch
            ("ambisonics_hoa7", 7, 64),  # 7次 64ch
        ]

        for fmt, order, expected_ch in formats:
            with self.subTest(spatial_format=fmt):
                # 左音源 (-60°)
                scene_left = {
                    "duration_sec": 1.2,
                    "sample_rate": SAMPLE_RATE,
                    "room": {"rt60_sec": 0.35, "size": "small"},
                    "objects": [
                        {
                            "id": "spk_left",
                            "label": "左話者",
                            "role": "dialog",
                            "audio_path": str(src_wav),
                            "is_bed": False,
                            "gain_db": 0.0,
                            "spread": 0.05,
                            "keyframes": [{"t": 0.0, "azimuth_deg": -60.0, "elevation_deg": 0.0, "distance_m": 1.2}],
                        }
                    ],
                }
                res_l = render_spatial_mix(
                    scene_left,
                    out_dir=str(self.tmpdir / f"out_{fmt}_left"),
                    spatial_format=fmt,
                    ambisonics_order=order,
                    target_lufs=-18.0,
                )
                bin_l, sr_l = sf.read(res_l["binaural_wav"], dtype="float32")
                self.assertEqual(sr_l, SAMPLE_RATE)
                self.assertEqual(res_l["metrics"]["channels"], expected_ch)
                master_info = sf.info(res_l["master_spatial_wav"])
                self.assertEqual(master_info.channels, expected_ch)

                diff_l_db = _measure_lr_diff_db(bin_l)
                itd_l = _estimate_itd_lead_samples(bin_l)
                self.assertGreaterEqual(
                    diff_l_db,
                    3.0,
                    f"{fmt}: 左音源 (-60°) で L > R が +3dB 未満 ({diff_l_db:.2f} dB)",
                )
                self.assertGreater(
                    itd_l,
                    0,
                    f"{fmt}: 左音源 (-60°) で左耳 ITD 先行が確認できません (lag={itd_l})",
                )
                self.assertLessEqual(res_l["metrics"]["binaural_lr_balance_db"], -3.0)

                # 右音源 (+60°)
                scene_right = {
                    "duration_sec": 1.2,
                    "sample_rate": SAMPLE_RATE,
                    "room": {"rt60_sec": 0.35, "size": "small"},
                    "objects": [
                        {
                            "id": "spk_right",
                            "label": "右話者",
                            "role": "dialog",
                            "audio_path": str(src_wav),
                            "is_bed": False,
                            "gain_db": 0.0,
                            "spread": 0.05,
                            "keyframes": [{"t": 0.0, "azimuth_deg": 60.0, "elevation_deg": 0.0, "distance_m": 1.2}],
                        }
                    ],
                }
                res_r = render_spatial_mix(
                    scene_right,
                    out_dir=str(self.tmpdir / f"out_{fmt}_right"),
                    spatial_format=fmt,
                    ambisonics_order=order,
                    target_lufs=-18.0,
                )
                bin_r, _ = sf.read(res_r["binaural_wav"], dtype="float32")
                diff_r_db = _measure_lr_diff_db(bin_r)  # L - R なので負が R > L
                itd_r = _estimate_itd_lead_samples(bin_r)
                self.assertLessEqual(
                    diff_r_db,
                    -3.0,
                    f"{fmt}: 右音源 (+60°) で R > L が +3dB 未満 (L-R={diff_r_db:.2f} dB)",
                )
                self.assertLess(
                    itd_r,
                    0,
                    f"{fmt}: 右音源 (+60°) で右耳 ITD 先行が確認できません (lag={itd_r})",
                )
                self.assertGreaterEqual(res_r["metrics"]["binaural_lr_balance_db"], 3.0)

                # Atmos のみ 7.1.4 12ch WAV 確認
                if fmt == "atmos":
                    self.assertIsNotNone(res_l["surround_714_wav"])
                    s714_info = sf.info(res_l["surround_714_wav"])
                    self.assertEqual(s714_info.channels, 12)
                else:
                    self.assertIsNone(res_l["surround_714_wav"])

    # ------------------------------------------------------------------
    # 3. 距離モデリング検証（1.0m vs 8.0m: 音量差・高域減衰差・リバーブ比差）
    # ------------------------------------------------------------------
    def test_distance_modeling_near_vs_far(self) -> None:
        # 0.35秒の広帯域バースト＋1.25秒の無音（残響テールを正確に計測するため）
        rng = np.random.default_rng(99)
        burst_n = int(0.35 * SAMPLE_RATE)
        total_n = int(1.60 * SAMPLE_RATE)
        burst = rng.standard_normal(burst_n)
        sos_bp = signal.butter(2, [200.0, 12000.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
        burst = signal.sosfilt(sos_bp, burst)
        # 立ち上がり/立ち下がりを 10ms フェード
        fade = int(0.01 * SAMPLE_RATE)
        burst[:fade] *= np.linspace(0.0, 1.0, fade)
        burst[-fade:] *= np.linspace(1.0, 0.0, fade)
        burst = (0.5 * burst / np.max(np.abs(burst))).astype(np.float32)

        full_sig = np.zeros(total_n, dtype=np.float32)
        full_sig[:burst_n] = burst
        src_wav = self._write_wav("burst.wav", full_sig)

        renderer = AtmosRenderer(hrtf_engine=HRTFEngine(), sample_rate=SAMPLE_RATE)

        def _render_at_dist(dist_m: float) -> np.ndarray:
            scene = {
                "duration_sec": 1.60,
                "sample_rate": SAMPLE_RATE,
                "room": {"rt60_sec": 0.60, "size": "medium"},
                "objects": [
                    {
                        "id": "src",
                        "label": "距離テスト",
                        "role": "sfx",
                        "audio_path": str(src_wav),
                        "is_bed": False,
                        "gain_db": 0.0,
                        "spread": 0.05,
                        "keyframes": [{"t": 0.0, "azimuth_deg": -25.0, "elevation_deg": 0.0, "distance_m": dist_m}],
                    }
                ],
            }
            return renderer.render_scene(scene)["binaural"]

        near_bin = _render_at_dist(1.0)
        far_bin = _render_at_dist(8.0)

        # (a) 全体音量差: 近距離 (1.0m) は遠距離 (8.0m) より明確に大音量 (+6 dB 以上)
        rms_near = float(np.sqrt(np.mean(near_bin.astype(np.float64) ** 2)))
        rms_far = float(np.sqrt(np.mean(far_bin.astype(np.float64) ** 2)))
        level_diff_db = 20.0 * math.log10(rms_near / max(rms_far, 1e-12))
        self.assertGreater(level_diff_db, 6.0, f"近距離/遠距離の音量差が不十分: {level_diff_db:.2f} dB")

        # (b) 高域減衰差（空気吸収 LPF + 室内高域減衰）: 直接区間 [0.05s, 0.30s] の HF/LF 比を比較
        d0, d1 = int(0.05 * SAMPLE_RATE), int(0.30 * SAMPLE_RATE)
        sos_lf = signal.butter(3, [250.0, 1200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
        sos_hf = signal.butter(3, [5000.0, 11000.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")

        def _hf_lf_ratio_db(seg: np.ndarray) -> float:
            mono = 0.5 * (seg[:, 0] + seg[:, 1])
            e_lf = float(np.mean(signal.sosfilt(sos_lf, mono) ** 2))
            e_hf = float(np.mean(signal.sosfilt(sos_hf, mono) ** 2))
            return 10.0 * math.log10(max(e_hf, 1e-15) / max(e_lf, 1e-15))

        hf_ratio_near = _hf_lf_ratio_db(near_bin[d0:d1])
        hf_ratio_far = _hf_lf_ratio_db(far_bin[d0:d1])
        self.assertGreater(
            hf_ratio_near - hf_ratio_far,
            3.5,
            f"遠距離での高域空気吸収減衰が不十分: near={hf_ratio_near:.2f}dB, far={hf_ratio_far:.2f}dB",
        )

        # (c) リバーブ比（残響テール区間 [0.42s, 1.00s] / 直接区間 [0.05s, 0.30s]）が遠距離で大幅に増加
        t0, t1 = int(0.42 * SAMPLE_RATE), int(1.00 * SAMPLE_RATE)

        def _reverb_to_direct_db(sig_2ch: np.ndarray) -> float:
            e_dir = float(np.mean(sig_2ch[d0:d1].astype(np.float64) ** 2))
            e_tail = float(np.mean(sig_2ch[t0:t1].astype(np.float64) ** 2))
            return 10.0 * math.log10(max(e_tail, 1e-15) / max(e_dir, 1e-15))

        rtod_near = _reverb_to_direct_db(near_bin)
        rtod_far = _reverb_to_direct_db(far_bin)
        self.assertGreater(
            rtod_far - rtod_near,
            10.0,
            f"遠距離での残響比率増加が不十分: near={rtod_near:.2f}dB, far={rtod_far:.2f}dB",
        )

    # ------------------------------------------------------------------
    # 4. 動的移動（左→右）の連続定位＆クリックノイズ不在検証
    # ------------------------------------------------------------------
    def test_dynamic_motion_left_to_right_click_free(self) -> None:
        dur = 2.0
        n = int(dur * SAMPLE_RATE)
        t = np.arange(n, dtype=np.float64) / float(SAMPLE_RATE)
        # 滑らかな複合正弦波（クリック・ブロック境界不連続があれば 2階微分に即座にスパイクが出る）
        tone = (0.35 * np.sin(2.0 * math.pi * 660.0 * t) + 0.20 * np.sin(2.0 * math.pi * 1320.0 * t)).astype(np.float32)
        src_wav = self._write_wav("sweep_tone.wav", tone)

        scene = {
            "duration_sec": dur,
            "sample_rate": SAMPLE_RATE,
            "room": {"rt60_sec": 0.30, "size": "small"},
            "objects": [
                {
                    "id": "mover",
                    "label": "移動音源",
                    "role": "sfx",
                    "audio_path": str(src_wav),
                    "is_bed": False,
                    "gain_db": 0.0,
                    "spread": 0.0,
                    "keyframes": [
                        {"t": 0.0, "azimuth_deg": -70.0, "elevation_deg": 0.0, "distance_m": 1.4},
                        {"t": 2.0, "azimuth_deg": 70.0, "elevation_deg": 0.0, "distance_m": 1.4},
                    ],
                }
            ],
        }

        for fmt in ("atmos", "ambisonics_hoa"):
            with self.subTest(spatial_format=fmt):
                res = render_spatial_mix(
                    scene,
                    out_dir=str(self.tmpdir / f"motion_{fmt}"),
                    spatial_format=fmt,
                    target_lufs=-18.0,
                )
                y, _ = sf.read(res["binaural_wav"], dtype="float64")
                q1 = y[int(0.10 * SAMPLE_RATE) : int(0.45 * SAMPLE_RATE)]
                q4 = y[int(1.55 * SAMPLE_RATE) : int(1.90 * SAMPLE_RATE)]

                self.assertGreater(_measure_lr_diff_db(q1), 3.0, f"{fmt}: 移動前半で L 優勢になっていません")
                self.assertLess(_measure_lr_diff_db(q4), -3.0, f"{fmt}: 移動後半で R 優勢になっていません")

                # 2階微分スパイク検出（定常正弦波の理論最大2階微分に対し異常クリックがないこと）
                interior = y[int(0.15 * SAMPLE_RATE) : int(1.85 * SAMPLE_RATE)]
                d2 = np.diff(interior, n=2, axis=0)
                rms_d2 = float(np.sqrt(np.mean(d2 * d2)))
                peak_d2 = float(np.max(np.abs(d2)))
                crest_d2 = peak_d2 / max(rms_d2, 1e-12)
                self.assertLess(
                    crest_d2,
                    8.0,
                    f"{fmt}: 動的移動中にクリックノイズ様の 2階微分スパイクを検出 (crest={crest_d2:.2f})",
                )

    # ------------------------------------------------------------------
    # 5. SOFA 読み込み・フォールバック・自動リサンプル検証
    # ------------------------------------------------------------------
    def test_sofa_loading_fallback_and_auto_resample(self) -> None:
        # 存在しない SOFA は自動で物理モデルへフォールバック
        eng_fb = HRTFEngine(sofa_path=self.tmpdir / "missing.sofa")
        self.assertEqual(eng_fb.source_name, "physical_woodworth_brown_duda")
        self.assertFalse(eng_fb.diffuse_field_eq_applied)
        self.assertIsNotNone(eng_fb.fallback_reason)

        # 合成 SimpleFreeFieldHRIR SOFA (h5py 利用可能時)
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError:
            h5py = None

        if h5py is not None:
            sofa_path = self.tmpdir / "mock_kemar.sofa"
            # 2測定点: SOFA az=+60° (左 = CineSS -60°) で左耳大, SOFA az=300° (-60°, 右 = CineSS +60°) で右耳大
            ir_len = 128
            irs = np.zeros((2, 2, ir_len), dtype=np.float32)
            irs[0, 0, 20] = 0.90  # 左音源 → 左耳強・早
            irs[0, 1, 38] = 0.25  # 左音源 → 右耳弱・遅
            irs[1, 0, 38] = 0.25  # 右音源 → 左耳弱・遅
            irs[1, 1, 20] = 0.90  # 右音源 → 右耳強・早
            pos = np.array([[60.0, 0.0, 1.4], [300.0, 0.0, 1.4]], dtype=np.float64)
            with h5py.File(str(sofa_path), "w") as f:
                f.create_dataset("Data.IR", data=irs)
                ds_pos = f.create_dataset("SourcePosition", data=pos)
                ds_pos.attrs["Type"] = "spherical"
                f.create_dataset("Data.SamplingRate", data=np.array([48000.0]))

            eng_sofa = HRTFEngine(sofa_path=sofa_path)
            self.assertEqual(eng_sofa.source_name, "sofa:mock_kemar.sofa")
            self.assertFalse(eng_sofa.diffuse_field_eq_applied)
            hl, hr = eng_sofa.get_hrir(-60.0, 0.0)
            self.assertGreater(float(np.max(np.abs(hl))), float(np.max(np.abs(hr))) * 2.5)

        # 22050Hz 入力の自動 48000Hz リサンプル検証
        sig_22k = _make_speech_like_signal(0.8, sr=22050, seed=11)
        wav_22k = self._write_wav("voice_22k.wav", sig_22k, sr=22050)
        scene = {
            "duration_sec": 0.8,
            "sample_rate": SAMPLE_RATE,
            "objects": [
                {
                    "id": "v22k",
                    "label": "22k話者",
                    "role": "dialog",
                    "audio_path": str(wav_22k),
                    "is_bed": False,
                    "keyframes": [{"t": 0.0, "azimuth_deg": -45.0, "elevation_deg": 0.0, "distance_m": 1.5}],
                }
            ],
        }
        res = render_spatial_mix(scene, out_dir=str(self.tmpdir / "resamp_out"), spatial_format="atmos")
        self.assertEqual(sf.info(res["binaural_wav"]).samplerate, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # 6. ADM BWF 書き出し・validate_adm_bwf・RF64 検証
    # ------------------------------------------------------------------
    def test_adm_bwf_write_validate_and_rf64(self) -> None:
        sig_a = _make_speech_like_signal(1.0, seed=1)
        sig_b = _make_speech_like_signal(1.0, seed=2)
        bgm = np.stack([_make_speech_like_signal(1.0, seed=3) * 0.25, _make_speech_like_signal(1.0, seed=4) * 0.25], axis=1)

        wav_a = self._write_wav("a.wav", sig_a)
        wav_b = self._write_wav("b.wav", sig_b)
        wav_bgm = self._write_wav("bgm.wav", bgm)

        scene = {
            "duration_sec": 1.0,
            "sample_rate": SAMPLE_RATE,
            "room": {"rt60_sec": 0.45, "size": "medium"},
            "objects": [
                {
                    "id": "bgm",
                    "label": "BGM",
                    "role": "music",
                    "audio_path": str(wav_bgm),
                    "is_bed": True,
                    "gain_db": -6.0,
                },
                {
                    "id": "speaker_a",
                    "label": "話者A",
                    "role": "dialog",
                    "audio_path": str(wav_a),
                    "is_bed": False,
                    "keyframes": [
                        {"t": 0.0, "azimuth_deg": -35.0, "elevation_deg": 0.0, "distance_m": 1.5},
                        {"t": 1.0, "azimuth_deg": -15.0, "elevation_deg": 5.0, "distance_m": 1.8},
                    ],
                },
                {
                    "id": "speaker_b",
                    "label": "話者B",
                    "role": "dialog",
                    "audio_path": str(wav_b),
                    "is_bed": False,
                    "keyframes": [{"t": 0.0, "azimuth_deg": 40.0, "elevation_deg": 0.0, "distance_m": 2.0}],
                },
            ],
        }

        res = render_spatial_mix(scene, out_dir=str(self.tmpdir / "adm_out"), spatial_format="atmos")
        adm_path = Path(res["master_spatial_wav"])
        report = validate_adm_bwf(adm_path)
        self.assertTrue(report["valid"], f"ADM BWF 検証エラー: {report['errors']}")
        self.assertEqual(report["container"], "RIFF")
        self.assertEqual(report["sample_rate"], SAMPLE_RATE)
        self.assertEqual(report["channels"], 14)  # 12ch Bed + 2 Dynamic Objects
        self.assertEqual(report["num_tracks_chna"], 14)
        self.assertEqual(report["num_objects"], 3)  # 1 Bed + 2 Dynamic Objects
        self.assertTrue(report["has_bext"])
        self.assertTrue(report["has_axml"])
        self.assertTrue(report["has_chna"])

        # libsndfile / soundfile での正常読み取り互換確認
        sf_info = sf.info(str(adm_path))
        self.assertEqual(sf_info.channels, 14)
        self.assertEqual(sf_info.samplerate, SAMPLE_RATE)

        # RF64 形式 ADM BWF の生成・検証
        rf64_adm = self.tmpdir / "test_rf64_adm.wav"
        bed_dummy = np.zeros((4800, 12), dtype=np.float32)
        dyn_dummy = [{"id": "obj1", "label": "Obj1", "audio": np.zeros(4800, dtype=np.float32), "adm_keyframes": [{"t": 0.0, "x": -0.5, "y": 0.8, "z": 0.0}]}]
        write_adm_bwf(rf64_adm, bed_dummy, dyn_dummy, force_rf64=True)
        rf64_rep = validate_adm_bwf(rf64_adm)
        self.assertTrue(rf64_rep["valid"], f"RF64 ADM BWF 検証エラー: {rf64_rep['errors']}")
        self.assertEqual(rf64_rep["container"], "RF64")
        self.assertEqual(rf64_rep["channels"], 13)

        # Ambisonics WAV のしきい値超過時自動 RF64 切替検証
        rf64_ambi, container = write_spatial_wav(
            self.tmpdir / "ambi_rf64.wav",
            np.zeros((2048, 64), dtype=np.float32),
            rf64_threshold_bytes=1024,
        )
        self.assertEqual(container, "RF64")
        with open(rf64_ambi, "rb") as f:
            self.assertEqual(f.read(4), b"RF64")

    # ------------------------------------------------------------------
    # 7. マスタリング精度検証（LUFS ±1.0 LU & True Peak ≤ -0.9 dBTP & L/R 非破壊）
    # ------------------------------------------------------------------
    def test_mastering_lufs_true_peak_and_balance_preservation(self) -> None:
        raw_l = _make_speech_like_signal(2.0, seed=55)
        raw_r = raw_l * 0.5  # 正確に L が +6.02 dB 優勢なテスト信号
        stereo = np.stack([raw_l, raw_r], axis=1)

        # 内蔵 BS.1770-4 と pyloudnorm の整合性確認
        lufs_auto = measure_lufs(stereo, sample_rate=SAMPLE_RATE)
        lufs_builtin = _measure_lufs_builtin(stereo.astype(np.float64), sample_rate=SAMPLE_RATE)
        self.assertLess(abs(lufs_auto - lufs_builtin), 0.25)

        for target in (-24.0, -18.0, -14.0):
            with self.subTest(target_lufs=target):
                mastered, metrics = master_audio(stereo, target_lufs=target, ceiling_dbtp=-1.0)
                measured_lufs = measure_lufs(mastered, sample_rate=SAMPLE_RATE)
                measured_tp = measure_true_peak_dbtp(mastered, oversample=4)

                self.assertLessEqual(
                    abs(measured_lufs - target),
                    1.0,
                    f"目標 {target} LUFS に対し実測 {measured_lufs:.2f} LUFS (誤差 > 1.0 LU)",
                )
                self.assertLessEqual(
                    measured_tp,
                    -0.9,
                    f"True Peak 超過: {measured_tp:.2f} dBTP (> -0.9 dBTP)",
                )
                # リンクコンプレッサー＆リミッターにより L/R バランス (-6.02 dB) が壊れていないこと
                self.assertAlmostEqual(metrics["lr_balance_db"], -6.02, delta=0.1)

        # 過大入力（+12 dB クリップ寸前信号）でも True Peak <= -0.9 dBTP を厳守
        hot = stereo * 8.0
        hot_mastered, _ = master_audio(hot, target_lufs=-14.0, ceiling_dbtp=-1.0)
        hot_metrics = measure_audio_metrics(hot_mastered, sample_rate=SAMPLE_RATE)
        self.assertLessEqual(hot_metrics["true_peak_dbtp"], -0.9)

    # ------------------------------------------------------------------
    # 8. MIT KEMAR SOFA Diffuse-Field EQ 実測検証
    # ------------------------------------------------------------------
    def test_mit_kemar_sofa_diffuse_field_eq(self) -> None:
        eng_phys = HRTFEngine()
        self.assertFalse(eng_phys.diffuse_field_eq_applied)

        sofa_path = Path(__file__).resolve().parent.parent / "models" / "hrtf" / "mit_kemar_normal_pinna.sofa"
        if not sofa_path.is_file():
            self.skipTest("MIT KEMAR SOFA ファイルが存在しないため実測 DF-EQ テストをスキップします")

        eng = HRTFEngine(sofa_path=sofa_path)
        self.assertTrue(eng.diffuse_field_eq_applied)

        hl0, hr0 = eng.get_hrir(0.0, 0.0)
        corr0 = float(np.dot(hl0, hr0) / (np.linalg.norm(hl0) * np.linalg.norm(hr0) + 1e-12))
        self.assertGreater(corr0, 0.85, f"正面 (0°,0°) 左右相関が不足: {corr0:.4f}")

        n_fft = 4096
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
        sl = np.abs(np.fft.rfft(hl0.astype(np.float64), n=n_fft))
        sr = np.abs(np.fft.rfft(hr0.astype(np.float64), n=n_fft))
        m_lo = (freqs >= 100.0) & (freqs <= 400.0)
        m_hi = (freqs >= 2000.0) & (freqs <= 6000.0)
        m_mid = (freqs >= 200.0) & (freqs <= 4000.0)

        lo_db = 20.0 * math.log10(math.sqrt(float(np.mean(0.5 * (sl[m_lo] ** 2 + sr[m_lo] ** 2)))))
        hi_db = 20.0 * math.log10(math.sqrt(float(np.mean(0.5 * (sl[m_hi] ** 2 + sr[m_hi] ** 2)))))
        mid_db = 20.0 * math.log10(math.sqrt(float(np.mean(0.5 * (sl[m_mid] ** 2 + sr[m_mid] ** 2)))))
        self.assertLessEqual(abs(hi_db - lo_db), 4.0, f"2-6kHz と 100-400Hz 差が過大: {hi_db - lo_db:+.2f} dB")
        self.assertLessEqual(abs(mid_db), 1.0, f"正面 200-4000Hz 正規化誤差が過大: {mid_db:+.2f} dB")

        # 左右定位・ITD 方向性が EQ 後も正常であること
        sig = _make_speech_like_signal(0.8, seed=77)
        bin_l, _ = eng.render_mono_dynamic(sig, [{"t": 0.0, "azimuth_deg": -60.0, "elevation_deg": 0.0, "distance_m": 1.4}])
        bin_r, _ = eng.render_mono_dynamic(sig, [{"t": 0.0, "azimuth_deg": 60.0, "elevation_deg": 0.0, "distance_m": 1.4}])
        self.assertGreaterEqual(_measure_lr_diff_db(bin_l), 3.0)
        self.assertGreater(_estimate_itd_lead_samples(bin_l), 0)
        self.assertLessEqual(_measure_lr_diff_db(bin_r), -3.0)
        self.assertLess(_estimate_itd_lead_samples(bin_r), 0)

    # ------------------------------------------------------------------
    # 9. セリフ処理チェーン（HPF・ボイスコンプレッサー・ディエッサー）検証
    # ------------------------------------------------------------------
    def test_process_dialog_stem_compressor_and_deesser(self) -> None:
        t_hpf = np.arange(SAMPLE_RATE // 2, dtype=np.float64) / float(SAMPLE_RATE)
        low_45 = (0.4 * np.sin(2.0 * math.pi * 45.0 * t_hpf)).astype(np.float32)
        out_low = process_dialog_stem(low_45, enable_compressor=False, enable_deesser=False, presence_db=0.0)
        rms_in_low = math.sqrt(float(np.mean(low_45[2400:] ** 2)))
        rms_out_low = math.sqrt(float(np.mean(out_low[2400:] ** 2)))
        self.assertLess(20.0 * math.log10(rms_out_low / rms_in_low), -8.0)

        # ボイスコンプレッサー: 強弱差縮小 & ゲート (-55dB 以下) 無動作
        n_half = int(0.4 * SAMPLE_RATE)
        t_seg = np.arange(n_half, dtype=np.float64) / float(SAMPLE_RATE)
        loud = 0.50 * np.sin(2.0 * math.pi * 1000.0 * t_seg)
        quiet = 0.04 * np.sin(2.0 * math.pi * 1000.0 * t_seg)
        dyn_sig = np.concatenate([loud, quiet]).astype(np.float32)

        out_comp = process_dialog_stem(dyn_sig, enable_compressor=True, enable_deesser=False, presence_db=0.0)
        out_nocomp = process_dialog_stem(dyn_sig, enable_compressor=False, enable_deesser=False, presence_db=0.0)

        def _dr_db(x: np.ndarray) -> float:
            r1 = math.sqrt(float(np.mean(x[int(0.1 * SAMPLE_RATE) : n_half - 480] ** 2)))
            r2 = math.sqrt(float(np.mean(x[n_half + int(0.15 * SAMPLE_RATE) : -480] ** 2)))
            return 20.0 * math.log10(max(r1, 1e-12) / max(r2, 1e-12))

        self.assertLess(_dr_db(out_comp), _dr_db(out_nocomp) - 1.5)

        tiny_sig = (10.0 ** (-65.0 / 20.0) * np.sin(2.0 * math.pi * 1000.0 * t_seg)).astype(np.float32)
        out_tiny_c = process_dialog_stem(tiny_sig, enable_compressor=True, enable_deesser=False, presence_db=0.0)
        out_tiny_nc = process_dialog_stem(tiny_sig, enable_compressor=False, enable_deesser=False, presence_db=0.0)
        tiny_diff = 20.0 * math.log10(
            math.sqrt(float(np.mean(out_tiny_c[2400:] ** 2))) / max(1e-12, math.sqrt(float(np.mean(out_tiny_nc[2400:] ** 2))))
        )
        self.assertLess(abs(tiny_diff), 0.2)

        # ディエッサー: 7kHz 歯擦音減衰 & 1kHz 声の芯無劣化
        t_ds = np.arange(SAMPLE_RATE, dtype=np.float64) / float(SAMPLE_RATE)
        core_1k = (0.30 * np.sin(2.0 * math.pi * 1000.0 * t_ds)).astype(np.float32)
        sib_7k = (0.35 * np.sin(2.0 * math.pi * 7000.0 * t_ds)).astype(np.float32)

        c_ds = process_dialog_stem(core_1k, enable_compressor=False, enable_deesser=True, presence_db=0.0)
        c_nods = process_dialog_stem(core_1k, enable_compressor=False, enable_deesser=False, presence_db=0.0)
        core_delta_db = 20.0 * math.log10(
            math.sqrt(float(np.mean(c_ds[4800:] ** 2))) / max(1e-12, math.sqrt(float(np.mean(c_nods[4800:] ** 2))))
        )
        self.assertLess(abs(core_delta_db), 0.2)

        s_ds = process_dialog_stem(sib_7k, enable_compressor=False, enable_deesser=True, presence_db=0.0)
        s_nods = process_dialog_stem(sib_7k, enable_compressor=False, enable_deesser=False, presence_db=0.0)
        sib_delta_db = 20.0 * math.log10(
            math.sqrt(float(np.mean(s_ds[4800:] ** 2))) / max(1e-12, math.sqrt(float(np.mean(s_nods[4800:] ** 2))))
        )
        self.assertLess(sib_delta_db, -2.5)

    # ------------------------------------------------------------------
    # 10. サイドチェインダッキング＆render_spatial_mix 統合検証
    # ------------------------------------------------------------------
    def test_dialog_ducking_gain_and_render_spatial_mix(self) -> None:
        n_tot = int(2.0 * SAMPLE_RATE)
        silent = np.zeros(n_tot, dtype=np.float32)
        g_silent = compute_dialog_ducking_gain(silent)
        self.assertEqual(g_silent.dtype, np.float32)
        self.assertEqual(g_silent.shape, (n_tot,))
        np.testing.assert_allclose(g_silent, np.ones(n_tot, dtype=np.float32), atol=1e-6)

        speech = np.zeros(n_tot, dtype=np.float32)
        n_sp = int(0.8 * SAMPLE_RATE)
        speech[:n_sp] = _make_speech_like_signal(0.8, seed=31)
        g_duck = compute_dialog_ducking_gain(speech, duck_db=-3.0)
        self.assertEqual(g_duck.dtype, np.float32)
        mid_speech = float(np.mean(g_duck[int(0.2 * SAMPLE_RATE) : int(0.65 * SAMPLE_RATE)]))
        tail_quiet = float(np.min(g_duck[int(1.65 * SAMPLE_RATE) :]))
        self.assertGreater(mid_speech, 0.65)
        self.assertLess(mid_speech, 0.76)
        self.assertGreaterEqual(tail_quiet, 0.98)

        # render_spatial_mix でダッキングON/OFFを比較
        t_bgm = np.arange(n_tot, dtype=np.float64) / float(SAMPLE_RATE)
        bgm_m = (0.18 * np.sin(2.0 * math.pi * 220.0 * t_bgm) + 0.10 * np.sin(2.0 * math.pi * 440.0 * t_bgm)).astype(
            np.float32
        )
        bgm_stereo = np.stack([bgm_m, bgm_m], axis=1)
        sfx_tone = (0.15 * np.sin(2.0 * math.pi * 880.0 * t_bgm)).astype(np.float32)

        wav_dlg = self._write_wav("duck_dlg.wav", speech)
        wav_bgm = self._write_wav("duck_bgm.wav", bgm_stereo)
        wav_sfx = self._write_wav("duck_sfx.wav", sfx_tone)

        scene = {
            "duration_sec": 2.0,
            "sample_rate": SAMPLE_RATE,
            "room": {"rt60_sec": 0.25, "size": "small"},
            "objects": [
                {"id": "dlg", "role": "dialog", "audio_path": str(wav_dlg), "is_bed": False, "keyframes": [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0}]},
                {"id": "bgm", "role": "music", "audio_path": str(wav_bgm), "is_bed": True, "gain_db": 0.0},
                {"id": "sfx", "role": "sfx", "audio_path": str(wav_sfx), "is_bed": False, "gain_db": -3.0, "keyframes": [{"t": 0.0, "azimuth_deg": 35.0, "elevation_deg": 0.0, "distance_m": 2.5}]},
            ],
        }

        res_on = render_spatial_mix(
            scene,
            out_dir=str(self.tmpdir / "duck_on"),
            spatial_format="atmos",
            target_lufs=-18.0,
            enable_dialog_ducking=True,
        )
        res_off = render_spatial_mix(
            scene,
            out_dir=str(self.tmpdir / "duck_off"),
            spatial_format="atmos",
            target_lufs=-18.0,
            enable_dialog_ducking=False,
        )
        self.assertTrue(res_on["metrics"]["dialog_ducking_applied"])
        self.assertFalse(res_off["metrics"]["dialog_ducking_applied"])
        self.assertEqual(res_on["metrics"]["target_lufs"], -18.0)
        self.assertIn("diffuse_field_eq", res_on["metrics"])

        y_on, _ = sf.read(res_on["binaural_wav"], dtype="float64")
        y_off, _ = sf.read(res_off["binaural_wav"], dtype="float64")

        def _tail_to_speech_ratio_db(sig: np.ndarray) -> float:
            e_sp = float(np.mean(sig[int(0.2 * SAMPLE_RATE) : int(0.7 * SAMPLE_RATE)] ** 2))
            e_tail = float(np.mean(sig[int(1.4 * SAMPLE_RATE) : int(1.9 * SAMPLE_RATE)] ** 2))
            return 10.0 * math.log10(max(e_tail, 1e-15) / max(e_sp, 1e-15))

        # ダッキングON時は発話区間のBGM/SFXが抑えられるため、後半BGM単独区間との相対比が明確に増大する
        self.assertGreater(_tail_to_speech_ratio_db(y_on) - _tail_to_speech_ratio_db(y_off), 1.5)

    # ------------------------------------------------------------------
    # 11. セリフ専用 Perspective 距離カーブ検証（role="dialog" vs "sfx"）
    # ------------------------------------------------------------------
    def test_dialog_perspective_distance_model(self) -> None:
        self.assertAlmostEqual(distance_direct_gain(2.0, role="dialog"), 1.0, places=6)
        self.assertAlmostEqual(distance_direct_gain(1.0, role="dialog"), 1.22, places=6)
        self.assertAlmostEqual(distance_direct_gain(15.0, role="dialog"), 0.38, places=6)
        self.assertGreater(distance_direct_gain(8.0, role="dialog"), distance_direct_gain(8.0, role="sfx"))

        self.assertAlmostEqual(
            distance_reverb_send(4.0, role="dialog"),
            0.55 * distance_reverb_send(4.0, role="sfx"),
            places=6,
        )
        self.assertAlmostEqual(air_absorption_cutoff_hz(2.0, role="dialog"), 20000.0, places=3)
        self.assertLess(air_absorption_cutoff_hz(2.0, role="sfx"), 15000.0)

        sig = _make_speech_like_signal(0.8, seed=88)
        wav = self._write_wav("dist_role.wav", sig)
        eng = HRTFEngine()
        for renderer in (AtmosRenderer(hrtf_engine=eng), AmbisonicsRenderer(order=3, hrtf_engine=eng)):
            def _render_role(r: str) -> np.ndarray:
                sc = {
                    "duration_sec": 0.8,
                    "sample_rate": SAMPLE_RATE,
                    "room": {"rt60_sec": 0.25, "size": "small"},
                    "objects": [
                        {
                            "id": "src",
                            "role": r,
                            "audio_path": str(wav),
                            "is_bed": False,
                            "keyframes": [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 6.0}],
                        }
                    ],
                }
                return renderer.render_scene(sc)["binaural"]

            b_dlg = _render_role("dialog")
            b_sfx = _render_role("sfx")
            rms_dlg = math.sqrt(float(np.mean(b_dlg.astype(np.float64) ** 2)))
            rms_sfx = math.sqrt(float(np.mean(b_sfx.astype(np.float64) ** 2)))
            self.assertGreater(20.0 * math.log10(rms_dlg / max(rms_sfx, 1e-12)), 4.0)

    # ------------------------------------------------------------------
    # 12. キーフレーム gain_db 補間＆duck_exempt マスク検証
    # ------------------------------------------------------------------
    def test_interpolate_keyframe_gain_db_and_duck_exempt(self) -> None:
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

        # 空キーフレームは base_gain_db 配列を返す
        empty_g = interpolate_keyframe_gain_db([], times, base_gain_db=-4.5)
        np.testing.assert_allclose(empty_g, np.full(5, -4.5))
        none_g = interpolate_keyframe_gain_db(None, times, base_gain_db=2.0)
        np.testing.assert_allclose(none_g, np.full(5, 2.0))

        # t_sec / gain_db 線形補間（範囲外は端点保持・重複時刻は後優先）
        kfs = [
            {"t_sec": 0.5, "gain_db": -6.0},
            {"t_sec": 0.5, "gain_db": -2.0},
            {"t_sec": 1.5, "gain_db": 4.0},
        ]
        interp_g = interpolate_keyframe_gain_db(kfs, times, base_gain_db=0.0)
        np.testing.assert_allclose(interp_g, [-2.0, -2.0, 1.0, 4.0, 4.0], atol=1e-9)

        # t キーのみかつ gain_db 欠損時は base_gain_db へフォールバック
        legacy_kfs = [{"t": 0.0, "azimuth_deg": -30.0}, {"t": 2.0, "azimuth_deg": 30.0}]
        fallback_g = interpolate_keyframe_gain_db(legacy_kfs, times, base_gain_db=-3.0)
        np.testing.assert_allclose(fallback_g, np.full(5, -3.0))

        # キーフレーム単位 duck_exempt: [0.5, 1.5) 区間のみ True
        sample_times = np.arange(int(2.0 * SAMPLE_RATE), dtype=np.float64) / float(SAMPLE_RATE)
        ex_kfs = [
            {"t_sec": 0.0, "duck_exempt": False},
            {"t_sec": 0.5, "duck_exempt": True},
            {"t_sec": 1.5, "duck_exempt": False},
        ]
        ex_mask = interpolate_keyframe_duck_exempt(ex_kfs, sample_times, sample_rate=SAMPLE_RATE)
        self.assertLess(float(ex_mask[int(0.20 * SAMPLE_RATE)]), 0.05)
        self.assertGreater(float(ex_mask[int(1.00 * SAMPLE_RATE)]), 0.95)
        self.assertLess(float(ex_mask[int(1.80 * SAMPLE_RATE)]), 0.05)

    # ------------------------------------------------------------------
    # 13. 新規距離モデル（sfx_impact / sfx_sub / ambience）検証
    # ------------------------------------------------------------------
    def test_sfx_impact_sfx_sub_and_ambience_distance_models(self) -> None:
        # sfx_impact: 2.0m で 0 dB、クリップ [0.32, 1.45]、リバーブ 0.70x、2.2m まで空気吸収なし
        self.assertAlmostEqual(distance_direct_gain(2.0, role="sfx_impact"), 1.0, places=6)
        self.assertAlmostEqual(distance_direct_gain(1.0, role="sfx_impact"), 1.45, places=6)
        self.assertAlmostEqual(distance_direct_gain(8.0, role="sfx_impact"), (2.0 / 8.0) ** 0.55, places=6)
        self.assertGreaterEqual(distance_direct_gain(15.0, role="sfx_impact"), 0.32)
        self.assertAlmostEqual(
            distance_reverb_send(3.5, role="sfx_impact"),
            0.70 * distance_reverb_send(3.5, role="sfx"),
            places=6,
        )
        self.assertAlmostEqual(air_absorption_cutoff_hz(2.2, role="sfx_impact"), 20000.0, places=3)
        self.assertAlmostEqual(air_absorption_cutoff_hz(1.5, role="sfx_impact"), 20000.0, places=3)
        self.assertGreater(
            air_absorption_cutoff_hz(6.0, role="sfx_impact"),
            air_absorption_cutoff_hz(6.0, role="sfx"),
        )

        # sfx_sub: 距離にかかわらず常に 1.0 / 0.0 / 20000.0
        for d in (0.5, 2.0, 8.0, 15.0):
            self.assertAlmostEqual(distance_direct_gain(d, role="sfx_sub"), 1.0, places=6)
            self.assertAlmostEqual(distance_reverb_send(d, role="sfx_sub"), 0.0, places=6)
            self.assertAlmostEqual(air_absorption_cutoff_hz(d, role="sfx_sub"), 20000.0, places=3)

        # ambience: 緩やかな距離減衰・広めリバーブ・遠距離での高域ロールオフ
        self.assertGreater(distance_direct_gain(1.0, role="ambience"), distance_direct_gain(8.0, role="ambience"))
        self.assertGreater(distance_direct_gain(8.0, role="ambience"), distance_direct_gain(8.0, role="sfx"))
        self.assertGreater(distance_reverb_send(3.0, role="ambience"), distance_reverb_send(3.0, role="sfx"))
        self.assertLess(air_absorption_cutoff_hz(8.0, role="ambience"), 10000.0)

    # ------------------------------------------------------------------
    # 14. Atmos 7.1.4 新規ルート（sfx_sub LFE / ambience / sfx_impact aux / 配列ゲイン）
    # ------------------------------------------------------------------
    def test_atmos_bed_routing_sfx_sub_ambience_and_impact_lfe(self) -> None:
        renderer = AtmosRenderer(hrtf_engine=HRTFEngine(), sample_rate=SAMPLE_RATE)
        n = SAMPLE_RATE
        t = np.arange(n, dtype=np.float64) / float(SAMPLE_RATE)

        # sfx_sub: 60Hz 低音に対し Front L/R (0.48) + LFE 140Hz LPF (0.88)、他チャンネル 0
        sub_60 = (0.5 * np.sin(2.0 * math.pi * 60.0 * t)).astype(np.float32)
        sub_stereo = np.stack([sub_60, sub_60], axis=1)
        bed_sub = renderer._route_bed_track_714(sub_stereo, gain_db=0.0, role="sfx_sub")
        np.testing.assert_allclose(bed_sub[:, 0], 0.48 * sub_60, atol=1e-5)
        np.testing.assert_allclose(bed_sub[:, 1], 0.48 * sub_60, atol=1e-5)
        for ch in (2, 4, 5, 6, 7, 8, 9, 10, 11):
            np.testing.assert_allclose(bed_sub[:, ch], 0.0, atol=1e-6)

        rms_in_60 = float(np.sqrt(np.mean(sub_60[4800:] ** 2)))
        rms_lfe_60 = float(np.sqrt(np.mean(bed_sub[4800:, 3] ** 2)))
        self.assertAlmostEqual(rms_lfe_60 / rms_in_60, 0.88, delta=0.05)

        # 300Hz 高め音は 140Hz LPF で LFE が大幅減衰し、100Hz は通過することを確認
        sub_100 = (0.5 * np.sin(2.0 * math.pi * 100.0 * t)).astype(np.float32)
        sub_300 = (0.5 * np.sin(2.0 * math.pi * 300.0 * t)).astype(np.float32)
        bed_100 = renderer._route_bed_track_714(sub_100[:, np.newaxis], gain_db=0.0, role="sfx_sub")
        bed_300 = renderer._route_bed_track_714(sub_300[:, np.newaxis], gain_db=0.0, role="sfx_sub")
        rms_lfe_100 = float(np.sqrt(np.mean(bed_100[4800:, 3] ** 2)))
        rms_lfe_300 = float(np.sqrt(np.mean(bed_300[4800:, 3] ** 2)))
        self.assertGreater(20.0 * math.log10(rms_lfe_100 / max(rms_lfe_300, 1e-12)), 20.0)

        # ambience: リア・サイド・ハイトへの広がりが music より高い（ステレオ・モノとも）
        amb_noise = _make_speech_like_signal(1.0, seed=101)
        amb_st = np.stack([amb_noise, _make_speech_like_signal(1.0, seed=102)], axis=1)
        bed_amb = renderer._route_bed_track_714(amb_st, gain_db=0.0, role="ambience")
        bed_mus = renderer._route_bed_track_714(amb_st, gain_db=0.0, role="music")
        surround_ratio_amb = float(np.sum(bed_amb[:, 4:12] ** 2) / max(1e-12, np.sum(bed_amb[:, 0:2] ** 2)))
        surround_ratio_mus = float(np.sum(bed_mus[:, 4:12] ** 2) / max(1e-12, np.sum(bed_mus[:, 0:2] ** 2)))
        self.assertGreater(surround_ratio_amb, surround_ratio_mus * 2.0)

        bed_amb_mono = renderer._route_bed_track_714(amb_noise[:, np.newaxis], gain_db=0.0, role="ambience")
        self.assertGreater(float(np.mean(bed_amb_mono[:, 10] ** 2)), 1e-6)

        # ベッド配列ゲイン（前半 -12dB -> 後半 +3dB）の適用検証
        gain_curve = np.concatenate([np.full(n // 2, -12.0), np.full(n - n // 2, 3.0)])
        bed_dyn = renderer._route_bed_track_714(sub_stereo, gain_db=gain_curve, role="sfx_sub")
        rms_first = float(np.sqrt(np.mean(bed_dyn[: n // 2 - 2400, 3] ** 2)))
        rms_second = float(np.sqrt(np.mean(bed_dyn[n // 2 + 4800 :, 3] ** 2)))
        self.assertAlmostEqual(20.0 * math.log10(rms_second / max(rms_first, 1e-12)), 15.0, delta=0.5)

        # _pan_dynamic_object_714: sfx_impact に LFE aux send 0.28
        kfs = [{"t": 0.0, "azimuth_deg": -30.0, "elevation_deg": 0.0, "distance_m": 2.0}]
        pan_imp = renderer._pan_dynamic_object_714(sub_60, kfs, role="sfx_impact")
        pan_sfx = renderer._pan_dynamic_object_714(sub_60, kfs, role="sfx")
        pan_dlg = renderer._pan_dynamic_object_714(sub_60, kfs, role="dialog")
        rms_imp_lfe = float(np.sqrt(np.mean(pan_imp[4800:, 3] ** 2)))
        rms_sfx_lfe = float(np.sqrt(np.mean(pan_sfx[4800:, 3] ** 2)))
        self.assertAlmostEqual(rms_imp_lfe / max(rms_sfx_lfe, 1e-12), 0.28 / 0.20, delta=0.03)
        np.testing.assert_allclose(pan_dlg[:, 3], 0.0, atol=1e-7)

    # ------------------------------------------------------------------
    # 15. Ambisonics sfx_sub (ACN 0 主体) ＆時間変化ゲイン配列検証
    # ------------------------------------------------------------------
    def test_ambisonics_bed_sfx_sub_and_time_varying_gain(self) -> None:
        renderer = AmbisonicsRenderer(order=3, hrtf_engine=HRTFEngine(), sample_rate=SAMPLE_RATE)
        n = SAMPLE_RATE
        t = np.arange(n, dtype=np.float64) / float(SAMPLE_RATE)
        sub_sig = (0.4 * np.sin(2.0 * math.pi * 65.0 * t)).astype(np.float32)
        sub_st = np.stack([sub_sig, sub_sig], axis=1)

        b_sub = renderer.encode_bed_source(sub_st, gain_db=0.0, role="sfx_sub")
        self.assertEqual(b_sub.shape, (n, 16))
        energy_w = float(np.sum(b_sub[:, 0].astype(np.float64) ** 2))
        energy_others = float(np.sum(b_sub[:, 1:].astype(np.float64) ** 2))
        self.assertGreater(energy_w, 10.0)
        self.assertLess(energy_others, 1e-8)

        # 時間変化ゲイン配列とキーフレームゲイン付き render_scene
        gain_arr = np.concatenate([np.full(n // 2, -12.0), np.full(n - n // 2, 0.0)])
        b_dyn = renderer.encode_bed_source(sub_st, gain_db=gain_arr, role="sfx_sub")
        rms_1 = float(np.sqrt(np.mean(b_dyn[: n // 2, 0] ** 2)))
        rms_2 = float(np.sqrt(np.mean(b_dyn[n // 2 :, 0] ** 2)))
        self.assertAlmostEqual(20.0 * math.log10(rms_2 / max(rms_1, 1e-12)), 12.0, delta=0.2)

        tone = (0.3 * np.sin(2.0 * math.pi * 500.0 * t)).astype(np.float32)
        wav_tone = self._write_wav("kf_gain_tone.wav", tone)
        wav_sub = self._write_wav("kf_gain_sub.wav", sub_st)
        scene = {
            "duration_sec": 1.0,
            "sample_rate": SAMPLE_RATE,
            "room": {"rt60_sec": 0.2, "size": "small"},
            "objects": [
                {
                    "id": "sub_bed",
                    "role": "sfx_sub",
                    "audio_path": str(wav_sub),
                    "is_bed": True,
                    "gain_db": 0.0,
                    "keyframes": [
                        {"t_sec": 0.0, "gain_db": -12.0},
                        {"t_sec": 0.5, "gain_db": -12.0},
                        {"t_sec": 0.55, "gain_db": 0.0},
                    ],
                },
                {
                    "id": "imp_obj",
                    "role": "sfx_impact",
                    "audio_path": str(wav_tone),
                    "is_bed": False,
                    "gain_db": 0.0,
                    "keyframes": [
                        {"t_sec": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0, "gain_db": -12.0},
                        {"t_sec": 0.5, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0, "gain_db": -12.0},
                        {"t_sec": 0.55, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0, "gain_db": 0.0},
                    ],
                },
            ],
        }
        rendered = renderer.render_scene(scene)
        bin_out = rendered["binaural"]
        e_early = float(np.sqrt(np.mean(bin_out[int(0.1 * SAMPLE_RATE) : int(0.4 * SAMPLE_RATE)] ** 2)))
        e_late = float(np.sqrt(np.mean(bin_out[int(0.65 * SAMPLE_RATE) : int(0.95 * SAMPLE_RATE)] ** 2)))
        self.assertGreater(20.0 * math.log10(e_late / max(e_early, 1e-12)), 9.0)

    # ------------------------------------------------------------------
    # 16. スマート・インパクト免除ダッキング ＆ duck_exempt / role 別免除検証
    # ------------------------------------------------------------------
    def test_smart_impact_ducking_and_duck_exempt_roles(self) -> None:
        dur = 2.0
        n = int(dur * SAMPLE_RATE)
        speech = _make_speech_like_signal(dur, seed=201)

        # 前半 (0..1.0s) は -28 dBFS（閾値未満）、後半 (1.1..1.5s) は -6 dBFS の衝撃バースト
        t = np.arange(n, dtype=np.float64) / float(SAMPLE_RATE)
        imp_sig = (0.04 * np.sin(2.0 * math.pi * 500.0 * t)).astype(np.float32)
        s_burst0, s_burst1 = int(1.10 * SAMPLE_RATE), int(1.45 * SAMPLE_RATE)
        imp_sig[s_burst0:s_burst1] = (0.65 * np.sin(2.0 * math.pi * 500.0 * t[s_burst0:s_burst1])).astype(np.float32)

        g_smart = compute_dialog_ducking_gain(
            speech,
            sample_rate=SAMPLE_RATE,
            duck_db=-2.0,
            impact_override_mono=imp_sig,
            impact_threshold_db=-16.0,
        )
        # 前半の非インパクト区間はしっかりダッキング (-2.0 dB ≈ 0.794)
        quiet_gain = float(np.mean(g_smart[int(0.30 * SAMPLE_RATE) : int(0.80 * SAMPLE_RATE)]))
        self.assertLess(quiet_gain, 0.84)
        # 衝撃バースト区間はスムーズに 1.0 側へ引き戻される
        burst_gain = float(np.mean(g_smart[int(1.15 * SAMPLE_RATE) : int(1.40 * SAMPLE_RATE)]))
        self.assertGreater(burst_gain, 0.96)

        # _prepare_stem_overrides: sfx_sub / duck_exempt / ambience / キーフレーム duck_exempt
        amb_sig = (0.12 * np.sin(2.0 * math.pi * 300.0 * t)).astype(np.float32)
        sub_sig = (0.25 * np.sin(2.0 * math.pi * 55.0 * t)).astype(np.float32)
        wav_dlg = self._write_wav("sd_dlg.wav", speech)
        wav_sub = self._write_wav("sd_sub.wav", sub_sig)
        wav_imp = self._write_wav("sd_imp.wav", imp_sig)
        wav_amb = self._write_wav("sd_amb.wav", np.stack([amb_sig, amb_sig], axis=1))
        wav_ex_obj = self._write_wav("sd_ex_obj.wav", amb_sig)
        wav_ex_kf = self._write_wav("sd_ex_kf.wav", amb_sig)

        scene = {
            "duration_sec": dur,
            "sample_rate": SAMPLE_RATE,
            "objects": [
                {"id": "dlg", "role": "dialog", "audio_path": str(wav_dlg), "is_bed": False},
                {"id": "sub", "role": "sfx_sub", "audio_path": str(wav_sub), "is_bed": True},
                {"id": "imp", "role": "sfx_impact", "audio_path": str(wav_imp), "is_bed": False},
                {"id": "amb", "role": "ambience", "audio_path": str(wav_amb), "is_bed": True},
                {"id": "ex_obj", "role": "ambience", "audio_path": str(wav_ex_obj), "is_bed": True, "duck_exempt": True},
                {
                    "id": "ex_kf",
                    "role": "sfx",
                    "audio_path": str(wav_ex_kf),
                    "is_bed": False,
                    "keyframes": [
                        {"t_sec": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0, "duck_exempt": False},
                        {"t_sec": 0.8, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0, "duck_exempt": True},
                        {"t_sec": 1.4, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.0, "duck_exempt": False},
                    ],
                },
            ],
        }
        overrides, applied = _prepare_stem_overrides(scene, base_dir=None, sample_rate=SAMPLE_RATE, enable_dialog_ducking=True)
        self.assertTrue(applied)
        # sfx_sub と obj["duck_exempt"]=True は対象外（overrides に含まれない）
        self.assertNotIn("sub", overrides)
        self.assertNotIn("ex_obj", overrides)

        # ambience は穏やかな -2.0 dB 程度のダッキング（約 0.794）
        amb_ratio = float(np.sqrt(np.mean(overrides["amb"][int(0.4 * SAMPLE_RATE) : int(0.9 * SAMPLE_RATE)] ** 2))) / max(
            1e-9, float(np.sqrt(np.mean(amb_sig[int(0.4 * SAMPLE_RATE) : int(0.9 * SAMPLE_RATE)] ** 2)))
        )
        self.assertGreater(amb_ratio, 0.75)
        self.assertLess(amb_ratio, 0.85)

        # sfx_impact は前半ダッキング、後半インパクトで 1.0 近くへ引き戻し
        imp_ov = overrides["imp"][:, 0] if overrides["imp"].ndim == 2 else overrides["imp"]
        imp_ratio_quiet = float(np.sqrt(np.mean(imp_ov[int(0.35 * SAMPLE_RATE) : int(0.85 * SAMPLE_RATE)] ** 2))) / max(
            1e-9, float(np.sqrt(np.mean(imp_sig[int(0.35 * SAMPLE_RATE) : int(0.85 * SAMPLE_RATE)] ** 2)))
        )
        imp_ratio_burst = float(np.sqrt(np.mean(imp_ov[int(1.15 * SAMPLE_RATE) : int(1.40 * SAMPLE_RATE)] ** 2))) / max(
            1e-9, float(np.sqrt(np.mean(imp_sig[int(1.15 * SAMPLE_RATE) : int(1.40 * SAMPLE_RATE)] ** 2)))
        )
        self.assertLess(imp_ratio_quiet, 0.85)
        self.assertGreater(imp_ratio_burst, 0.95)

        # キーフレーム上の duck_exempt=True 区間 [0.8, 1.4) が滑らかに 1.0 へ免除されること
        ex_kf_ov = overrides["ex_kf"][:, 0] if overrides["ex_kf"].ndim == 2 else overrides["ex_kf"]
        ratio_before = float(np.sqrt(np.mean(ex_kf_ov[int(0.30 * SAMPLE_RATE) : int(0.65 * SAMPLE_RATE)] ** 2))) / max(
            1e-9, float(np.sqrt(np.mean(amb_sig[int(0.30 * SAMPLE_RATE) : int(0.65 * SAMPLE_RATE)] ** 2)))
        )
        ratio_exempt = float(np.sqrt(np.mean(ex_kf_ov[int(0.95 * SAMPLE_RATE) : int(1.25 * SAMPLE_RATE)] ** 2))) / max(
            1e-9, float(np.sqrt(np.mean(amb_sig[int(0.95 * SAMPLE_RATE) : int(1.25 * SAMPLE_RATE)] ** 2)))
        )
        self.assertLess(ratio_before, 0.85)
        self.assertGreater(ratio_exempt, 0.96)

    # ------------------------------------------------------------------
    # 17. Atmos render_spatial_mix での SFX 4層・時間変化ゲイン統合検証
    # ------------------------------------------------------------------
    def test_atmos_render_spatial_mix_multilayer_sfx_and_keyframe_gains(self) -> None:
        dur = 1.0
        n = int(dur * SAMPLE_RATE)
        t = np.arange(n, dtype=np.float64) / float(SAMPLE_RATE)
        sub_sig = (0.35 * np.sin(2.0 * math.pi * 60.0 * t)).astype(np.float32)
        imp_sig = (0.30 * np.sin(2.0 * math.pi * 800.0 * t)).astype(np.float32)
        amb_sig = _make_speech_like_signal(dur, seed=303) * 0.25

        wav_sub = self._write_wav("e2e_sub.wav", np.stack([sub_sig, sub_sig], axis=1))
        wav_imp = self._write_wav("e2e_imp.wav", imp_sig)
        wav_amb = self._write_wav("e2e_amb.wav", np.stack([amb_sig, amb_sig], axis=1))

        scene = {
            "duration_sec": dur,
            "sample_rate": SAMPLE_RATE,
            "room": {"rt60_sec": 0.3, "size": "medium"},
            "objects": [
                {
                    "id": "sfx_sub",
                    "role": "sfx_sub",
                    "audio_path": str(wav_sub),
                    "is_bed": True,
                    "gain_db": 0.0,
                    "keyframes": [{"t_sec": 0.0, "gain_db": -6.0}, {"t_sec": 0.5, "gain_db": 0.0}],
                },
                {
                    "id": "sfx_impact_left",
                    "role": "sfx_impact",
                    "audio_path": str(wav_imp),
                    "is_bed": False,
                    "gain_db": 0.0,
                    "keyframes": [
                        {"t_sec": 0.0, "azimuth_deg": -50.0, "elevation_deg": 0.0, "distance_m": 2.0, "gain_db": -6.0},
                        {"t_sec": 0.8, "azimuth_deg": -45.0, "elevation_deg": 0.0, "distance_m": 1.6, "gain_db": 2.0},
                    ],
                },
                {
                    "id": "sfx_ambience",
                    "role": "ambience",
                    "audio_path": str(wav_amb),
                    "is_bed": True,
                    "gain_db": -3.0,
                },
            ],
        }
        res = render_spatial_mix(
            scene,
            out_dir=str(self.tmpdir / "atmos_multilayer"),
            spatial_format="atmos",
            target_lufs=-18.0,
        )
        adm_rep = validate_adm_bwf(Path(res["master_spatial_wav"]))
        self.assertTrue(adm_rep["valid"], f"ADM BWF 検証エラー: {adm_rep['errors']}")
        self.assertEqual(res["metrics"]["channels"], 13)
        bin_sig, _ = sf.read(res["binaural_wav"], dtype="float64")
        self.assertGreater(_measure_lr_diff_db(bin_sig), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# coding: utf-8
"""CineSS 統合パイプライン（映像解析・分離・話者別声分け・Foley・空間Mix・再Mix）自動テスト。

OpenCV + FFmpeg で以下の合成動画を動的に生成して E2E 検証する:
1. 左に顔・右に顔が並び、前半は左の人物の口元が動き低音話者が話し、
   後半は右の人物の口元が動き高音話者が話す「2話者有音動画」（入力音声は完全中央モノラル L==R）。
2. 音声ストリームを持たず、物体が左から右へ横切る「無音モーション動画」。

検証ポイント:
- 映像検査（有音/無音自動判定）
- 人物検出・口元モーション追跡・動き重心抽出
- HunyuanVideo-Foley 未取得時の本番規定エラー＆プレビューフォールバック
- run_job（Dolby Atmos 互換 ADM BWF / Ambisonics HOA）の完走
- 左話者区間ではバイノーラル L > R、右話者区間では R > L になることの数値証明
- remix_job で左右配置を入れ替えた際に L/R 定位が実際に反転することの数値証明
"""
from __future__ import annotations

import copy
import math
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from app.audio.foley_engine import MISSING_FOLEY_MESSAGE, find_hunyuan_foley, generate_foley
from app.audio.speaker_splitter import split_speakers
from app.config import SAMPLE_RATE
from app.pipeline import remix_job, run_job
from app.spatial.adm_bwf import validate_adm_bwf
from app.vision.analyzer import analyze_video_spatial, inspect_video


def _draw_synthetic_face(img: np.ndarray, cx: int, cy: int, radius: int, mouth_open: float = 0.0) -> None:
    """左右対称な顔特徴と開閉アニメーションする口元を持つ合成人物を描画する。"""
    y, x = np.ogrid[: img.shape[0], : img.shape[1]]
    dx = (x - cx) / (radius * 0.82)
    dy = (y - cy) / (radius * 1.05)
    dist2 = dx * dx + dy * dy
    mask = dist2 <= 1.0
    shade = np.clip(1.0 - 0.32 * dist2 - 0.12 * dy, 0.45, 1.0)
    for c, base in enumerate((155, 180, 228)):  # 肌色 BGR
        img[..., c] = np.where(mask, (base * shade).astype(np.uint8), img[..., c])

    # 髪
    hair_mask = (
        (dx * dx + ((y - (cy - radius * 0.56)) / (radius * 0.62)) ** 2 <= 1.0)
        & (y < cy - radius * 0.36)
        & (dist2 <= 1.06)
    )
    img[hair_mask] = (35, 38, 46)

    # 両目・眉
    for side in (-1, 1):
        ex = int(cx + side * radius * 0.30)
        ey = int(cy - radius * 0.12)
        cv2.ellipse(img, (ex, int(ey - radius * 0.16)), (int(radius * 0.18), int(radius * 0.05)), 0, 180, 360, (42, 46, 55), -1)
        cv2.ellipse(img, (ex, ey), (int(radius * 0.16), int(radius * 0.10)), 0, 0, 360, (110, 128, 160), -1)
        cv2.ellipse(img, (ex, ey), (int(radius * 0.12), int(radius * 0.07)), 0, 0, 360, (240, 240, 244), -1)
        cv2.circle(img, (ex, ey), max(2, int(radius * 0.06)), (25, 25, 30), -1)

    # 鼻筋・鼻下の影
    cv2.line(img, (cx, int(cy - radius * 0.14)), (cx, int(cy + radius * 0.15)), (192, 214, 248), max(2, int(radius * 0.07)))
    cv2.ellipse(img, (cx, int(cy + radius * 0.20)), (int(radius * 0.13), int(radius * 0.05)), 0, 0, 180, (105, 120, 152), -1)

    # 口元（開閉度 mouth_open ∈ [0, 1] で大きく変化させる）
    my = int(cy + radius * 0.46)
    mh = max(3, int(radius * (0.04 + 0.20 * mouth_open)))
    mw = int(radius * (0.24 + 0.06 * mouth_open))
    cv2.ellipse(img, (cx, my), (mw, mh), 0, 0, 360, (40, 42, 96), -1)
    if mouth_open > 0.25:
        cv2.ellipse(img, (cx, my), (int(mw * 0.72), int(mh * 0.62)), 0, 0, 360, (18, 15, 24), -1)


def _synth_voice_wave(
    t: np.ndarray,
    f0_hz: float,
    formant_weights: list[tuple[int, float]],
    syllable_hz: float,
) -> np.ndarray:
    """音声帯域に倍音フォルマントと音節変調を持つ合成声波形を生成する。"""
    vibrato = f0_hz * (1.0 + 0.015 * np.sin(2.0 * math.pi * 5.2 * t))
    phase = 2.0 * math.pi * np.cumsum(vibrato) / SAMPLE_RATE
    sig = np.zeros_like(t, dtype=np.float64)
    for harmonic, weight in formant_weights:
        sig += weight * np.sin(harmonic * phase)
    # 音節エンベロープ（はっきりした発話リズム）
    env = 0.55 + 0.45 * np.sin(2.0 * math.pi * syllable_hz * t) ** 2
    sig *= env
    peak = float(np.max(np.abs(sig))) + 1e-9
    return (sig / peak * 0.42).astype(np.float32)


def _encode_frames_to_mp4(frames: list[np.ndarray], fps: int, out_mp4: Path, audio_wav: Path | None = None) -> Path:
    """OpenCV フレーム列と任意WAVを FFmpeg で確実な H.264 MP4 に書き出す。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("テスト実行に ffmpeg が必要です。")
    h, w = frames[0].shape[:2]
    raw_video = out_mp4.with_suffix(".raw.mp4")

    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_wav is not None:
        cmd += ["-i", str(audio_wav)]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"]
    if audio_wav is not None:
        cmd += ["-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-ac", "2", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += [str(raw_video)]

    raw_bytes = b"".join(f.tobytes() for f in frames)
    proc = subprocess.run(cmd, input=raw_bytes, capture_output=True, timeout=120)
    if proc.returncode != 0 or not raw_video.is_file():
        raise RuntimeError(f"テスト動画生成に失敗しました: {proc.stderr.decode('utf-8', errors='replace')}")
    raw_video.replace(out_mp4)
    return out_mp4


def _make_two_speaker_voiced_video(work_dir: Path, duration_sec: float = 4.0, fps: int = 12) -> Path:
    """左顔（前半発話）＆右顔（後半発話）の2話者合成有音動画を生成する。

    入力音声は左右同一（L == R の完全モノラル相当ステレオ）とする。
    これにより、バイノーラル出力で前半 L>R・後半 R>L になれば、
    「映像の顔位置＆口元動き→話者紐付け→空間配置」が真に機能した数値証明になる。
    """
    n_frames = int(round(duration_sec * fps))
    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        t = fi / fps
        img = np.full((360, 640, 3), (42, 46, 56), dtype=np.uint8)
        # 前半 (0.2s..1.85s): 左の人物の口が動く
        left_mouth = 0.5 + 0.5 * math.sin(2.0 * math.pi * 4.0 * t) if 0.2 <= t <= 1.85 else 0.0
        # 後半 (2.15s..3.85s): 右の人物の口が動く
        right_mouth = 0.5 + 0.5 * math.sin(2.0 * math.pi * 4.5 * t) if 2.15 <= t <= 3.85 else 0.0
        _draw_synthetic_face(img, 155, 180, 62, mouth_open=left_mouth)
        _draw_synthetic_face(img, 485, 180, 62, mouth_open=right_mouth)
        frames.append(img)

    n_samples = int(round(duration_sec * SAMPLE_RATE))
    t_all = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    mono = np.zeros(n_samples, dtype=np.float32)

    # 話者1（低声・男性フォルマント帯域 130Hz〜2.0kHz・前半 0.25s..1.80s）
    s1, e1 = int(0.25 * SAMPLE_RATE), int(1.80 * SAMPLE_RATE)
    v1 = _synth_voice_wave(
        t_all[s1:e1] - t_all[s1],
        f0_hz=130.0,
        formant_weights=[(1, 0.70), (2, 0.85), (4, 1.00), (7, 0.80), (11, 0.65), (15, 0.45)],
        syllable_hz=3.5,
    )
    fade = int(0.03 * SAMPLE_RATE)
    v1[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    v1[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    mono[s1:e1] += v1

    # 話者2（高声・女性フォルマント帯域 285Hz〜2.8kHz・後半 2.20s..3.80s）
    s2, e2 = int(2.20 * SAMPLE_RATE), int(3.80 * SAMPLE_RATE)
    v2 = _synth_voice_wave(
        t_all[s2:e2] - t_all[s2],
        f0_hz=285.0,
        formant_weights=[(1, 0.70), (2, 0.95), (3, 1.00), (5, 0.85), (7, 0.65), (9, 0.45)],
        syllable_hz=4.2,
    )
    v2[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    v2[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    mono[s2:e2] += v2

    # 入力は左右同一（L == R）＋ごく控えめな低域ステレオパッド
    pad_l = (0.015 * np.sin(2.0 * math.pi * 72.0 * t_all)).astype(np.float32)
    pad_r = (0.015 * np.cos(2.0 * math.pi * 72.0 * t_all)).astype(np.float32)
    stereo = np.stack([mono + pad_l, mono + pad_r], axis=1)

    wav_path = work_dir / "two_speaker_src.wav"
    sf.write(str(wav_path), stereo, SAMPLE_RATE, subtype="FLOAT")
    return _encode_frames_to_mp4(frames, fps, work_dir / "two_speaker_voiced.mp4", audio_wav=wav_path)


def _make_silent_moving_video(work_dir: Path, duration_sec: float = 3.0, fps: int = 12) -> Path:
    """音声なし・青い移動物体が左から右へ横切る無音動画を生成する。"""
    n_frames = int(round(duration_sec * fps))
    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        frac = fi / max(n_frames - 1, 1)
        bg_val = 28 if frac < 0.5 else 36
        img = np.full((360, 640, 3), (bg_val + 10, bg_val, bg_val), dtype=np.uint8)
        cx = int(70 + frac * 500)
        cy = int(180 + 40 * math.sin(2.0 * math.pi * frac))
        cv2.circle(img, (cx, cy), 34, (230, 190, 40), -1)  # シアン系（非肌色）
        cv2.circle(img, (cx, cy), 14, (255, 250, 220), -1)
        frames.append(img)
    return _encode_frames_to_mp4(frames, fps, work_dir / "silent_motion.mp4", audio_wav=None)


def _lr_diff_db(wav_path: Path, start_s: float, end_s: float) -> tuple[float, float, float]:
    """指定区間の L/R RMS (dBFS) と (L - R) dB 差を返す。正なら L 優勢、負なら R 優勢。"""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    s = max(0, int(round(start_s * sr)))
    e = min(len(audio), int(round(end_s * sr)))
    seg = audio[s:e]
    rms_l = float(np.sqrt(np.mean(seg[:, 0].astype(np.float64) ** 2))) + 1e-12
    rms_r = float(np.sqrt(np.mean(seg[:, 1].astype(np.float64) ** 2))) + 1e-12
    db_l = 20.0 * math.log10(rms_l)
    db_r = 20.0 * math.log10(rms_r)
    return db_l, db_r, db_l - db_r


class TestCineSSPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="ciness_test_pipeline_")
        cls.root = Path(cls._tmpdir.name)
        cls.voiced_video = _make_two_speaker_voiced_video(cls.root, duration_sec=4.0, fps=12)
        cls.silent_video = _make_silent_moving_video(cls.root, duration_sec=3.0, fps=12)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_01_inspect_video_voiced_and_silent(self) -> None:
        v_info = inspect_video(self.voiced_video)
        self.assertTrue(v_info["has_audio_stream"])
        self.assertFalse(v_info["is_silent"])
        self.assertEqual(v_info["suggested_mode"], "voiced_remix")
        self.assertGreater(v_info["audio_rms_db"], -45.0)
        self.assertEqual(v_info["width"], 640)
        self.assertEqual(v_info["height"], 360)

        s_info = inspect_video(self.silent_video)
        self.assertFalse(s_info["has_audio_stream"])
        self.assertTrue(s_info["is_silent"])
        self.assertEqual(s_info["suggested_mode"], "silent_foley")
        self.assertLess(s_info["audio_rms_db"], -55.0)

    def test_02_analyze_video_spatial_faces_mouth_and_motion(self) -> None:
        res = analyze_video_spatial(self.voiced_video, sample_fps=6.0)
        persons = res["person_tracks"]
        self.assertEqual(len(persons), 2, f"2名の顔が検出されるべきです: {persons}")

        left_p, right_p = persons[0], persons[1]
        self.assertLess(left_p["mean_azimuth_deg"], -15.0)
        self.assertGreater(right_p["mean_azimuth_deg"], 15.0)

        # 前半は左人物の口元活性度が高く、後半は右人物の口元活性度が高い
        left_first = float(np.mean([k["mouth_activity"] for k in left_p["keyframes"] if 0.3 <= k["t"] <= 1.8]))
        left_second = float(np.mean([k["mouth_activity"] for k in left_p["keyframes"] if 2.2 <= k["t"] <= 3.8]))
        right_first = float(np.mean([k["mouth_activity"] for k in right_p["keyframes"] if 0.3 <= k["t"] <= 1.8]))
        right_second = float(np.mean([k["mouth_activity"] for k in right_p["keyframes"] if 2.2 <= k["t"] <= 3.8]))
        self.assertGreater(left_first, left_second * 1.5, f"左人物は前半の口動きが大きいべき: {left_first} vs {left_second}")
        self.assertGreater(right_second, right_first * 1.5, f"右人物は後半の口動きが大きいべき: {right_second} vs {right_first}")

        # 無音動画の動き重心が左→右に追従すること
        s_res = analyze_video_spatial(self.silent_video, sample_fps=6.0)
        self.assertEqual(len(s_res["person_tracks"]), 0, "非顔物体を人物誤検出すべきではありません")
        mt = [m for m in s_res["motion_track"] if m["motion_energy"] > 0.005]
        self.assertGreaterEqual(len(mt), 4)
        self.assertLess(mt[0]["azimuth_deg"], 0.0)
        self.assertGreater(mt[-1]["azimuth_deg"], 0.0)

    def test_03_foley_missing_model_and_preview_fallback(self) -> None:
        s_analysis = analyze_video_spatial(self.silent_video, sample_fps=4.0)
        if not find_hunyuan_foley()["available"]:
            with self.assertRaises(RuntimeError) as ctx:
                generate_foley(
                    self.silent_video,
                    self.root / "foley_should_fail.wav",
                    scene_analysis=s_analysis,
                    allow_preview_fallback=False,
                )
            self.assertIn(MISSING_FOLEY_MESSAGE, str(ctx.exception))

        fb = generate_foley(
            self.silent_video,
            self.root / "foley_preview.wav",
            scene_analysis=s_analysis,
            user_prompt="風と移動音",
            allow_preview_fallback=True,
        )
        self.assertEqual(fb["method"], "motion_synth_preview")
        self.assertTrue(Path(fb["audio_wav"]).is_file())
        self.assertTrue(Path(fb["sfx_mono_wav"]).is_file())
        self.assertTrue(Path(fb["ambience_wav"]).is_file())
        data, sr = sf.read(fb["audio_wav"], dtype="float32")
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertGreater(float(np.sqrt(np.mean(data ** 2))), 1e-3)

    def test_04_run_job_voiced_atmos_localization_and_remix(self) -> None:
        job_dir = self.root / "job_voiced_atmos"
        logs: list[str] = []
        stages: list[str] = []

        def on_prog(pct: float, stage: str, msg: str) -> None:
            stages.append(stage)

        res = run_job(
            job_dir,
            {
                "video_path": str(self.voiced_video),
                "mode": "auto",
                "spatial_format": "atmos",
                "quality_preset": "fast",
                "target_lufs": -18.0,
                "enable_dereverb": False,
                "force_dsp_fallback": True,
            },
            progress_cb=on_prog,
            log_cb=logs.append,
        )

        self.assertTrue(res["ok"])
        self.assertEqual(res["mode"], "voiced")
        self.assertEqual(res["spatial_format"], "atmos")
        self.assertIn("vision", stages)
        self.assertIn("spatial", stages)
        self.assertIn("mux", stages)
        self.assertGreater(len(logs), 0)

        # 全成果物の存在確認
        bin_mp4 = Path(res["binaural_output_mp4"])
        bin_wav = Path(res["binaural_wav"])
        master_wav = Path(res["master_spatial_wav"])
        surr_714 = Path(res["surround_714_wav"])
        orig_wav = Path(res["original_preview_wav"])
        stems_zip = Path(res["stems_zip"])
        for p in (bin_mp4, bin_wav, master_wav, surr_714, orig_wav, stems_zip, job_dir / "scene.json", job_dir / "result.json"):
            self.assertTrue(p.is_file() and p.stat().st_size > 0, f"成果物不足: {p}")

        # ADM BWF 構造検証
        adm_check = validate_adm_bwf(master_wav)
        self.assertTrue(adm_check["valid"], f"ADM BWF検証エラー: {adm_check}")

        # stems.zip の中身確認
        with zipfile.ZipFile(stems_zip, "r") as zf:
            names = zf.namelist()
            self.assertIn("scene.json", names)
            self.assertTrue(any(n.startswith("stems/speaker_") for n in names))

        # 話者↔人物紐付け検証（左話者 < -10°, 右話者 > +10°）
        self.assertEqual(len(res["speakers"]), 2, f"2話者に分離されるべき: {res['speakers']}")
        self.assertLess(res["speakers"][0]["mean_azimuth_deg"], -10.0)
        self.assertGreater(res["speakers"][1]["mean_azimuth_deg"], 10.0)

        # バイノーラル左右定位の数値検証（前半: 左話者 → L > R、後半: 右話者 → R > L）
        _, _, diff_first = _lr_diff_db(bin_wav, 0.35, 1.70)
        _, _, diff_second = _lr_diff_db(bin_wav, 2.30, 3.70)
        self.assertGreater(diff_first, 1.5, f"前半（左話者）は L > R になるべき: L-R={diff_first:.2f} dB")
        self.assertLess(diff_second, -1.5, f"後半（右話者）は R > L になるべき: L-R={diff_second:.2f} dB")

        # --- remix_job 検証: 左右話者の位置を入れ替えて即座に再Mix ---
        scene = copy.deepcopy(json_load := __import__("json").loads((job_dir / "scene.json").read_text(encoding="utf-8")))
        for obj in scene["objects"]:
            if obj["id"] == "speaker_a":
                obj["keyframes"] = [{"t": 0.0, "azimuth_deg": 42.0, "elevation_deg": 0.0, "distance_m": 1.6}]
            elif obj["id"] == "speaker_b":
                obj["keyframes"] = [{"t": 0.0, "azimuth_deg": -42.0, "elevation_deg": 0.0, "distance_m": 1.6}]

        remix_res = remix_job(
            job_dir,
            scene,
            spatial_format="atmos",
            target_lufs=-18.0,
        )
        self.assertTrue(remix_res["ok"])
        remixed_wav = Path(remix_res["binaural_wav"])
        _, _, r_diff_first = _lr_diff_db(remixed_wav, 0.35, 1.70)
        _, _, r_diff_second = _lr_diff_db(remixed_wav, 2.30, 3.70)
        self.assertLess(r_diff_first, -1.5, f"再Mix後は前半が R > L へ反転すべき: L-R={r_diff_first:.2f} dB")
        self.assertGreater(r_diff_second, 1.5, f"再Mix後は後半が L > R へ反転すべき: L-R={r_diff_second:.2f} dB")

    def test_05_run_job_voiced_ambisonics_hoa(self) -> None:
        job_dir = self.root / "job_voiced_hoa"
        res = run_job(
            job_dir,
            {
                "video_path": str(self.voiced_video),
                "mode": "voiced",
                "spatial_format": "ambisonics_hoa",
                "quality_preset": "fast",
                "target_lufs": -18.0,
                "enable_dereverb": False,
                "force_dsp_fallback": True,
            },
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["spatial_format"], "ambisonics_hoa")
        self.assertEqual(res["metrics"]["channels"], 16)
        self.assertTrue(Path(res["binaural_output_mp4"]).is_file())
        self.assertTrue(Path(res["master_spatial_wav"]).is_file())

        bin_wav = Path(res["binaural_wav"])
        _, _, diff_first = _lr_diff_db(bin_wav, 0.35, 1.70)
        _, _, diff_second = _lr_diff_db(bin_wav, 2.30, 3.70)
        self.assertGreater(diff_first, 1.5, f"HOA 前半（左話者）は L > R になるべき: L-R={diff_first:.2f} dB")
        self.assertLess(diff_second, -1.5, f"HOA 後半（右話者）は R > L になるべき: L-R={diff_second:.2f} dB")

        # 仕様追加(#8) & 回帰確認: request.json に旧次数(order=3)が残っていても
        # remix_job で ambisonics_order 省略時に ambisonics_hoa5(36ch) / ambisonics_hoa7(64ch) へ正しく切り替わること
        _json = __import__("json")
        (job_dir / "request.json").write_text(
            _json.dumps({"video_path": str(self.voiced_video), "spatial_format": "ambisonics_hoa", "ambisonics_order": 3}),
            encoding="utf-8",
        )
        scene = _json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
        remix_hoa5 = remix_job(job_dir, scene, spatial_format="ambisonics_hoa5", target_lufs=-18.0)
        self.assertEqual(remix_hoa5["spatial_format"], "ambisonics_hoa5")
        self.assertEqual(remix_hoa5["ambisonics_order"], 5)
        self.assertEqual(remix_hoa5["metrics"]["channels"], 36)

        remix_hoa7 = remix_job(job_dir, scene, spatial_format="ambisonics_hoa7", target_lufs=-18.0)
        self.assertEqual(remix_hoa7["spatial_format"], "ambisonics_hoa7")
        self.assertEqual(remix_hoa7["ambisonics_order"], 7)
        self.assertEqual(remix_hoa7["metrics"]["channels"], 64)
        _, _, diff_first_7 = _lr_diff_db(Path(remix_hoa7["binaural_wav"]), 0.35, 1.70)
        _, _, diff_second_7 = _lr_diff_db(Path(remix_hoa7["binaural_wav"]), 2.30, 3.70)
        self.assertGreater(diff_first_7, 1.5, f"HOA7 前半（左話者）は L > R になるべき: L-R={diff_first_7:.2f} dB")
        self.assertLess(diff_second_7, -1.5, f"HOA7 後半（右話者）は R > L になるべき: L-R={diff_second_7:.2f} dB")

        # 逆に request.json に旧 order=7 が残っていても ambisonics_hoa(order省略) は既定3次(16ch)に戻ること
        (job_dir / "request.json").write_text(
            _json.dumps({"video_path": str(self.voiced_video), "spatial_format": "ambisonics_hoa7", "ambisonics_order": 7}),
            encoding="utf-8",
        )
        remix_hoa3_again = remix_job(job_dir, scene, spatial_format="ambisonics_hoa", target_lufs=-18.0)
        self.assertEqual(remix_hoa3_again["spatial_format"], "ambisonics_hoa")
        self.assertEqual(remix_hoa3_again["ambisonics_order"], 3)
        self.assertEqual(remix_hoa3_again["metrics"]["channels"], 16)

    def test_06_run_job_silent_foley_e2e(self) -> None:
        job_dir = self.root / "job_silent_foley"
        res = run_job(
            job_dir,
            {
                "video_path": str(self.silent_video),
                "mode": "auto",
                "spatial_format": "atmos",
                "quality_preset": "fast",
                "target_lufs": -18.0,
                "foley_prompt": "smooth whoosh motion across room",
                "allow_foley_fallback": True,
            },
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["mode"], "silent_foley")
        self.assertTrue(Path(res["binaural_output_mp4"]).is_file())
        self.assertTrue(Path(res["master_spatial_wav"]).is_file())
        self.assertTrue(Path(res["original_preview_wav"]).is_file())
        # 生成されたバイノーラルMP4が有音になっていること
        out_info = inspect_video(res["binaural_output_mp4"])
        self.assertTrue(out_info["has_audio_stream"])
        self.assertFalse(out_info["is_silent"])

    def test_07_speaker_splitter_overlap_handling(self) -> None:
        """重なり発話を含む音声でクラッシュ・NaNなくモノラル話者WAVが生成されることを確認する。"""
        sr = SAMPLE_RATE
        dur = 2.5
        n = int(dur * sr)
        t = np.arange(n, dtype=np.float64) / sr
        sp1 = _synth_voice_wave(t, 130.0, [(1, 1.0), (2, 0.7), (3, 0.5)], 3.0)
        sp2 = _synth_voice_wave(t, 290.0, [(1, 0.7), (2, 1.0), (4, 0.6)], 4.0)
        audio = np.zeros((n, 2), dtype=np.float32)
        # 左に話者1 (0.1..1.6s)、右に話者2 (1.0..2.4s) → 1.0..1.6s が重なり区間
        audio[int(0.1 * sr):int(1.6 * sr), 0] += sp1[int(0.1 * sr):int(1.6 * sr)]
        audio[int(0.1 * sr):int(1.6 * sr), 1] += 0.25 * sp1[int(0.1 * sr):int(1.6 * sr)]
        audio[int(1.0 * sr):int(2.4 * sr), 1] += sp2[int(1.0 * sr):int(2.4 * sr)]
        audio[int(1.0 * sr):int(2.4 * sr), 0] += 0.25 * sp2[int(1.0 * sr):int(2.4 * sr)]

        ov_dir = self.root / "overlap_test"
        ov_dir.mkdir(parents=True, exist_ok=True)
        in_wav = ov_dir / "dialog.wav"
        sf.write(str(in_wav), audio, sr, subtype="FLOAT")

        out = split_speakers(in_wav, ov_dir / "out", vision_tracks=None, quality="standard", force_fallback=True)
        self.assertGreaterEqual(len(out["speakers"]), 2)
        for sp in out["speakers"]:
            wav, r_sr = sf.read(sp["wav_path"], dtype="float32")
            self.assertEqual(r_sr, SAMPLE_RATE)
            self.assertEqual(wav.ndim, 1, "話者出力は48kHzモノラルであるべきです")
            self.assertTrue(np.isfinite(wav).all())


if __name__ == "__main__":
    unittest.main()

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
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from app.audio.foley_engine import MISSING_FOLEY_MESSAGE, build_foley_prompt, find_hunyuan_foley, generate_foley
from app.audio.sfx_splitter import split_sfx_layers
from app.audio.speaker_splitter import (
    apply_dialog_pan_mode_to_keyframes,
    map_dialog_azimuth,
    split_speakers,
)
from app.config import SAMPLE_RATE
from app.director import (
    apply_director_plan_to_scene,
    build_auto_director_plan,
    validate_director_plan,
)
from app.pipeline import apply_director_plan_and_remix, inspect_video_world, remix_job, run_job
from app.spatial.adm_bwf import validate_adm_bwf
from app.spatial.mastering import measure_lufs
from app.vision.analyzer import (
    analyze_video_spatial,
    export_director_contact_sheet,
    inspect_video,
)


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
        unavail = {
            "available": False,
            "weights_found": False,
            "root": None,
            "code_dir": None,
            "infer_script": None,
            "python_exe": None,
            "hf_home": None,
            "variants": {"XXL": None, "XL": None},
        }
        with mock.patch("app.audio.foley_engine.find_hunyuan_foley", return_value=unavail):
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
        unavail = {
            "available": False,
            "weights_found": False,
            "root": None,
            "code_dir": None,
            "infer_script": None,
            "python_exe": None,
            "hf_home": None,
            "variants": {"XXL": None, "XL": None},
        }
        with mock.patch("app.audio.foley_engine.find_hunyuan_foley", return_value=unavail):
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

    def test_08_dialog_pan_mode_cinema_vs_immersive_and_match_original(self) -> None:
        """dialog_pan_mode (cinema vs immersive) と loudness_mode=match_original (run_job & remix_job) を検証する。"""
        _json = __import__("json")

        # 1) 定位写像関数とキーフレーム適用（元配列非破壊）の単体確認
        self.assertEqual(map_dialog_azimuth(28.0, "immersive"), 28.0)
        cin_28 = map_dialog_azimuth(28.0, "cinema")
        self.assertGreater(cin_28, 10.0)
        self.assertLessEqual(cin_28, 24.0)

        raw_kfs = [{"t": 0.0, "azimuth_deg": -32.0, "elevation_deg": 2.0, "distance_m": 1.8}]
        cin_kfs = apply_dialog_pan_mode_to_keyframes(raw_kfs, "cinema")
        imm_kfs = apply_dialog_pan_mode_to_keyframes(raw_kfs, "immersive")
        self.assertEqual(raw_kfs[0]["azimuth_deg"], -32.0)
        self.assertEqual(imm_kfs[0]["azimuth_deg"], -32.0)
        self.assertGreater(cin_kfs[0]["azimuth_deg"], -24.5)
        self.assertLess(cin_kfs[0]["azimuth_deg"], -10.0)

        # 2) split_speakers での cinema vs immersive 角度差とオフスクリーン分散・raw_keyframes 保持
        sr = SAMPLE_RATE
        dur = 2.4
        n = int(dur * sr)
        t = np.arange(n, dtype=np.float64) / sr
        sp1 = _synth_voice_wave(t, 130.0, [(1, 1.0), (2, 0.8), (4, 0.6)], 3.2)
        sp2 = _synth_voice_wave(t, 285.0, [(1, 0.8), (2, 1.0), (5, 0.6)], 4.0)
        mono_dlg = np.zeros(n, dtype=np.float32)
        mono_dlg[int(0.15 * sr):int(1.05 * sr)] = sp1[int(0.15 * sr):int(1.05 * sr)]
        mono_dlg[int(1.30 * sr):int(2.25 * sr)] = sp2[int(1.30 * sr):int(2.25 * sr)]
        dlg_dir = self.root / "pan_mode_split"
        dlg_dir.mkdir(parents=True, exist_ok=True)
        dlg_wav = dlg_dir / "dialog.wav"
        sf.write(str(dlg_wav), np.stack([mono_dlg, mono_dlg], axis=1), sr, subtype="FLOAT")

        dummy_tracks = [
            {
                "person_id": "person_0",
                "label": "人物1（左）",
                "mean_azimuth_deg": -34.0,
                "mean_elevation_deg": 0.0,
                "mean_distance_m": 1.8,
                "keyframes": [
                    {"t": 0.1, "azimuth_deg": -34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.85},
                    {"t": 1.0, "azimuth_deg": -34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.85},
                    {"t": 1.3, "azimuth_deg": -34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.05},
                    {"t": 2.2, "azimuth_deg": -34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.05},
                ],
            },
            {
                "person_id": "person_1",
                "label": "人物2（右）",
                "mean_azimuth_deg": 34.0,
                "mean_elevation_deg": 0.0,
                "mean_distance_m": 1.8,
                "keyframes": [
                    {"t": 0.1, "azimuth_deg": 34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.05},
                    {"t": 1.0, "azimuth_deg": 34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.05},
                    {"t": 1.3, "azimuth_deg": 34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.85},
                    {"t": 2.2, "azimuth_deg": 34.0, "elevation_deg": 0.0, "distance_m": 1.8, "mouth_activity": 0.85},
                ],
            },
        ]
        res_cin = split_speakers(dlg_wav, dlg_dir / "cin", vision_tracks=dummy_tracks, force_fallback=True, dialog_pan_mode="cinema")
        res_imm = split_speakers(dlg_wav, dlg_dir / "imm", vision_tracks=dummy_tracks, force_fallback=True, dialog_pan_mode="immersive")
        self.assertEqual(len(res_cin["speakers"]), 2)
        self.assertEqual(len(res_imm["speakers"]), 2)
        for s_c, s_i in zip(res_cin["speakers"], res_imm["speakers"]):
            self.assertTrue(s_c.get("raw_keyframes"))
            self.assertLess(abs(s_c["mean_azimuth_deg"]), abs(s_i["mean_azimuth_deg"]) - 5.0)
            self.assertLessEqual(abs(s_c["mean_azimuth_deg"]), 24.0)
            self.assertGreater(abs(s_i["mean_azimuth_deg"]), 28.0)

        # オフスクリーン複数話者の cinema vs immersive プリセット確認
        off_cin = split_speakers(dlg_wav, dlg_dir / "off_cin", vision_tracks=None, force_fallback=True, dialog_pan_mode="cinema")
        off_imm = split_speakers(dlg_wav, dlg_dir / "off_imm", vision_tracks=None, force_fallback=True, dialog_pan_mode="immersive")
        self.assertEqual([abs(sp["mean_azimuth_deg"]) for sp in off_cin["speakers"][:2]], [14.0, 14.0])
        self.assertEqual([abs(sp["mean_azimuth_deg"]) for sp in off_imm["speakers"][:2]], [22.0, 22.0])

        # 3) run_job (match_original + cinema + ducking=True) の LUFS 整合＆scene保存確認
        job_dir = self.root / "job_match_original"
        res = run_job(
            job_dir,
            {
                "video_path": str(self.voiced_video),
                "mode": "voiced",
                "spatial_format": "atmos",
                "quality_preset": "fast",
                "loudness_mode": "match_original",
                "dialog_pan_mode": "cinema",
                "enable_dialog_ducking": True,
                "enable_dereverb": False,
                "force_dsp_fallback": True,
            },
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["loudness_mode"], "match_original")
        self.assertEqual(res["dialog_pan_mode"], "cinema")
        self.assertTrue(res["enable_dialog_ducking"])
        self.assertIsNotNone(res["original_lufs"])

        expected_lufs = max(-24.0, min(-11.0, float(res["original_lufs"])))
        self.assertAlmostEqual(float(res["effective_target_lufs"]), expected_lufs, delta=0.1)

        bin_data, bin_sr = sf.read(res["binaural_wav"], dtype="float32", always_2d=True)
        out_lufs = measure_lufs(bin_data, sample_rate=bin_sr)
        self.assertAlmostEqual(out_lufs, expected_lufs, delta=1.2, msg=f"match_original LUFS誤差過大: out={out_lufs:.2f}, expected={expected_lufs:.2f}")

        scene = _json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
        self.assertEqual(scene["mix_settings"]["dialog_pan_mode"], "cinema")
        self.assertEqual(scene["mix_settings"]["loudness_mode"], "match_original")
        dialog_objs = [o for o in scene["objects"] if o.get("role") == "dialog" and o.get("raw_keyframes")]
        self.assertGreaterEqual(len(dialog_objs), 2)
        for obj in dialog_objs:
            self.assertLessEqual(abs(obj["keyframes"][0]["azimuth_deg"]), 24.0)
            self.assertGreater(abs(obj["raw_keyframes"][0]["azimuth_deg"]), 24.0)

        # 4) remix_job での固定→match_original 復帰と recompute_dialog_pan=True (immersive)
        remix_fixed = remix_job(job_dir, scene, spatial_format="atmos", loudness_mode="fixed", target_lufs=-24.0)
        fixed_data, f_sr = sf.read(remix_fixed["binaural_wav"], dtype="float32", always_2d=True)
        self.assertAlmostEqual(measure_lufs(fixed_data, sample_rate=f_sr), -24.0, delta=1.2)

        scene_recompute = _json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
        scene_recompute["recompute_dialog_pan"] = True
        remix_match = remix_job(
            job_dir,
            scene_recompute,
            spatial_format="atmos",
            target_lufs="match_original",
            loudness_mode="match_original",
            dialog_pan_mode="immersive",
            enable_dialog_ducking=False,
        )
        self.assertEqual(remix_match["loudness_mode"], "match_original")
        self.assertEqual(remix_match["dialog_pan_mode"], "immersive")
        self.assertFalse(remix_match["enable_dialog_ducking"])
        match_data, m_sr = sf.read(remix_match["binaural_wav"], dtype="float32", always_2d=True)
        match_lufs = measure_lufs(match_data, sample_rate=m_sr)
        self.assertAlmostEqual(match_lufs, expected_lufs, delta=1.2, msg=f"remix match_original LUFS誤差過大: out={match_lufs:.2f}, expected={expected_lufs:.2f}")

        scene_after = _json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
        self.assertNotIn("recompute_dialog_pan", scene_after)
        self.assertEqual(scene_after["mix_settings"]["dialog_pan_mode"], "immersive")
        self.assertFalse(scene_after["mix_settings"]["enable_dialog_ducking"])
        for obj in scene_after["objects"]:
            if obj.get("role") == "dialog" and obj.get("raw_keyframes"):
                self.assertGreater(abs(obj["keyframes"][0]["azimuth_deg"]), 24.0)

    def test_09_anime_skin_fallback_conservative(self) -> None:
        """Haar/通常顔検出が0件のアニメ調動画では連続>=3フレームで補助検出し、単発フラッシュは弾く。"""
        fps = 6
        # ケースA: 非対称の横向きアニメ風肌領域が6フレーム連続で存在
        frames_a: list[np.ndarray] = []
        for fi in range(8):
            img = np.full((360, 640, 3), (34, 38, 46), dtype=np.uint8)
            if 1 <= fi <= 6:
                # 画面左寄りの上部に非対称なアニメ調頭部肌色＋濃い目の線
                cv2.ellipse(img, (210, 145), (42, 48), 0, 0, 360, (165, 192, 232), -1)
                cv2.line(img, (185, 130), (205, 130), (35, 32, 40), 3)
                cv2.line(img, (188, 170), (205, 170), (45, 40, 60), 2)
            frames_a.append(img)
        vid_a = _encode_frames_to_mp4(frames_a, fps, self.root / "anime_fallback_hit.mp4")
        res_a = analyze_video_spatial(vid_a, sample_fps=6.0)
        self.assertEqual(len(res_a["person_tracks"]), 1, f"連続アニメ調人物は1名検出されるべき: {res_a['person_tracks']}")
        self.assertLess(res_a["person_tracks"][0]["mean_azimuth_deg"], -5.0)

        # ケースB: 単発1フレームの暖色フラッシュは連続>=3フレーム条件で除外される
        frames_b: list[np.ndarray] = []
        for fi in range(8):
            img = np.full((360, 640, 3), (34, 38, 46), dtype=np.uint8)
            if fi == 3:
                cv2.ellipse(img, (210, 145), (42, 48), 0, 0, 360, (165, 192, 232), -1)
                cv2.line(img, (185, 130), (205, 130), (35, 32, 40), 3)
            frames_b.append(img)
        vid_b = _encode_frames_to_mp4(frames_b, fps, self.root / "anime_fallback_flash.mp4")
        res_b = analyze_video_spatial(vid_b, sample_fps=6.0)
        self.assertEqual(len(res_b["person_tracks"]), 0, "単発フラッシュを誤検出すべきではありません")

    def test_10_sfx_splitter_lossless_and_silent_fold(self) -> None:
        """SFX 4層分解の厳密総和ロスレス性および -62dBFS 未満 L/R 無音畳み込みを検証する。"""
        sr = SAMPLE_RATE
        dur = 2.2
        n = int(dur * sr)
        t = np.arange(n, dtype=np.float64) / sr

        # ケースA: 低域 Sub (75Hz) + 持続 Ambience (650Hz) + 左右中央の衝撃クリック
        sig_a = np.zeros((n, 2), dtype=np.float32)
        sub = (0.22 * np.sin(2.0 * math.pi * 75.0 * t)).astype(np.float32)
        amb = (0.10 * np.sin(2.0 * math.pi * 650.0 * t)).astype(np.float32)
        sig_a[:, 0] += sub + amb
        sig_a[:, 1] += sub + amb

        def _click(length: int = 320, amp: float = 0.65) -> np.ndarray:
            lt = np.arange(length, dtype=np.float64)
            return (amp * np.exp(-lt / 42.0) * (1.0 - np.exp(-lt / 4.0))).astype(np.float32)

        clk = _click()
        i_l, i_c, i_r = int(0.45 * sr), int(1.10 * sr), int(1.75 * sr)
        sig_a[i_l : i_l + len(clk), 0] += clk
        sig_a[i_c : i_c + len(clk), 0] += clk * 0.8
        sig_a[i_c : i_c + len(clk), 1] += clk * 0.8
        sig_a[i_r : i_r + len(clk), 1] += clk

        dir_a = self.root / "sfx_split_a"
        dir_a.mkdir(parents=True, exist_ok=True)
        wav_a = dir_a / "effect.wav"
        sf.write(str(wav_a), sig_a, sr, subtype="FLOAT")

        res_a = split_sfx_layers(wav_a, dir_a / "out", sample_rate=sr)
        self.assertLess(res_a["reconstruction_error_db"], -100.0)
        names_a = {layer["name"] for layer in res_a["layers"]}
        self.assertEqual(
            names_a,
            {"sfx_sub", "sfx_ambience", "sfx_impact_left", "sfx_impact_center", "sfx_impact_right"},
        )
        # 出力WAVの全層加算和が元の信号と一致
        sum_a = np.zeros_like(sig_a, dtype=np.float64)
        for layer in res_a["layers"]:
            arr, r_sr = sf.read(layer["path"], dtype="float32", always_2d=True)
            self.assertEqual(r_sr, sr)
            sum_a += arr
        self.assertLess(float(np.max(np.abs(sum_a - sig_a))), 1e-5)

        sides = [ev["dominant_side"] for ev in res_a["impact_events"]]
        self.assertIn("left", sides)
        self.assertIn("right", sides)

        # ケースB: 完全中央パン (L == R) では L/R 出力がスキップされて center に畳み込まれ、総和は一致
        sig_b = np.zeros((n, 2), dtype=np.float32)
        sig_b[:, 0] = sub + amb
        sig_b[:, 1] = sub + amb
        sig_b[i_c : i_c + len(clk), 0] += clk
        sig_b[i_c : i_c + len(clk), 1] += clk

        dir_b = self.root / "sfx_split_b"
        dir_b.mkdir(parents=True, exist_ok=True)
        wav_b = dir_b / "effect.wav"
        sf.write(str(wav_b), sig_b, sr, subtype="FLOAT")

        res_b = split_sfx_layers(wav_b, dir_b / "out", sample_rate=sr)
        self.assertIn("left", res_b["folded_channels"])
        self.assertIn("right", res_b["folded_channels"])
        self.assertFalse((dir_b / "out" / "sfx_impact_left.wav").exists())
        self.assertFalse((dir_b / "out" / "sfx_impact_right.wav").exists())
        sum_b = np.zeros_like(sig_b, dtype=np.float64)
        for layer in res_b["layers"]:
            arr, _ = sf.read(layer["path"], dtype="float32", always_2d=True)
            sum_b += arr
        self.assertLess(float(np.max(np.abs(sum_b - sig_b))), 1e-5)

    def test_11_vision_sfx_distance_intensity_and_contact_sheet(self) -> None:
        """激しいアクション近接(1.3..2.2m)・静寂遠景(3.2..5.2m)・シーン温度・コンタクトシートを検証する。"""
        fps = 8
        # 動画1: 激しいアクション＋フラッシュ (2.0秒)
        act_frames: list[np.ndarray] = []
        for fi in range(16):
            flash = 165 if fi in (4, 10, 11) else 25
            img = np.full((240, 426, 3), (flash, flash, flash), dtype=np.uint8)
            x1 = int((fi * 55) % 360) + 30
            y1 = int(60 + 90 * math.sin(fi * 1.1))
            x2 = 400 - x1
            cv2.rectangle(img, (x1 - 45, y1 - 45), (x1 + 45, y1 + 45), (40, 120, 250), -1)
            cv2.circle(img, (x2, 120), 42, (240, 230, 60), -1)
            act_frames.append(img)
        act_vid = _encode_frames_to_mp4(act_frames, fps, self.root / "vision_action.mp4")

        # 動画2: 静かな微小動き (2.0秒)
        quiet_frames: list[np.ndarray] = []
        for fi in range(16):
            img = np.full((240, 426, 3), (30, 34, 40), dtype=np.uint8)
            cx = 200 + (fi % 3)
            cv2.circle(img, (cx, 120), 10, (90, 110, 130), -1)
            quiet_frames.append(img)
        quiet_vid = _encode_frames_to_mp4(quiet_frames, fps, self.root / "vision_quiet.mp4")

        res_act = analyze_video_spatial(act_vid, sample_fps=8.0)
        res_quiet = analyze_video_spatial(quiet_vid, sample_fps=8.0)

        act_dists = [m["distance_m"] for m in res_act["motion_track"]]
        quiet_dists = [m["distance_m"] for m in res_quiet["motion_track"]]
        act_mean_d = float(np.mean(act_dists))
        quiet_mean_d = float(np.mean(quiet_dists))

        self.assertLess(act_mean_d, 2.35, f"激しいアクションは近接すべき: {act_mean_d:.2f}m")
        self.assertGreaterEqual(min(act_dists), 1.3)
        self.assertGreater(quiet_mean_d, 3.2, f"静かなシーンは遠景になるべき: {quiet_mean_d:.2f}m")
        self.assertLessEqual(max(quiet_dists), 5.2)
        self.assertLess(act_mean_d, quiet_mean_d - 1.2)

        self.assertTrue(res_act.get("intensity_track"))
        self.assertTrue(res_act.get("scene_segments"))
        self.assertTrue(res_quiet.get("scene_segments"))
        self.assertGreater(
            max(p["temperature"] for p in res_act["intensity_track"]),
            max(p["temperature"] for p in res_quiet["intensity_track"]),
        )
        self.assertEqual(res_quiet["scene_segments"][0]["label"], "静寂")

        # コンタクトシート生成検証 (<= 1280px, <= 8枚)
        sheet_path = self.root / "test_contact_sheet.jpg"
        sheet = export_director_contact_sheet(act_vid, res_act, sheet_path, max_frames=8)
        self.assertTrue(sheet_path.is_file() and sheet_path.stat().st_size > 0)
        self.assertLessEqual(sheet["width"], 1280)
        self.assertLessEqual(sheet["frame_count"], 8)
        self.assertGreaterEqual(sheet["frame_count"], 2)

    def test_12_director_plan_idempotency_and_remix(self) -> None:
        """AI監督プランの検証・複数回適用冪等性・run_job統合・apply_director_plan_and_remix を検証する。"""
        # 1) validate_director_plan のクランプ＆ソート検証
        dirty_plan = {
            "tracks": {
                "sfx_impact": [
                    {"t_sec": 100.0, "gain_db": 40.0, "distance_m": 0.1, "duck_exempt": True},
                    {"t_sec": -2.0, "gain_db": -40.0, "distance_m": 99.0, "duck_exempt": False},
                ]
            }
        }
        clean = validate_director_plan(dirty_plan, duration_sec=4.0)
        kfs = clean["tracks"]["sfx_impact"]
        self.assertEqual([k["t_sec"] for k in kfs], [0.0, 4.0])
        self.assertEqual(kfs[0]["gain_db"], -18.0)
        self.assertEqual(kfs[0]["distance_m"], 12.0)
        self.assertEqual(kfs[1]["gain_db"], 12.0)
        self.assertEqual(kfs[1]["distance_m"], 0.6)

        # 2) apply_director_plan_to_scene の複数回適用冪等性検証
        dummy_vr = {
            "duration_sec": 4.0,
            "scene_segments": [
                {"segment_id": 0, "start_sec": 0.0, "end_sec": 2.0, "mean_temperature": 0.15, "label": "静寂", "label_code": "quiet"},
                {"segment_id": 1, "start_sec": 2.0, "end_sec": 4.0, "mean_temperature": 0.85, "label": "クライマックス", "label_code": "climax"},
            ],
        }
        auto_plan = build_auto_director_plan(dummy_vr, sfx_info={"impact_events": [{"t_sec": 2.8, "peak_db": -8.0, "dominant_side": "left"}]})
        base_scene = {
            "version": 1,
            "duration_sec": 4.0,
            "sample_rate": SAMPLE_RATE,
            "objects": [
                {
                    "id": "sfx_impact_center",
                    "role": "sfx_impact",
                    "is_bed": False,
                    "gain_db": 0.0,
                    "keyframes": [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.5}],
                },
                {
                    "id": "sfx_ambience",
                    "role": "ambience",
                    "is_bed": True,
                    "gain_db": -0.5,
                    "keyframes": [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 4.5}],
                },
            ],
        }
        once = apply_director_plan_to_scene(base_scene, auto_plan)
        twice = apply_director_plan_to_scene(once, auto_plan)
        thrice = apply_director_plan_to_scene(twice, auto_plan)
        self.assertEqual(once["objects"][0]["keyframes"], twice["objects"][0]["keyframes"])
        self.assertEqual(once["objects"][0]["keyframes"], thrice["objects"][0]["keyframes"])
        self.assertEqual(once["objects"][1]["keyframes"], thrice["objects"][1]["keyframes"])

        # 別プラン適用後に元のプランに戻しても完全一致
        alt_plan = {"tracks": {"sfx_impact": [{"t_sec": 0.0, "gain_db": -6.0, "distance_m": 5.0, "duck_exempt": False}]}}
        swapped = apply_director_plan_to_scene(twice, alt_plan)
        restored = apply_director_plan_to_scene(swapped, auto_plan)
        self.assertEqual(once["objects"][0]["keyframes"], restored["objects"][0]["keyframes"])

        # 3) run_job + apply_director_plan_and_remix 統合テスト
        # 効果音イベントも確実に含まれる有音動画を簡易合成
        fps = 10
        dur = 2.4
        n_frames = int(dur * fps)
        frames = []
        for fi in range(n_frames):
            img = np.full((240, 426, 3), (38, 42, 52), dtype=np.uint8)
            cx = 90 if fi < 12 else 330
            _draw_synthetic_face(img, cx, 120, 48, mouth_open=0.5 + 0.5 * math.sin(fi))
            frames.append(img)
        n_s = int(dur * SAMPLE_RATE)
        t_ax = np.arange(n_s, dtype=np.float64) / SAMPLE_RATE
        voice = _synth_voice_wave(t_ax, 150.0, [(1, 0.8), (2, 1.0), (4, 0.7)], 3.5)
        st = np.stack([voice * 0.6, voice * 0.6], axis=1).astype(np.float32)
        # 80Hz sub + 左右非対称クリック (SFX分離対象)
        st[:, 0] += (0.14 * np.sin(2.0 * math.pi * 80.0 * t_ax)).astype(np.float32)
        st[:, 1] += (0.14 * np.sin(2.0 * math.pi * 80.0 * t_ax)).astype(np.float32)
        hit = int(1.2 * SAMPLE_RATE)
        st[hit : hit + 300, 0] += (0.35 * np.exp(-np.arange(300) / 45.0)).astype(np.float32)

        wav_mix = self.root / "dir_test_src.wav"
        sf.write(str(wav_mix), st, SAMPLE_RATE, subtype="FLOAT")
        vid_mix = _encode_frames_to_mp4(frames, fps, self.root / "dir_test_vid.mp4", audio_wav=wav_mix)

        job_dir = self.root / "job_director_e2e"
        res = run_job(
            job_dir,
            {
                "video_path": str(vid_mix),
                "mode": "voiced",
                "spatial_format": "atmos",
                "quality_preset": "fast",
                "target_lufs": -18.0,
                "enable_dereverb": False,
                "force_dsp_fallback": True,
                "enable_sfx_multilayer": True,
                "enable_director": True,
            },
        )
        self.assertTrue(res["ok"])
        self.assertTrue(res["enable_sfx_multilayer"])
        self.assertTrue(res["enable_director"])
        self.assertTrue((job_dir / "work" / "director_sheet.jpg").is_file())
        self.assertTrue((job_dir / "work" / "director_context.json").is_file())
        self.assertTrue((job_dir / "work" / "scene_base.json").is_file())
        self.assertGreaterEqual(len(res["sfx_layers"]), 3)

        ctx = json.loads((job_dir / "work" / "director_context.json").read_text(encoding="utf-8"))
        self.assertIn("intensity_track", ctx)
        self.assertIn("scene_segments", ctx)
        self.assertIn("director_plan", ctx)

        # apply_director_plan_and_remix が再解析・再分離なしで高速動作＆冪等であることを確認
        custom_plan = {
            "source": "test_custom_director",
            "summary_ja": "テスト用カスタム監督プラン",
            "tracks": {
                "sfx_impact": [{"t_sec": 0.0, "gain_db": 5.0, "distance_m": 1.3, "duck_exempt": True}],
                "sfx_sub": [{"t_sec": 0.0, "gain_db": 4.0, "duck_exempt": True}],
                "ambience": [{"t_sec": 0.0, "gain_db": -3.0, "duck_exempt": False}],
            },
        }
        with (
            mock.patch("app.pipeline.analyze_video_spatial", side_effect=AssertionError("再映像解析してはならない")),
            mock.patch("app.pipeline.separate_dme", side_effect=AssertionError("再分離してはならない")),
        ):
            r1 = apply_director_plan_and_remix(job_dir, custom_plan)
            scene_after_1 = json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
            r2 = apply_director_plan_and_remix(job_dir, custom_plan)
            scene_after_2 = json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))

        self.assertTrue(r1["ok"])
        self.assertTrue(r2["ok"])
        self.assertEqual(r2["director_plan"]["source"], "test_custom_director")
        self.assertEqual(scene_after_1["objects"], scene_after_2["objects"])
        self.assertTrue(Path(r2["binaural_output_mp4"]).is_file())
        ctx_after = json.loads((job_dir / "work" / "director_context.json").read_text(encoding="utf-8"))
        self.assertEqual(ctx_after["director_plan"]["source"], "test_custom_director")

    def test_13_world_360_flyby_director_coords_and_inspect_world(self) -> None:
        """360°ワールドプロファイル・フライバイ検出・監督360°座標冪等適用・inspect_video_world を検証する。"""
        # 1) 画面中央から右端へ急接近通過 ＋ 上端へ急上昇する合成動画を生成
        fps = 12
        frames: list[np.ndarray] = []
        # 前半 (0..11): 明るい屋外空背景上で物体が中央→右端へ拡大しながら高速移動
        for fi in range(12):
            img = np.full((240, 426, 3), (210, 175, 120), dtype=np.uint8)
            img[:90, :, 0] = 235  # 上部に青空風帯域
            cx = 175 + fi * 21
            rad = 26 + fi * 4
            cv2.circle(img, (min(420, cx), 128), rad, (40, 80, 220), -1)
            cv2.rectangle(img, (max(0, cx - 22), 95), (min(425, cx + 22), 160), (20, 230, 240), 3)
            frames.append(img)
        # 後半 (12..23): 上部エリアで中央から上端へ高速上昇移動
        for fi in range(12):
            img = np.full((240, 426, 3), (210, 175, 120), dtype=np.uint8)
            img[:90, :, 0] = 235
            cy = max(10, 86 - fi * 7)
            cv2.circle(img, (213, cy), 30, (30, 90, 235), -1)
            frames.append(img)

        vid_fly = _encode_frames_to_mp4(frames, fps, self.root / "world360_flyby.mp4")

        vr = analyze_video_spatial(vid_fly, sample_fps=12.0)
        wp = vr.get("world_profile") or {}
        flybys = vr.get("flyby_events") or []
        self.assertIn("environment_type", wp)
        self.assertIn("rear_immersion_strength", wp)
        self.assertIn("sky_overhead_presence", wp)
        self.assertTrue(wp.get("foley_world_hints"))
        kinds = {e.get("kind") for e in flybys}
        self.assertTrue(
            "side_rear_right" in kinds or "overhead_pass" in kinds,
            f"画面外通過イベントが未検出でした: {flybys}",
        )

        # 2) inspect_video_world の返却・コンタクトシート・Foleyプロンプト検証
        insp_dir = self.root / "world_inspect_out"
        w_info = inspect_video_world(vid_fly, out_dir=insp_dir, sample_fps=8.0, max_frames=6)
        self.assertTrue(w_info["ok"])
        self.assertTrue(Path(w_info["director_sheet_path"]).is_file())
        self.assertTrue(w_info["suggested_foley_prompt"])
        self.assertIn("360°", w_info["spatial_360_guide_ja"])

        # build_foley_prompt の 360° 環境 + 物体ヒント結合検証
        built_prompt = build_foley_prompt(
            vr.get("scene_summary"),
            user_prompt="雨",
            world_profile=wp,
            scene_objects_hint="two speeding cars, metallic collision",
        )
        self.assertIn("rain", built_prompt)
        self.assertIn("two speeding cars, metallic collision", built_prompt)
        self.assertTrue(any(h in built_prompt for h in wp["foley_world_hints"]))

        # 3) AI監督 360°座標制御（後方 -135°・頭上 +65°・world_360）＆ 2回連続完全冪等リミックス
        auto_w360 = build_auto_director_plan(
            vr,
            sfx_info={"impact_events": [{"t_sec": 0.4, "peak_db": -10.0, "dominant_side": "right"}]},
            enable_world_360=True,
        )
        self.assertTrue((auto_w360.get("world_360") or {}).get("enabled"))
        self.assertGreaterEqual((auto_w360["world_360"]).get("rear_level", 0.0), 1.0)

        # 軽量E2Eジョブを作成しカスタム360°プランを2回適用して波形完全一致を検証
        dur = 1.6
        n_s = int(dur * SAMPLE_RATE)
        t_ax = np.arange(n_s, dtype=np.float64) / SAMPLE_RATE
        speech = (0.28 * np.sin(2.0 * math.pi * 240.0 * t_ax)).astype(np.float32)
        imp_noise = np.random.default_rng(99).standard_normal(n_s).astype(np.float32) * 0.25
        stereo_src = np.stack([speech + imp_noise * 0.4, speech + imp_noise * 0.8], axis=1).astype(np.float32)
        wav_src = self.root / "w360_src.wav"
        sf.write(str(wav_src), stereo_src, SAMPLE_RATE, subtype="FLOAT")

        f_list = [np.full((180, 320, 3), (40, 44, 52), dtype=np.uint8) for _ in range(16)]
        vid_job = _encode_frames_to_mp4(f_list, 10, self.root / "w360_job.mp4", audio_wav=wav_src)

        job_dir = self.root / "job_world360_idemp"
        r0 = run_job(
            job_dir,
            {
                "video_path": str(vid_job),
                "mode": "voiced",
                "spatial_format": "atmos",
                "quality_preset": "fast",
                "target_lufs": -18.0,
                "enable_dereverb": False,
                "force_dsp_fallback": True,
                "enable_sfx_multilayer": True,
                "enable_director": True,
                "enable_world_360": True,
            },
        )
        self.assertTrue(r0["ok"])
        self.assertTrue(r0["enable_world_360"])

        custom_360_plan = {
            "source": "custom_360_director",
            "summary_ja": "真後ろ通過＆頭上ハイト360°検証プラン",
            "world_360": {
                "enabled": True,
                "environment_type": "outdoor_night_action",
                "environment_label_ja": "夜間・暗所アクション空間",
                "rear_level": 1.38,
                "height_level": 1.30,
                "surround_wrap": 0.90,
                "rt60_sec": 0.58,
            },
            "tracks": {
                "sfx_impact": [
                    {"t_sec": 0.0, "azimuth_deg": -135.0, "elevation_deg": 65.0, "distance_m": 1.4, "gain_db": 3.0, "duck_exempt": True},
                    {"t_sec": dur, "azimuth_deg": 135.0, "elevation_deg": 20.0, "distance_m": 1.8, "gain_db": 2.0, "duck_exempt": True},
                ],
                "sfx_impact_left": [
                    {"t_sec": 0.0, "azimuth_deg": -135.0, "elevation_deg": 65.0, "distance_m": 1.4, "gain_db": 3.0, "duck_exempt": True},
                    {"t_sec": dur, "azimuth_offset_deg": -25.0, "elevation_deg": 35.0, "distance_m": 1.7, "gain_db": 2.5, "duck_exempt": True},
                ],
                "ambience": [
                    {"t_sec": 0.0, "gain_db": 1.5, "duck_exempt": True},
                ],
            },
        }
        res_1 = apply_director_plan_and_remix(job_dir, custom_360_plan)
        scene_1 = json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
        wav_1, _ = sf.read(res_1["binaural_wav"], dtype="float64")

        res_2 = apply_director_plan_and_remix(job_dir, custom_360_plan)
        scene_2 = json.loads((job_dir / "scene.json").read_text(encoding="utf-8"))
        wav_2, _ = sf.read(res_2["binaural_wav"], dtype="float64")

        self.assertEqual(scene_1["world_360"], scene_2["world_360"])
        self.assertEqual(scene_1["objects"], scene_2["objects"])
        self.assertAlmostEqual(float(np.max(np.abs(wav_1 - wav_2))), 0.0, places=6)
        imp_objs = [o for o in scene_2["objects"] if o.get("role") == "sfx_impact"]
        self.assertTrue(imp_objs)
        self.assertTrue(
            any(abs(float(k.get("azimuth_deg", 0.0))) >= 110.0 and float(k.get("elevation_deg", 0.0)) >= 50.0 for o in imp_objs for k in (o.get("keyframes") or [])),
            f"360° 真後ろ・頭上座標が反映されていません: {imp_objs[0].get('keyframes')}",
        )


if __name__ == "__main__":
    unittest.main()

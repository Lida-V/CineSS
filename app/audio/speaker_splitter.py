# coding: utf-8
"""人物別話者分離＆映像人物自動紐付けエンジン。

以下の3段階でセリフトラック（`dialog.wav`）を人物ごとのモノラル点音源へ分解し、
映像上のどの人物（左・右・奥行き）が話しているかを自動決定する：

1. 話者ダイアリゼーション:
   - SoundSeparator の `Nemotron-3-Diarization` が取得済みなら `.venv` 経由で
     `NemotronDiarizer.diarize()` を安全に呼び出す（最大8話者・重なり発話対応）。
   - 未取得／フォールバック時は、適応VAD＋MFCC/ピッチ/ステレオ定位特徴の階層クラスタリング
     （`method: "acoustic_clustering"`）で話者区間を推定する。
2. 重なり区間の分離強化＋10msクロスフェード時間振り分け:
   - SoundSeparator の `diarization_stems.py` と同等のビットマスク＆10msクロスフェード方式を踏襲。
   - 重なり発話区間は SpeechBrain SepFormer（ローカル導入済み時）または
     話者別スペクトル・ステレオ定位のソフト Wiener マスクで分離を強化し、
     総エネルギー欠損を防止した上で各話者を 48kHz モノラル WAV に書き出す。
3. 話者 ↔ 映像上の人物の自動紐付け:
   - 各話者の発話エンベロープと各人物の口元モーション活性度（`mouth_activity`）の
     時間相関・コントラスト（＋ステレオ定位補助）をスコア化し、
     ハンガリアン法（`scipy.optimize.linear_sum_assignment`）で最適対応を決定する。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.fft import dct
from scipy.optimize import linear_sum_assignment
from scipy.signal import istft, resample_poly, stft

from app.audio.separator_bridge import detect_soundseparator
from app.config import MODELS_DIR, SAMPLE_RATE

SPEAKER_LETTERS = "ABCDEFGH"
SPEAKER_NAMES = [f"話者{c}" for c in SPEAKER_LETTERS]

DIALOG_PAN_MODES = ("cinema", "immersive")
CINEMA_DIALOG_PAN_LIMIT_DEG = 22.0


def _normalize_dialog_pan_mode(dialog_pan_mode: str | None) -> str:
    mode = str(dialog_pan_mode or "cinema").strip().lower()
    if mode == "immersive":
        return "immersive"
    return "cinema"


def map_dialog_azimuth(azimuth_deg: float, dialog_pan_mode: str = "cinema") -> float:
    """映像水平角をセリフ定位モード（cinema: ±22°センター軸 / immersive: 原角）へ写像する。"""
    az = max(-180.0, min(180.0, float(azimuth_deg)))
    mode = _normalize_dialog_pan_mode(dialog_pan_mode)
    if mode == "immersive":
        return round(az, 2)
    limit = CINEMA_DIALOG_PAN_LIMIT_DEG
    return round(limit * math.tanh(az / limit), 2)


def apply_dialog_pan_mode_to_keyframes(
    raw_keyframes: list[dict[str, Any]],
    dialog_pan_mode: str = "cinema",
) -> list[dict[str, Any]]:
    """原角キーフレーム列へセリフ定位モードを適用した新しいキーフレーム列を返す。"""
    mode = _normalize_dialog_pan_mode(dialog_pan_mode)
    mapped: list[dict[str, Any]] = []
    for kf in raw_keyframes or []:
        new_kf = dict(kf)
        new_kf["azimuth_deg"] = map_dialog_azimuth(float(kf.get("azimuth_deg", 0.0)), mode)
        mapped.append(new_kf)
    return mapped


def _emit_progress(cb: Callable[..., Any] | None, pct: float, message: str) -> None:
    if not cb:
        return
    try:
        cb(pct, message)
    except TypeError:
        try:
            cb(pct)
        except Exception:
            pass


def _emit_log(log_cb: Callable[..., Any] | None, message: str) -> None:
    if not log_cb:
        return
    try:
        log_cb(message)
    except TypeError:
        try:
            log_cb("speaker_split", message)
        except Exception:
            pass


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(audio.astype(np.float32), int(dst_sr) // g, int(src_sr) // g, axis=0).astype(np.float32)


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return round(20.0 * math.log10(rms), 2) if rms > 1e-10 else -120.0


# ---------------------------------------------------------------------------
# 音響特徴量（純 NumPy/SciPy MFCC + F0 + ステレオ定位）
# ---------------------------------------------------------------------------
def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26, fmin: float = 80.0, fmax: float = 7600.0) -> np.ndarray:
    def _hz2mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel2hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_pts = np.linspace(_hz2mel(fmin), _hz2mel(min(fmax, sr * 0.48)), n_mels + 2)
    hz_pts = np.array([_mel2hz(m) for m in mel_pts])
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return fb


def _extract_mfcc_vector(mono: np.ndarray, sr: int) -> np.ndarray:
    """音声クリップから 13次元 MFCC（平均）を算出する。"""
    if mono.size < 256:
        return np.zeros(13, dtype=np.float32)
    n_fft = 1024
    hop = 256
    _, _, Z = stft(mono.astype(np.float32), fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    power = (np.abs(Z) ** 2).astype(np.float32)
    fb = _mel_filterbank(sr, n_fft, n_mels=26)
    mel_spec = np.dot(fb, power) + 1e-10
    log_mel = np.log(mel_spec)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:13]
    return np.mean(mfcc, axis=1).astype(np.float32)


def _estimate_f0_hz(mono: np.ndarray, sr: int) -> float:
    """自己相関により基本周波数 F0 (75..450Hz) を簡易推定する。"""
    if mono.size < int(sr * 0.04):
        return 150.0
    # 計算高速化のため 16kHz 相当へ間引き
    step = max(1, sr // 16000)
    x = mono[::step].astype(np.float64)
    eff_sr = sr / step
    frame_len = int(eff_sr * 0.04)
    hop = max(1, frame_len // 2)
    min_lag = max(1, int(eff_sr / 450.0))
    max_lag = min(frame_len - 1, int(eff_sr / 75.0))
    f0s: list[float] = []
    for start in range(0, len(x) - frame_len, hop):
        seg = x[start:start + frame_len]
        seg = seg - np.mean(seg)
        energy = float(np.dot(seg, seg))
        if energy < 1e-6:
            continue
        ac = np.correlate(seg, seg, mode="full")[frame_len - 1:]
        if max_lag <= min_lag or max_lag >= len(ac):
            continue
        roi = ac[min_lag:max_lag]
        peak_idx = int(np.argmax(roi))
        if roi[peak_idx] > 0.25 * energy:
            f0s.append(eff_sr / (min_lag + peak_idx))
    return float(np.median(f0s)) if f0s else 150.0


def _segment_features(audio_stereo: np.ndarray, sr: int, start_s: float, end_s: float) -> dict[str, Any]:
    s = max(0, int(round(start_s * sr)))
    e = min(len(audio_stereo), int(round(end_s * sr)))
    clip = audio_stereo[s:e]
    mono = clip.mean(axis=1)
    mfcc = _extract_mfcc_vector(mono, sr)
    f0 = _estimate_f0_hz(mono, sr)
    # スペクトル重心
    n_fft = min(2048, max(256, 1 << int(math.floor(math.log2(max(len(mono), 256))))))
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(min(len(mono), n_fft)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    centroid = float(np.sum(freqs * spec) / max(float(np.sum(spec)), 1e-9))

    rms_l = float(np.sqrt(np.mean(clip[:, 0].astype(np.float64) ** 2)))
    rms_r = float(np.sqrt(np.mean(clip[:, 1].astype(np.float64) ** 2)))
    pan = (rms_r - rms_l) / max(rms_r + rms_l, 1e-9)  # -1 (左) .. +1 (右)
    return {"mfcc": mfcc, "f0": f0, "centroid": centroid, "pan": pan}


# ---------------------------------------------------------------------------
# Step 1: 話者ダイアリゼーション（Nemotron 優先 + 音響クラスタリング）
# ---------------------------------------------------------------------------
def _run_nemotron_diarizer(
    dialog_wav: Path,
    ss_info: dict[str, Any],
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """SoundSeparator の `.venv` 経由で `NemotronDiarizer.diarize()` を安全実行する。"""
    _emit_progress(progress_cb, 15, "Nemotron-3-Diarization で話者区間を推定中...")
    _emit_log(log_cb, "SoundSeparator 共有の Nemotron-3-Diarization で最大8話者の発話区間を推定します。")

    inline_code = r"""
import json, sys
from pathlib import Path
import soundfile as sf

ss_dir = Path(sys.argv[1])
wav_path = Path(sys.argv[2])
sys.path.insert(0, str(ss_dir / "app"))

real_stdout, sys.stdout = sys.stdout, sys.stderr
from transcriber import NemotronDiarizer

audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
diarizer = NemotronDiarizer()
turns = diarizer.diarize(audio, sr)
print(json.dumps({"turns": turns}, ensure_ascii=False), file=real_stdout)
"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [ss_info["python"], "-X", "utf8", "-c", inline_code, ss_info["dir"], str(dialog_wav)],
        cwd=ss_info["dir"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Nemotron 推論エラー: {proc.stderr.strip()[-300:]}")
    payload = json.loads(proc.stdout.strip() or "{}")
    turns = payload.get("turns", [])
    return [
        {
            "start": float(t["start"]),
            "end": float(t["end"]),
            "speaker": str(t["speaker"]),
        }
        for t in turns
        if float(t["end"]) > float(t["start"])
    ]


def _diarize_acoustic_clustering(
    audio_stereo: np.ndarray,
    sr: int,
    vision_tracks: list[dict[str, Any]] | None = None,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """適応VAD + MFCC/F0/ステレオ定位クラスタリングによるローカル話者区間推定。"""
    _emit_progress(progress_cb, 20, "音響特徴（VAD・MFCC・ピッチ・定位）で話者クラスタリング中...")
    _emit_log(log_cb, "ローカル音響クラスタリング（VAD + MFCC/F0/ステレオ定位）で話者区間を推定します。")

    mono = audio_stereo.mean(axis=1)
    duration = len(mono) / sr
    win = max(1, int(round(0.025 * sr)))
    hop = max(1, int(round(0.010 * sr)))
    n_frames = max(1, (len(mono) - win) // hop + 1)

    frame_rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = mono[i * hop:i * hop + win]
        frame_rms[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))

    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))
    peak_db = float(np.percentile(frame_db, 98))
    if peak_db < -55.0:
        return []
    noise_db = float(np.percentile(frame_db, 15))
    thresh_db = max(-50.0, min(peak_db - 18.0, max(noise_db + 10.0, peak_db - 26.0)))
    active = frame_db >= thresh_db

    # 短い無音ギャップ（< 0.20s）を結合し、短すぎる雑音（< 0.18s）を除去
    raw_segs: list[tuple[float, float]] = []
    in_seg = False
    seg_start = 0
    for i, flag in enumerate(active):
        if flag and not in_seg:
            seg_start = i
            in_seg = True
        elif not flag and in_seg:
            raw_segs.append((seg_start * hop / sr, (i * hop + win) / sr))
            in_seg = False
    if in_seg:
        raw_segs.append((seg_start * hop / sr, duration))

    merged: list[list[float]] = []
    for s, e in raw_segs:
        if merged and (s - merged[-1][1]) < 0.20:
            merged[-1][1] = e
        else:
            merged.append([s, min(e, duration)])
    vad_segs = [(s, e) for s, e in merged if (e - s) >= 0.18]
    if not vad_segs:
        return [{"start": 0.0, "end": round(duration, 3), "speaker": SPEAKER_NAMES[0]}]

    # 長い発話区間内で話者が連続交代している場合、0.45秒窓の音響変化点で細分割する
    split_segs: list[tuple[float, float]] = []
    for s, e in vad_segs:
        if (e - s) <= 1.2:
            split_segs.append((s, e))
            continue
        sub_step = 0.45
        sub_times = list(np.arange(s, e - 0.25, sub_step))
        sub_feats = [_segment_features(audio_stereo, sr, t, min(e, t + sub_step)) for t in sub_times]
        cuts = [s]
        for idx in range(1, len(sub_feats)):
            f_prev, f_cur = sub_feats[idx - 1], sub_feats[idx]
            mfcc_dist = float(np.linalg.norm(f_cur["mfcc"] - f_prev["mfcc"]))
            f0_ratio = abs(math.log2(max(f_cur["f0"], 50.0) / max(f_prev["f0"], 50.0)))
            pan_diff = abs(f_cur["pan"] - f_prev["pan"])
            if pan_diff > 0.30 or f0_ratio > 0.38 or mfcc_dist > 12.0:
                cuts.append(sub_times[idx])
        cuts.append(e)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 0.22:
                split_segs.append((a, b))

    if len(split_segs) == 1:
        s, e = split_segs[0]
        return [{"start": round(s, 3), "end": round(e, 3), "speaker": SPEAKER_NAMES[0]}]

    feats = [_segment_features(audio_stereo, sr, s, e) for s, e in split_segs]
    has_stereo_cue = float(np.max([abs(f["pan"]) for f in feats])) > 0.15

    # 特徴行列の構築（MFCC + log2(F0) + log(centroid) + ステレオ定位）
    rows = []
    for f in feats:
        mfcc_norm = f["mfcc"] / (float(np.linalg.norm(f["mfcc"])) + 1e-6)
        vec = list(mfcc_norm * 1.4)
        vec.append(math.log2(max(f["f0"], 60.0) / 150.0) * 1.8)
        vec.append(math.log2(max(f["centroid"], 120.0) / 1000.0) * 1.0)
        vec.append(f["pan"] * (2.6 if has_stereo_cue else 0.0))
        rows.append(vec)
    X = np.asarray(rows, dtype=np.float64)

    # 距離行列から話者数を自動推定
    Z = linkage(X, method="average", metric="euclidean")
    max_dist = float(np.max(Z[:, 2])) if len(Z) else 0.0

    # 複数話者か単一話者かの判定
    if max_dist < 0.72:
        labels = np.ones(len(split_segs), dtype=int)
    else:
        # 距離しきい値で自然クラスタリングし、最大8話者に収める
        labels = fcluster(Z, t=0.88, criterion="distance")
        if len(set(labels.tolist())) > 8:
            labels = fcluster(Z, t=8, criterion="maxclust")
        # 映像に2名以上見えていて距離も中程度以上離れていれば最小2話者まで考慮
        if (
            len(set(labels.tolist())) == 1
            and vision_tracks
            and len(vision_tracks) >= 2
            and max_dist >= 0.62
        ):
            labels = fcluster(Z, t=2, criterion="maxclust")

    # 出現順に 話者A, 話者B, ... を割り当て
    order: dict[int, str] = {}
    turns: list[dict[str, Any]] = []
    for (s, e), lb in zip(split_segs, labels):
        lb_int = int(lb)
        if lb_int not in order:
            order[lb_int] = SPEAKER_NAMES[min(len(order), len(SPEAKER_NAMES) - 1)]
        turns.append({"start": round(float(s), 3), "end": round(float(e), 3), "speaker": order[lb_int]})
    return turns


# ---------------------------------------------------------------------------
# Step 2: 重なり分離強化＋10msクロスフェード時間振り分け
# ---------------------------------------------------------------------------
def _resolve_torch_device() -> tuple[str, str]:
    """config.get_torch_device() または torch バックエンドからデバイス (cuda/mps/cpu) を解決する。"""
    try:
        from app import config as ciness_config

        fn = getattr(ciness_config, "get_torch_device", None)
        if callable(fn):
            res = fn()
            if isinstance(res, tuple) and len(res) >= 2:
                return str(res[0]), str(res[1])
            if isinstance(res, str) and res:
                return res, res.upper()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "NVIDIA CUDA"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps", "Apple Silicon MPS"
    except Exception:
        pass
    return "cpu", "CPU"


def _try_load_local_sepformer() -> Any | None:
    """ローカル保存済みの SpeechBrain SepFormer がある場合のみ読み込む（暗黙DLしない）。"""
    sep_dir = MODELS_DIR / "sepformer" / "sepformer-whamr"
    required = ("hyperparams.yaml", "encoder.ckpt", "masknet.ckpt", "decoder.ckpt")
    if not sep_dir.is_dir() or not all((sep_dir / f).is_file() for f in required):
        return None
    try:
        from speechbrain.inference.separation import SepformerSeparation

        device, _ = _resolve_torch_device()
        if device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            return SepformerSeparation.from_hparams(
                source=str(sep_dir),
                savedir=str(sep_dir),
                run_opts={"device": device},
                local_strategy="copy",
            )
        except Exception:
            if device != "cpu":
                return SepformerSeparation.from_hparams(
                    source=str(sep_dir),
                    savedir=str(sep_dir),
                    run_opts={"device": "cpu"},
                    local_strategy="copy",
                )
            raise
    except Exception:
        return None


def _compute_speaker_templates(
    audio_stereo: np.ndarray,
    active_bits: np.ndarray,
    sr: int,
    present_speakers: list[int],
) -> dict[int, dict[str, Any]]:
    """各話者の単独区間から平均パワースペクトル・MFCC・ステレオ定位を学習する。"""
    n_fft = 1024
    hop = 256
    templates: dict[int, dict[str, Any]] = {}
    for sp_idx in present_speakers:
        solo_mask = active_bits == np.uint8(1 << sp_idx)
        indices = np.flatnonzero(solo_mask)
        if indices.size < 512:
            templates[sp_idx] = {"psd": np.ones(n_fft // 2 + 1, dtype=np.float32), "mfcc": np.zeros(13, dtype=np.float32), "pan": 0.0}
            continue
        # 最大5秒分を抽出して高速にプロファイル生成
        sample_clip = audio_stereo[indices[: sr * 5]]
        mono = sample_clip.mean(axis=1)
        _, _, Z = stft(mono, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        psd = np.mean(np.abs(Z) ** 2, axis=1).astype(np.float32)
        psd /= float(np.mean(psd)) + 1e-9
        rms_l = float(np.sqrt(np.mean(sample_clip[:, 0].astype(np.float64) ** 2)))
        rms_r = float(np.sqrt(np.mean(sample_clip[:, 1].astype(np.float64) ** 2)))
        pan = (rms_r - rms_l) / max(rms_r + rms_l, 1e-9)
        templates[sp_idx] = {
            "psd": psd,
            "mfcc": _extract_mfcc_vector(mono, sr),
            "pan": pan,
        }
    return templates


def _separate_overlap_slice_wiener(
    slice_stereo: np.ndarray,
    sr: int,
    sp_indices: list[int],
    templates: dict[int, dict[str, Any]],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """重なり発話スライスを、スペクトル＋ステレオ定位ソフトWienerマスクで各話者へ分配する。"""
    n = len(slice_stereo)
    mono = slice_stereo.mean(axis=1).astype(np.float32)
    if n < 256 or len(sp_indices) < 2:
        return {idx: np.zeros(n, dtype=np.float32) for idx in sp_indices}, mono

    n_fft = 1024
    hop = 256
    _, _, Z_l = stft(slice_stereo[:, 0], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z_r = stft(slice_stereo[:, 1], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    Z_m = 0.5 * (Z_l + Z_r)

    mag_l = np.abs(Z_l).astype(np.float32)
    mag_r = np.abs(Z_r).astype(np.float32)
    tf_pan = (mag_r - mag_l) / (mag_r + mag_l + 1e-8)

    # 話者間の分離しやすさ（スペクトル差・ステレオ定位差）から信頼度 alpha を決める
    pans = [templates[i]["pan"] for i in sp_indices]
    pan_spread = max(pans) - min(pans)
    psd_list = [templates[i]["psd"] for i in sp_indices]
    cos_sim = float(
        np.dot(psd_list[0], psd_list[1])
        / (float(np.linalg.norm(psd_list[0]) * np.linalg.norm(psd_list[1])) + 1e-9)
    )
    alpha = min(0.88, max(0.40, 0.45 + 0.45 * min(pan_spread, 1.0) + 0.30 * max(0.0, 1.0 - cos_sim)))

    scores = []
    for sp_idx in sp_indices:
        tpl = templates[sp_idx]
        spec_prior = tpl["psd"][:, None]
        if pan_spread > 0.15:
            pan_aff = np.exp(-2.5 * (tf_pan - tpl["pan"]) ** 2).astype(np.float32)
        else:
            pan_aff = 1.0
        scores.append(spec_prior * pan_aff + 1e-8)

    denom = sum(scores)
    separated: dict[int, np.ndarray] = {}
    for sp_idx, sc in zip(sp_indices, scores):
        mask = sc / denom
        _, y = istft(Z_m * mask, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        separated[sp_idx] = (alpha * y).astype(np.float32)

    residual_overlap = ((1.0 - alpha) * mono).astype(np.float32)
    return separated, residual_overlap


def _separate_overlap_slice_sepformer(
    slice_stereo: np.ndarray,
    sr: int,
    sp_pair: list[int],
    templates: dict[int, dict[str, Any]],
    sepformer: Any,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """SpeechBrain SepFormer による2話者音響分離＋48kHz帯域復元マスク帰属。"""
    import torch

    n = len(slice_stereo)
    mono = slice_stereo.mean(axis=1).astype(np.float32)
    mono_8k = _resample(mono[:, None], sr, 8000)[:, 0]
    if len(mono_8k) < 400:
        return _separate_overlap_slice_wiener(slice_stereo, sr, sp_pair, templates)

    try:
        with torch.inference_mode():
            wav_t = torch.from_numpy(mono_8k).unsqueeze(0)
            est = sepformer.separate_batch(wav_t)[0].detach().cpu().numpy()  # (samples_8k, 2)
    except Exception:
        return _separate_overlap_slice_wiener(slice_stereo, sr, sp_pair, templates)
    if est.ndim != 2 or est.shape[1] < 2:
        return _separate_overlap_slice_wiener(slice_stereo, sr, sp_pair, templates)

    s0_48 = _resample(est[:, 0:1], 8000, sr)[:, 0]
    s1_48 = _resample(est[:, 1:2], 8000, sr)[:, 0]
    s0_48 = s0_48[:n] if len(s0_48) >= n else np.pad(s0_48, (0, n - len(s0_48)))
    s1_48 = s1_48[:n] if len(s1_48) >= n else np.pad(s1_48, (0, n - len(s1_48)))

    # 8kHz由来出力から 48kHz STFT マスクを作って帯域劣化を最小化
    n_fft, hop = 1024, 256
    _, _, Zm = stft(mono, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z0 = stft(s0_48, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z1 = stft(s1_48, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    p0 = np.abs(Z0) ** 2 + 1e-10
    p1 = np.abs(Z1) ** 2 + 1e-10
    m0 = p0 / (p0 + p1)
    m1 = p1 / (p0 + p1)
    _, y0 = istft(Zm * m0, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, y1 = istft(Zm * m1, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    y0 = (y0[:n] if len(y0) >= n else np.pad(y0, (0, n - len(y0)))).astype(np.float32)
    y1 = (y1[:n] if len(y1) >= n else np.pad(y1, (0, n - len(y1)))).astype(np.float32)

    # MFCC 相関で sp_pair[0] / sp_pair[1] へ帰属
    m_y0 = _extract_mfcc_vector(y0, sr)
    m_y1 = _extract_mfcc_vector(y1, sr)
    t0 = templates[sp_pair[0]]["mfcc"]
    t1 = templates[sp_pair[1]]["mfcc"]

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    direct = _cos(m_y0, t0) + _cos(m_y1, t1)
    swapped = _cos(m_y0, t1) + _cos(m_y1, t0)
    if direct >= swapped:
        out = {sp_pair[0]: y0, sp_pair[1]: y1}
    else:
        out = {sp_pair[0]: y1, sp_pair[1]: y0}
    return out, np.zeros(n, dtype=np.float32)


def _split_audio_with_overlap_refinement(
    audio_stereo: np.ndarray,
    sr: int,
    turns: list[dict[str, Any]],
    quality: str = "standard",
    fade_ms: float = 10.0,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, str]:
    """SoundSeparator の 10ms クロスフェード時間振り分け＋重なり発話の分離強化を実行する。"""
    n_samples = len(audio_stereo)
    mono = audio_stereo.mean(axis=1).astype(np.float32)

    # ビットマスクで各サンプルのアクティブ話者集合を管理（SoundSeparator diarization_stems 方式）
    active_bits = np.zeros(n_samples, dtype=np.uint8)
    present_set: set[int] = set()
    for turn in turns:
        sp = str(turn.get("speaker", ""))
        if len(sp) == 3 and sp[:2] == "話者" and sp[2] in SPEAKER_LETTERS:
            idx = SPEAKER_LETTERS.index(sp[2])
        else:
            idx = 0
        s = min(n_samples, max(0, int(round(float(turn["start"]) * sr))))
        e = min(n_samples, max(0, int(round(float(turn["end"]) * sr))))
        if s < e:
            active_bits[s:e] |= np.uint8(1 << idx)
            present_set.add(idx)

    if not present_set:
        return {0: mono.copy()}, np.zeros(n_samples, dtype=np.float32), np.zeros(n_samples, dtype=np.float32), "none"

    present_speakers = sorted(present_set)

    # 0=residual, 1..8=solo speaker, 9=overlap
    lut = np.full(256, 9, dtype=np.uint8)
    lut[0] = 0
    for idx in range(8):
        lut[1 << idx] = idx + 1
    classes = lut[active_bits]

    # 境界での 10ms クロスフェード遷移リスト作成
    boundaries = np.flatnonzero(classes[1:] != classes[:-1]) + 1
    fade_samples = int(round(fade_ms * sr / 1000.0))
    half = fade_samples // 2
    transitions: list[tuple[int, int, int, int]] = []
    if half:
        for i, boundary in enumerate(boundaries):
            prev_b = int(boundaries[i - 1]) if i else 0
            next_b = int(boundaries[i + 1]) if i + 1 < len(boundaries) else n_samples
            b = int(boundary)
            left = min(half, (b - prev_b) // 2)
            right = min(half, (next_b - b) // 2)
            if (b + right) - (b - left) >= 2:
                transitions.append((b - left, b + right, int(classes[b - 1]), int(classes[b])))

    def _category_gain(category: int) -> np.ndarray:
        gain = (classes == category).astype(np.float32)
        for s, e, before, after in transitions:
            if category == before or category == after:
                ramp = np.linspace(0.0, 1.0, e - s, dtype=np.float32)
                gain[s:e] = (1.0 - ramp) if category == before else ramp
        return gain

    speaker_tracks: dict[int, np.ndarray] = {}
    for sp_idx in present_speakers:
        speaker_tracks[sp_idx] = (mono * _category_gain(sp_idx + 1)).astype(np.float32)
    residual_track = (mono * _category_gain(0)).astype(np.float32)
    overlap_gain = _category_gain(9)
    overlap_track = np.zeros(n_samples, dtype=np.float32)

    # 重なり区間の分離強化
    overlap_mask = classes == 9
    overlap_method = "none"
    if np.any(overlap_mask):
        templates = _compute_speaker_templates(audio_stereo, active_bits, sr, present_speakers)
        sepformer = _try_load_local_sepformer() if quality in ("standard", "high") else None
        overlap_method = "sepformer_local" if sepformer is not None else "wiener_spectral_pan"

        # 同一の active_bits を持つ連続重なり区間ごとに処理
        ov_indices = np.flatnonzero(overlap_mask)
        if ov_indices.size > 0:
            splits = np.flatnonzero(
                (np.diff(ov_indices) > 1) | (active_bits[ov_indices[1:]] != active_bits[ov_indices[:-1]])
            ) + 1
            runs = np.split(ov_indices, splits)
            for run in runs:
                s = int(run[0])
                e = int(run[-1]) + 1
                bits = int(active_bits[s])
                sp_list = [i for i in present_speakers if bits & (1 << i)]
                # 境界のフェード窓込みで滑らかに加算
                pad = min(half, s, n_samples - e)
                ps, pe = s - pad, e + pad
                slice_st = audio_stereo[ps:pe]
                if sepformer is not None and len(sp_list) == 2:
                    sep_map, ov_res = _separate_overlap_slice_sepformer(slice_st, sr, sp_list, templates, sepformer)
                else:
                    sep_map, ov_res = _separate_overlap_slice_wiener(slice_st, sr, sp_list, templates)
                g_win = overlap_gain[ps:pe]
                for sp_idx, y_sep in sep_map.items():
                    speaker_tracks[sp_idx][ps:pe] += y_sep * g_win
                overlap_track[ps:pe] += ov_res * g_win

    return speaker_tracks, overlap_track, residual_track, overlap_method


# ---------------------------------------------------------------------------
# Step 3: 話者 ↔ 映像上の人物の自動紐付け（ハンガリアン法）
# ---------------------------------------------------------------------------
def _interpolate_person_series(
    track: dict[str, Any],
    grid_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """人物トラックの口元活性度と可視マスクを共通時間軸へ補間する。"""
    kfs = track.get("keyframes", [])
    if not kfs:
        return np.zeros_like(grid_t), np.zeros_like(grid_t)
    ts = np.array([float(k["t"]) for k in kfs], dtype=np.float64)
    mouth = np.array([float(k.get("mouth_activity", 0.0)) for k in kfs], dtype=np.float64)
    vis = ((grid_t >= ts[0] - 0.35) & (grid_t <= ts[-1] + 0.35)).astype(np.float64)
    interp_mouth = np.interp(grid_t, ts, mouth, left=0.0, right=0.0) * vis
    return interp_mouth, vis


def _speaker_activity_curve(
    mono: np.ndarray,
    sr: int,
    turns: list[dict[str, Any]],
    speaker_name: str,
    grid_t: np.ndarray,
    dt: float,
) -> np.ndarray:
    """話者の発話エンベロープ（0..1）を時間グリッド上で計算する。"""
    curve = np.zeros(len(grid_t), dtype=np.float64)
    win = max(1, int(round(dt * sr)))
    for i, t in enumerate(grid_t):
        s = int(round(t * sr))
        e = min(len(mono), s + win)
        if s < e:
            curve[i] = float(np.sqrt(np.mean(mono[s:e].astype(np.float64) ** 2)))
    peak = float(np.max(curve))
    if peak > 1e-8:
        curve /= peak
    # 発話ターン外は抑えてノイズ相関を防ぐ
    turn_mask = np.zeros(len(grid_t), dtype=np.float64)
    for turn in turns:
        if turn["speaker"] == speaker_name:
            turn_mask[(grid_t >= turn["start"] - dt) & (grid_t <= turn["end"] + dt)] = 1.0
    if np.any(turn_mask > 0):
        curve = curve * (0.2 + 0.8 * turn_mask)
    return curve


def _match_speakers_to_persons(
    speaker_items: list[dict[str, Any]],
    vision_tracks: list[dict[str, Any]] | None,
    duration_sec: float,
    dialog_pan_mode: str = "cinema",
) -> None:
    """口元モーション活性度の時間相関＋ステレオ定位補助で話者↔人物を自動対応付ける。"""
    mode = _normalize_dialog_pan_mode(dialog_pan_mode)
    tracks = vision_tracks or []
    n_sp = len(speaker_items)
    n_p = len(tracks)

    dt = 0.1
    grid_t = np.arange(0.0, max(duration_sec, dt), dt, dtype=np.float64)
    matched_person_by_sp: dict[int, tuple[int, float]] = {}

    if n_sp > 0 and n_p > 0 and len(grid_t) >= 2:
        person_mouths = [_interpolate_person_series(tr, grid_t) for tr in tracks]
        scores = np.full((n_sp, n_p), -1.0, dtype=np.float64)

        for si, sp in enumerate(speaker_items):
            act = sp["_activity_curve"]
            active_mask = act > 0.18
            inactive_mask = ~active_mask
            pan_bias = float(sp.get("_stereo_pan", 0.0))

            for pi, (mouth, vis) in enumerate(person_mouths):
                if np.any(active_mask):
                    vis_overlap = float(np.mean(vis[active_mask]))
                else:
                    vis_overlap = float(np.mean(vis))
                if vis_overlap < 0.08:
                    scores[si, pi] = -0.5
                    continue

                # 1) 口元活性度のコントラスト（話している時 − 黙っている時）
                m_on = float(np.mean(mouth[active_mask])) if np.any(active_mask) else 0.0
                m_off = float(np.mean(mouth[inactive_mask])) if np.any(inactive_mask) else 0.0
                contrast = max(-1.0, min(1.0, (m_on - m_off) * 5.0))

                # 2) ピアソン相関係数
                act_c = act - float(np.mean(act))
                mouth_c = mouth - float(np.mean(mouth))
                denom = float(np.linalg.norm(act_c) * np.linalg.norm(mouth_c))
                corr = float(np.dot(act_c, mouth_c) / denom) if denom > 1e-8 else 0.0

                # 3) ステレオ入力の左右パン差と画面左右位置の整合補助
                az_norm = float(tracks[pi].get("mean_azimuth_deg", 0.0)) / 45.0
                pan_score = max(-0.35, min(0.35, pan_bias * az_norm * 0.45)) if abs(pan_bias) > 0.12 else 0.0

                scores[si, pi] = 0.45 * corr + 0.35 * contrast + 0.20 * vis_overlap + pan_score

        row_ind, col_ind = linear_sum_assignment(-scores)
        for si, pi in zip(row_ind, col_ind):
            sc = float(scores[si, pi])
            if sc > 0.02 or (n_sp == 1 and n_p == 1 and sc > -0.2):
                conf = round(max(0.15, min(0.99, 0.5 + 0.5 * sc)), 3)
                matched_person_by_sp[si] = (pi, conf)

    # オフスクリーン話者配置用の水平角プリセット（cinema / immersive）
    unmatched_indices = [i for i in range(n_sp) if i not in matched_person_by_sp]
    if len(unmatched_indices) >= 2:
        cinema_slot = [-14.0, 14.0, -20.0, 20.0, -10.0, 10.0, -22.0, 22.0]
        immersive_slot = [-22.0, 22.0, -34.0, 34.0, -42.0, 42.0, -12.0, 12.0]
    else:
        cinema_slot = [0.0]
        immersive_slot = [0.0]
    offscreen_slot = cinema_slot if mode == "cinema" else immersive_slot

    off_counter = 0
    for si, sp in enumerate(speaker_items):
        if si in matched_person_by_sp:
            pi, conf = matched_person_by_sp[si]
            person = tracks[pi]
            raw_kfs = [
                {
                    "t": float(k["t"]),
                    "azimuth_deg": float(k["azimuth_deg"]),
                    "elevation_deg": float(k["elevation_deg"]),
                    "distance_m": float(k["distance_m"]),
                }
                for k in person.get("keyframes", [])
            ]
            if not raw_kfs:
                raw_kfs = [{
                    "t": 0.0,
                    "azimuth_deg": float(person.get("mean_azimuth_deg", 0.0)),
                    "elevation_deg": float(person.get("mean_elevation_deg", 0.0)),
                    "distance_m": float(person.get("mean_distance_m", 2.0)),
                }]
            elif raw_kfs[0]["t"] > 0.05:
                raw_kfs.insert(0, {**raw_kfs[0], "t": 0.0})

            kfs = apply_dialog_pan_mode_to_keyframes(raw_kfs, mode)
            sp["matched_person_id"] = person["person_id"]
            sp["matched_person_label"] = person.get("label", person["person_id"])
            sp["match_confidence"] = conf
            sp["is_offscreen"] = False
            sp["raw_keyframes"] = raw_kfs
            sp["keyframes"] = kfs
            sp["mean_azimuth_deg"] = round(float(np.mean([k["azimuth_deg"] for k in kfs])), 2)
            sp["mean_elevation_deg"] = round(float(np.mean([k["elevation_deg"] for k in kfs])), 2)
            sp["mean_distance_m"] = float(person.get("mean_distance_m", 2.0))
        else:
            az_raw = immersive_slot[off_counter % len(immersive_slot)]
            az = offscreen_slot[off_counter % len(offscreen_slot)]
            dist = 1.6 if len(unmatched_indices) == 1 else 2.0
            off_counter += 1
            sp["matched_person_id"] = None
            sp["matched_person_label"] = "オフスクリーン / ナレーション"
            sp["match_confidence"] = 0.0
            sp["is_offscreen"] = True
            sp["raw_keyframes"] = [{"t": 0.0, "azimuth_deg": az_raw, "elevation_deg": 0.0, "distance_m": dist}]
            sp["keyframes"] = [{"t": 0.0, "azimuth_deg": az, "elevation_deg": 0.0, "distance_m": dist}]
            sp["mean_azimuth_deg"] = az
            sp["mean_elevation_deg"] = 0.0
            sp["mean_distance_m"] = dist


def split_speakers(
    dialog_wav: str | Path,
    out_dir: str | Path,
    vision_tracks: list[dict[str, Any]] | None = None,
    quality: str = "standard",
    force_fallback: bool = False,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
    dialog_pan_mode: str = "cinema",
) -> dict[str, Any]:
    """セリフWAVを話者ごとの48kHzモノラルWAVへ分離し、映像人物と自動紐付けする。"""
    src = Path(dialog_wav)
    dst_dir = Path(out_dir)
    if not src.is_file():
        raise FileNotFoundError(f"セリフWAVが見つかりません: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    norm_pan_mode = _normalize_dialog_pan_mode(dialog_pan_mode)

    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        audio = _resample(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if audio.shape[1] == 1:
        audio_stereo = np.repeat(audio, 2, axis=1)
    else:
        audio_stereo = audio[:, :2]

    duration_sec = len(audio_stereo) / sr
    if _rms_db(audio_stereo) < -72.0:
        _emit_progress(progress_cb, 100, "セリフ成分がほぼ無音のため話者分離をスキップしました")
        return {
            "method": "silent_bypass",
            "overlap_method": "none",
            "dialog_pan_mode": norm_pan_mode,
            "speakers": [],
            "turns": [],
            "overlap_wav": None,
            "residual_wav": None,
            "sample_rate": SAMPLE_RATE,
        }

    # --- Step 1: 話者ダイアリゼーション ---
    ss_info = detect_soundseparator()
    method = "acoustic_clustering"
    turns: list[dict[str, Any]] = []

    if not force_fallback and ss_info["available"] and ss_info["models"].get("nemotron_diarization"):
        try:
            turns = _run_nemotron_diarizer(src, ss_info, progress_cb=progress_cb, log_cb=log_cb)
            if turns:
                method = "nemotron_diarization"
            else:
                _emit_log(log_cb, "Nemotron で発話区間が検出されなかったため音響クラスタリングへ切替えます。")
        except Exception as exc:
            _emit_log(log_cb, f"Nemotron 実行に失敗したため音響クラスタリングへ切替えます: {exc}")

    if not turns:
        method = "acoustic_clustering"
        turns = _diarize_acoustic_clustering(
            audio_stereo, sr, vision_tracks=vision_tracks, progress_cb=progress_cb, log_cb=log_cb
        )

    # --- Step 2: 10msクロスフェード時間振り分け＋重なり分離強化 ---
    _emit_progress(progress_cb, 55, "話者ごとの単独モノラルWAVと重なり区間を分離中...")
    speaker_tracks, overlap_track, residual_track, overlap_method = _split_audio_with_overlap_refinement(
        audio_stereo, sr, turns, quality=quality, fade_ms=10.0
    )

    dt = 0.1
    grid_t = np.arange(0.0, max(duration_sec, dt), dt, dtype=np.float64)
    speaker_items: list[dict[str, Any]] = []

    for sp_idx, mono_wav in sorted(speaker_tracks.items()):
        letter = SPEAKER_LETTERS[sp_idx]
        sp_id = f"speaker_{letter.lower()}"
        sp_label = SPEAKER_NAMES[sp_idx]
        wav_path = dst_dir / f"{sp_id}.wav"
        # 48kHz モノラル（点音源配置用）で書き出し
        sf.write(str(wav_path), mono_wav.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")

        sp_turns = [t for t in turns if t["speaker"] == sp_label]
        dur_speaking = round(sum(t["end"] - t["start"] for t in sp_turns), 3)

        # 元ステレオ内での話者の左右パン傾向（マッチング補助）
        pan_vals = []
        for t in sp_turns:
            s_i = max(0, int(round(t["start"] * sr)))
            e_i = min(len(audio_stereo), int(round(t["end"] * sr)))
            if e_i > s_i:
                rl = float(np.sqrt(np.mean(audio_stereo[s_i:e_i, 0].astype(np.float64) ** 2)))
                rr = float(np.sqrt(np.mean(audio_stereo[s_i:e_i, 1].astype(np.float64) ** 2)))
                pan_vals.append((rr - rl) / max(rr + rl, 1e-9))
        stereo_pan = float(np.mean(pan_vals)) if pan_vals else 0.0

        speaker_items.append({
            "id": sp_id,
            "label": sp_label,
            "wav_path": str(wav_path.resolve()),
            "rms_db": _rms_db(mono_wav),
            "turns": sp_turns,
            "duration_speaking_sec": dur_speaking,
            "_stereo_pan": stereo_pan,
            "_activity_curve": _speaker_activity_curve(mono_wav, sr, turns, sp_label, grid_t, dt),
        })

    overlap_wav_path: str | None = None
    if _rms_db(overlap_track) > -75.0:
        ov_p = dst_dir / "dialog_overlap.wav"
        sf.write(str(ov_p), overlap_track.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        overlap_wav_path = str(ov_p.resolve())

    residual_wav_path: str | None = None
    if _rms_db(residual_track) > -68.0:
        res_p = dst_dir / "dialog_residual.wav"
        sf.write(str(res_p), residual_track.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        residual_wav_path = str(res_p.resolve())

    # --- Step 3: 話者 ↔ 映像人物の自動紐付け ---
    _emit_progress(progress_cb, 82, "口元モーション相関から話者と映像上の人物を自動紐付け中...")
    _match_speakers_to_persons(speaker_items, vision_tracks, duration_sec, dialog_pan_mode=norm_pan_mode)

    # 内部作業キーを掃除してクリーンな返却JSONにする
    for sp in speaker_items:
        sp.pop("_activity_curve", None)
        sp.pop("_stereo_pan", None)

    matched_count = sum(1 for sp in speaker_items if not sp.get("is_offscreen"))
    _emit_log(
        log_cb,
        f"話者分離完了（方式={method}, 話者数={len(speaker_items)}, 映像人物紐付け={matched_count}名, "
        f"定位モード={norm_pan_mode}, 重なり処理={overlap_method}）。",
    )
    _emit_progress(progress_cb, 100, f"話者分離完了（{len(speaker_items)}名）")

    return {
        "method": method,
        "overlap_method": overlap_method,
        "dialog_pan_mode": norm_pan_mode,
        "speakers": speaker_items,
        "turns": turns,
        "overlap_wav": overlap_wav_path,
        "residual_wav": residual_wav_path,
        "overlap_rms_db": _rms_db(overlap_track),
        "residual_rms_db": _rms_db(residual_track),
        "sample_rate": SAMPLE_RATE,
    }

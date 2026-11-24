# coding: utf-8
"""音声抽出・音源3分離（DME）＆残響除去ブリッジ。

隣接する SoundSeparator フォルダまたは環境変数 SOUNDSEPARATOR_DIR が利用可能で
BandIt Plus / DBS-TIGER-DnR が取得済みなら、サブプロセス経由で高品質な
DME 3分離（dialog / effect / music）と任意の MelBand Dereverb を実行する。
単体環境・モデル未導入・CPU軽量フォールバック時は、NumPy/SciPy による
完全再構成型 STFT 3分離（`method: "dsp_fallback"`）で破綻なく動作する。
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import istft, resample_poly, stft

from app.config import SAMPLE_RATE, find_soundseparator_dir

_PCT_RE = re.compile(r"\[\s*(\d+)%\s*\]\s*(.*)")


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
            log_cb("separation", message)
        except Exception:
            pass


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません。FFmpeg をインストールして PATH に通してください。")
    return ffmpeg


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """(samples, channels) float32 を安全にリサンプルする。"""
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    factor = gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // factor
    down = int(src_sr) // factor
    return resample_poly(audio.astype(np.float32), up, down, axis=0).astype(np.float32)


def _measure_audio(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {"rms_db": -120.0, "peak_db": -120.0}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    return {
        "rms_db": round(20.0 * math.log10(rms), 2) if rms > 1e-10 else -120.0,
        "peak_db": round(20.0 * math.log10(peak), 2) if peak > 1e-10 else -120.0,
    }


def extract_audio_from_video(
    video_path: str | Path,
    out_wav: str | Path,
    sample_rate: int = SAMPLE_RATE,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """動画から 48kHz ステレオ 32bit float WAV を抽出する。"""
    src = Path(video_path)
    dst = Path(out_wav)
    if not src.is_file():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_ffmpeg()
    cmd = [ffmpeg, "-y", "-v", "error"]
    if start_sec and start_sec > 0:
        cmd += ["-ss", f"{float(start_sec):.3f}"]
    cmd += ["-i", str(src)]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{float(duration_sec):.3f}"]
    cmd += [
        "-vn",
        "-acodec", "pcm_f32le",
        "-ar", str(int(sample_rate)),
        "-ac", "2",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"動画からの音声抽出に失敗しました: {proc.stderr.strip()}")

    audio, sr = sf.read(str(dst), dtype="float32", always_2d=True)
    metrics = _measure_audio(audio)
    return {
        "wav_path": str(dst.resolve()),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "samples": int(audio.shape[0]),
        "duration_sec": round(audio.shape[0] / max(sr, 1), 3),
        **metrics,
    }


def detect_soundseparator() -> dict[str, Any]:
    """隣接する SoundSeparator 環境と共有済みモデルの有無を高速に確認する。"""
    ss_dir = find_soundseparator_dir()
    if ss_dir is None:
        return {"available": False, "dir": None, "python": None, "control_script": None, "models": {}}

    py_candidates = [
        ss_dir / ".venv" / "Scripts" / "python.exe",
        ss_dir / ".venv" / "bin" / "python",
    ]
    ss_python = next((p for p in py_candidates if p.is_file()), None)
    control_py = ss_dir / "tools" / "soundsep_control.py"
    models_dir = ss_dir / "models"

    def _ok(rel: str) -> bool:
        p = models_dir / rel
        return p.is_file() and p.stat().st_size > 0

    models = {
        "bandit": _ok("model_bandit_plus_dnr_sdr_11.47.chpt"),
        "dbs_dnr": _ok("dbs_tiger_dnr_config.json") and _ok("dbs_tiger_dnr.safetensors"),
        "cdx23": all(
            _ok(name)
            for name in ("97d170e1-a778de4a.th", "97d170e1-dbb4db15.th", "97d170e1-e41a5468.th")
        ),
        "dereverb": (
            _ok("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")
            and _ok("dereverb_mel_band_roformer_anvuew.yaml")
        ),
        "nemotron_diarization": all(
            _ok(f"nemotron-3-diarization/{name}")
            for name in ("config.json", "processor_config.json", "model.safetensors")
        ),
    }

    return {
        "available": bool(ss_python and control_py.is_file()),
        "dir": str(ss_dir),
        "python": str(ss_python) if ss_python else None,
        "control_script": str(control_py) if control_py.is_file() else None,
        "models": models,
    }


def _run_soundseparator_dme(
    input_wav: Path,
    out_dir: Path,
    ss_info: dict[str, Any],
    quality: str,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """SoundSeparator の CLI (`soundsep_control.py separate --mode dme`) をサブプロセス実行する。"""
    models = ss_info["models"]
    model_id = "bandit" if models.get("bandit") else "dbs_dnr"

    # SoundSeparator の dme quality=high は CDX23 を要求するため、未取得時は standard に安全調整
    if quality == "high" and models.get("cdx23") and model_id == "bandit":
        ss_quality = "high"
    elif quality == "fast":
        ss_quality = "fast"
    else:
        ss_quality = "standard"
        if quality == "high" and not models.get("cdx23"):
            _emit_log(log_cb, "SoundSeparator の CDX23 が未取得のため、BandIt Plus（standard）で高品質DME分離を実行します。")

    work_sub = out_dir / "_soundsep_dme_tmp"
    if work_sub.exists():
        shutil.rmtree(work_sub, ignore_errors=True)

    cmd = [
        ss_info["python"],
        "-X", "utf8",
        ss_info["control_script"],
        "separate",
        "--input", str(input_wav),
        "--out", str(work_sub),
        "--mode", "dme",
        "--quality", ss_quality,
        "--model", model_id,
    ]
    _emit_log(log_cb, f"SoundSeparator ({model_id}, quality={ss_quality}) で音声3分離（DME）を開始します。")
    _emit_progress(progress_cb, 5, f"SoundSeparator ({model_id}) 起動中...")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=ss_info["dir"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    stderr_lines: list[str] = []
    assert proc.stderr is not None
    for raw_line in proc.stderr:
        line = raw_line.strip()
        if not line:
            continue
        stderr_lines.append(line)
        m = _PCT_RE.match(line)
        if m:
            pct = int(m.group(1))
            msg = m.group(2).strip() or "DME分離中..."
            _emit_progress(progress_cb, min(78, int(pct * 0.78)), msg)

    stdout_text = proc.stdout.read() if proc.stdout else ""
    returncode = proc.wait(timeout=1800)
    if returncode != 0:
        tail = " | ".join(stderr_lines[-5:])
        raise RuntimeError(f"SoundSeparator DME分離が異常終了しました (code={returncode}): {tail}")

    manifest = {}
    try:
        manifest = json.loads(stdout_text) if stdout_text.strip() else {}
    except json.JSONDecodeError:
        manifest_path = work_sub / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 出力3ステムを 48kHz ステレオ 32bit float WAV に揃えて out_dir へ配置
    stems_out: dict[str, str] = {}
    stem_metrics: dict[str, dict[str, float]] = {}
    for stem_name in ("dialog", "effect", "music"):
        src_stem = work_sub / f"{stem_name}.wav"
        if not src_stem.is_file():
            raise RuntimeError(f"SoundSeparator の出力ステムが見つかりません: {src_stem}")
        data, sr = sf.read(str(src_stem), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            data = _resample_audio(data, sr, SAMPLE_RATE)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        dst_stem = out_dir / f"{stem_name}.wav"
        sf.write(str(dst_stem), data, SAMPLE_RATE, subtype="FLOAT")
        stems_out[f"{stem_name}_wav"] = str(dst_stem.resolve())
        stem_metrics[stem_name] = _measure_audio(data)

    shutil.rmtree(work_sub, ignore_errors=True)
    return {
        **stems_out,
        "method": f"soundseparator_{model_id}",
        "model_id": model_id,
        "soundsep_quality": ss_quality,
        "device": manifest.get("device", "unknown"),
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "stem_metrics": stem_metrics,
    }


def _run_soundseparator_dereverb(
    dialog_wav: Path,
    ss_info: dict[str, Any],
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> bool:
    """SoundSeparator の MelBand Dereverb をサブプロセス経由で dialog.wav に適用する。

    SoundSeparator 配下のファイルは一切変更せず、import 呼び出しのみ行う。
    """
    _emit_progress(progress_cb, 82, "セリフの残響を軽減中 (MelBand Dereverb)...")
    _emit_log(log_cb, "SoundSeparator の MelBand RoFormer Dereverb で話し声の残響を除去しています。")

    wet_backup = dialog_wav.with_name("dialog_wet.wav")
    shutil.copy2(dialog_wav, wet_backup)
    dry_tmp = dialog_wav.with_name("dialog_dry_tmp.wav")

    inline_code = r"""
import json, sys
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf

ss_dir = Path(sys.argv[1])
in_wav = Path(sys.argv[2])
out_wav = Path(sys.argv[3])
sys.path.insert(0, str(ss_dir / "app"))

real_stdout, sys.stdout = sys.stdout, sys.stderr
import separator

audio, sr = sf.read(str(in_wav), dtype="float32", always_2d=True)
orig_len = audio.shape[0]
if audio.shape[1] == 1:
    audio = np.repeat(audio, 2, axis=1)
elif audio.shape[1] > 2:
    audio = audio[:, :2]

def _rs(x, a, b):
    if a == b:
        return x
    try:
        import soxr
        return soxr.resample(x, a, b, quality="HQ").astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        g = gcd(a, b)
        return resample_poly(x, b // g, a // g, axis=0).astype(np.float32)

in_44 = np.ascontiguousarray(_rs(audio, sr, 44100).T)
eng = separator.Engine()
dry_44 = eng.dereverb(in_44)
dry = _rs(dry_44.T, 44100, sr)
if dry.shape[0] > orig_len:
    dry = dry[:orig_len]
elif dry.shape[0] < orig_len:
    dry = np.pad(dry, ((0, orig_len - dry.shape[0]), (0, 0)))
sf.write(str(out_wav), dry.astype(np.float32), sr, subtype="FLOAT")
print(json.dumps({"ok": True, "device": eng.device}), file=real_stdout)
"""

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [ss_info["python"], "-X", "utf8", "-c", inline_code, ss_info["dir"], str(dialog_wav), str(dry_tmp)],
        cwd=ss_info["dir"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        env=env,
    )
    if proc.returncode != 0 or not dry_tmp.is_file():
        dry_tmp.unlink(missing_ok=True)
        _emit_log(log_cb, f"残響除去をスキップしました（元のセリフを維持）: {proc.stderr.strip()[-200:]}")
        return False

    os.replace(dry_tmp, dialog_wav)
    _emit_progress(progress_cb, 96, "残響除去が完了しました")
    return True


def _separate_dme_dsp_fallback(
    input_wav: Path,
    out_dir: Path,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """NumPy/SciPy によるローカルDSP 3分離（完全再構成 Wiener ソフトマスク）。

    - HPSS（時間方向/周波数方向メディアンフィルタ）で打撃・トランジェントを効果音へ分離
    - 音声帯域（130Hz〜4.2kHz）の調和成分を左右パンを壊さず dialog へ保持
    - 低域/高域の持続成分・定常ステレオ広がり成分を music へ分離
    - 3マスクの和を 1.0 に正規化し、エネルギー欠損・破綻を防止する
    """
    _emit_log(log_cb, "ローカルDSPフォールバック（STFT + HPSS + 帯域/ステレオ分離）で3分離を実行します。")
    _emit_progress(progress_cb, 15, "ローカルDSP 3分離を解析中...")

    audio, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        audio = _resample_audio(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    n_samples = audio.shape[0]
    n_fft = 2048
    hop = 512

    freqs, _, Z_l = stft(audio[:, 0], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z_r = stft(audio[:, 1], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)

    mag_l = np.abs(Z_l).astype(np.float32)
    mag_r = np.abs(Z_r).astype(np.float32)
    mag = np.sqrt(0.5 * (mag_l ** 2 + mag_r ** 2) + 1e-12)

    # HPSS（調和・打撃分離）
    harm = median_filter(mag, size=(1, 15))
    perc = median_filter(mag, size=(15, 1))
    hp_sum = harm + perc + 1e-12
    h_ratio = harm / hp_sum
    p_ratio = perc / hp_sum

    _emit_progress(progress_cb, 50, "帯域・調和/打撃マスクを構築中...")

    # 周波数帯域の滑らかな重み（低音男性声の基音 90Hz〜明瞭子音 4.5kHz を保護）
    f = freqs[:, None].astype(np.float32)
    low_roll = 1.0 / (1.0 + np.exp(-(f - 90.0) / 16.0))
    high_roll = 1.0 / (1.0 + np.exp((f - 4500.0) / 380.0))
    voice_band = low_roll * high_roll  # 95Hz..4.5kHz で約 1.0

    # 持続的なステレオ広がり成分（BGM特徴）のみ抽出（左右に定位した短い話し声は保護）
    side_mag = np.abs(Z_l - Z_r).astype(np.float32) * 0.5
    sustained_side = median_filter(side_mag, size=(1, 31))
    non_voice_band = np.clip(1.0 - voice_band, 0.0, 1.0)

    # 3ステムのエネルギースコア（Wiener パワー比）
    p_dialog = (voice_band * (0.80 * harm + 0.20 * mag) * (0.35 + 0.65 * h_ratio)) ** 2
    p_effect = (
        (perc * p_ratio * (0.45 + 0.55 * (f > 1800.0).astype(np.float32)))
        + 0.12 * mag * (f > 5500.0).astype(np.float32) * p_ratio
    ) ** 2
    p_music = (
        non_voice_band * harm * (0.55 + 0.45 * h_ratio)
        + 0.22 * sustained_side * non_voice_band
        + 0.08 * sustained_side * (f < 250.0).astype(np.float32)
    ) ** 2

    total = p_dialog + p_effect + p_music + 1e-12
    m_dialog = p_dialog / total
    m_effect = p_effect / total
    m_music = p_music / total

    _emit_progress(progress_cb, 75, "3ステム波形を再合成中...")

    def _reconstruct(mask: np.ndarray) -> np.ndarray:
        _, y_l = istft(Z_l * mask, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, y_r = istft(Z_r * mask, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        out = np.stack([y_l, y_r], axis=1).astype(np.float32)
        if out.shape[0] > n_samples:
            out = out[:n_samples]
        elif out.shape[0] < n_samples:
            out = np.pad(out, ((0, n_samples - out.shape[0]), (0, 0)))
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    stems = {
        "dialog": _reconstruct(m_dialog),
        "effect": _reconstruct(m_effect),
        "music": _reconstruct(m_music),
    }

    stems_out: dict[str, str] = {}
    stem_metrics: dict[str, dict[str, float]] = {}
    for name, data in stems.items():
        dst = out_dir / f"{name}.wav"
        sf.write(str(dst), data, SAMPLE_RATE, subtype="FLOAT")
        stems_out[f"{name}_wav"] = str(dst.resolve())
        stem_metrics[name] = _measure_audio(data)

    _emit_progress(progress_cb, 100, "ローカルDSP 3分離が完了しました")
    return {
        **stems_out,
        "method": "dsp_fallback",
        "model_id": "dsp_hpss_band",
        "dereverb_applied": False,
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "stem_metrics": stem_metrics,
    }


def separate_dme(
    input_wav: str | Path,
    out_dir: str | Path,
    quality: str = "standard",
    enable_dereverb: bool = True,
    force_fallback: bool = False,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """音声を dialog / effect / music の3ステムに分離し、任意で dialog に残響除去を適用する。"""
    src = Path(input_wav)
    dst_dir = Path(out_dir)
    if not src.is_file():
        raise FileNotFoundError(f"入力WAVが見つかりません: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # ほぼ完全無音ならモデルを立ち上げず高速スキップ
    probe_audio, probe_sr = sf.read(str(src), dtype="float32", always_2d=True)
    probe_metrics = _measure_audio(probe_audio)
    if probe_metrics["rms_db"] < -80.0:
        if probe_sr != SAMPLE_RATE:
            probe_audio = _resample_audio(probe_audio, probe_sr, SAMPLE_RATE)
        zeros = np.zeros((probe_audio.shape[0], 2), dtype=np.float32)
        out_paths = {}
        for name in ("dialog", "effect", "music"):
            p = dst_dir / f"{name}.wav"
            sf.write(str(p), zeros, SAMPLE_RATE, subtype="FLOAT")
            out_paths[f"{name}_wav"] = str(p.resolve())
        _emit_progress(progress_cb, 100, "無音のため3分離をスキップしました")
        return {
            **out_paths,
            "method": "silent_bypass",
            "model_id": "none",
            "dereverb_applied": False,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "stem_metrics": {n: {"rms_db": -120.0, "peak_db": -120.0} for n in ("dialog", "effect", "music")},
        }

    ss_info = detect_soundseparator()
    has_dme_model = bool(
        ss_info["available"]
        and (ss_info["models"].get("bandit") or ss_info["models"].get("dbs_dnr"))
    )

    if not force_fallback and has_dme_model:
        try:
            result = _run_soundseparator_dme(
                src, dst_dir, ss_info, quality=quality, progress_cb=progress_cb, log_cb=log_cb
            )
            dereverb_applied = False
            if (
                enable_dereverb
                and quality in ("standard", "high")
                and ss_info["models"].get("dereverb")
                and result["stem_metrics"]["dialog"]["rms_db"] > -65.0
            ):
                dereverb_applied = _run_soundseparator_dereverb(
                    Path(result["dialog_wav"]), ss_info, progress_cb=progress_cb, log_cb=log_cb
                )
            result["dereverb_applied"] = dereverb_applied
            _emit_progress(progress_cb, 100, "SoundSeparator DME 3分離が完了しました")
            return result
        except Exception as exc:
            _emit_log(log_cb, f"SoundSeparator 呼び出しに失敗したため DSP フォールバックへ切替えます: {exc}")

    return _separate_dme_dsp_fallback(src, dst_dir, progress_cb=progress_cb, log_cb=log_cb)

# coding: utf-8
"""無音動画向け HunyuanVideo-Foley 連携＆モーション同期プレビュー合成エンジン。

本番規定:
- `engine/hunyuan_foley/` または `models/hunyuanvideo-foley/` にローカルモデルが
  導入済みの場合、VRAM に応じて XXL / XL および `--enable_offload` を選択し、
  長尺動画はチャンク推論＋1秒オーバーラップ・クロスフェードで 48kHz WAV を生成する。
- モデル未取得時は「HunyuanVideo-Foleyモデルが未取得です。モデル管理画面から取得してください」
  と明確な RuntimeError を返す（暗黙ダウンロード禁止）。
- テストやオフライン動作確認で `allow_preview_fallback=True` が明示指定された場合のみ、
  映像の動きエネルギー・カット点・明度から 48kHz 環境音＋モーション同期効果音を生成する
  プレビュー合成器（`method: "motion_synth_preview"`）を動作させる。
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from app.config import ENGINE_DIR, MODELS_DIR, SAMPLE_RATE, find_hunyuan_foley_dir

MISSING_FOLEY_MESSAGE = "HunyuanVideo-Foleyモデルが未取得です。モデル管理画面から取得してください"

_JA_TO_EN_HINTS = {
    "雨": "rain falling",
    "雷": "distant thunder",
    "風": "wind blowing",
    "波": "ocean waves",
    "海": "ocean ambience and waves",
    "川": "flowing river water",
    "森": "forest ambience, rustling leaves, distant birds",
    "街": "city street ambience, distant traffic",
    "室内": "quiet indoor room tone",
    "部屋": "indoor room ambience",
    "足音": "footsteps",
    "歩く": "walking footsteps",
    "走る": "running footsteps and cloth movement",
    "ドア": "door opening and closing",
    "車": "car engine and tire rolling",
    "電車": "train rumbling on rails",
    "炎": "crackling fire",
    "火": "crackling flames",
    "鳥": "birds chirping",
    "拍手": "applause",
    "爆発": "cinematic explosion rumble",
    "水": "water splashing",
    "ガラス": "glass clinking",
    "金属": "metallic clank",
    "キーボード": "keyboard typing clicks",
    "夜": "quiet nighttime crickets and gentle breeze",
}


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
            log_cb("foley", message)
        except Exception:
            pass


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(audio.astype(np.float32), int(dst_sr) // g, int(src_sr) // g, axis=0).astype(np.float32)


def build_foley_prompt(scene_summary: dict[str, Any] | None = None, user_prompt: str = "") -> str:
    """ユーザー入力テキストと映像特徴サマリーから HunyuanVideo-Foley 用英語プロンプトを構築する。"""
    parts: list[str] = []
    raw_user = (user_prompt or "").strip()
    if raw_user:
        # 日本語キーワードが含まれる場合は英語ヒントも併記
        mapped = [en for ja, en in _JA_TO_EN_HINTS.items() if ja in raw_user]
        if raw_user.isascii():
            parts.append(raw_user)
        elif mapped:
            parts.append(", ".join(dict.fromkeys(mapped)))
        else:
            parts.append(raw_user)

    if isinstance(scene_summary, dict):
        tags = scene_summary.get("suggested_foley_tags") or []
        for tag in tags:
            if tag and tag not in parts:
                parts.append(str(tag))

    if not parts:
        parts.append("natural realistic cinematic foley sound effects and subtle room ambience")
    return ", ".join(parts)


def _has_weights(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for pattern in ("*.safetensors", "*.pth", "*.pt", "*.ckpt"):
        if any(p.is_file() and p.stat().st_size > 0 for p in folder.glob(pattern)):
            return True
        if any(p.is_file() and p.stat().st_size > 0 for p in folder.rglob(pattern)):
            return True
    return False


def _detect_official_foley_variants(model_dirs: list[Path]) -> dict[str, str | None]:
    """公式 HunyuanVideo-Foley 重み構成 (synchformer + vae + xl/xxl) または汎用フォールバックを検出する。"""
    variants: dict[str, str | None] = {"XXL": None, "XL": None}
    for mdir in model_dirs:
        if not mdir.is_dir():
            continue
        has_shared = (mdir / "synchformer_state_dict.pth").is_file() and (mdir / "vae_128d_48k.pth").is_file()
        if has_shared:
            if variants["XXL"] is None and (mdir / "hunyuanvideo_foley.pth").is_file():
                variants["XXL"] = str(mdir)
            if variants["XL"] is None and (mdir / "hunyuanvideo_foley_xl.pth").is_file():
                variants["XL"] = str(mdir)

    # テスト・カスタム配置互換: xxl / xl サブフォルダ配下に *.pth 等が置かれている場合も拾う
    for mdir in model_dirs:
        if variants["XXL"] is None and mdir.name.lower() == "xxl" and _has_weights(mdir):
            variants["XXL"] = str(mdir)
        if variants["XL"] is None and mdir.name.lower() == "xl" and _has_weights(mdir):
            variants["XL"] = str(mdir)
    return variants


def find_hunyuan_foley(
    models_root: Path | None = None,
    check_external: bool | None = None,
) -> dict[str, Any]:
    """ローカルに導入された HunyuanVideo-Foley コード・専用venv・HFキャッシュ・重みの有無を検出する。"""
    use_external = (models_root is None) if check_external is None else bool(check_external)
    active_models_root = models_root or MODELS_DIR

    root_candidates: list[Path] = []
    if use_external:
        ext = find_hunyuan_foley_dir()
        if ext is not None:
            root_candidates.append(ext)
        root_candidates.extend(
            [
                ENGINE_DIR / "hunyuan_foley",
                ENGINE_DIR / "HunyuanVideo-Foley",
            ]
        )
    root_candidates.append(active_models_root / "hunyuanvideo-foley")

    seen_roots: set[Path] = set()
    ordered_roots: list[Path] = []
    for rc in root_candidates:
        try:
            resolved = rc.resolve()
        except OSError:
            resolved = rc
        if resolved not in seen_roots and rc.is_dir():
            seen_roots.add(resolved)
            ordered_roots.append(rc)

    root_dir: Path | None = None
    code_dir: Path | None = None
    infer_script: Path | None = None
    for rc in ordered_roots:
        for cdir in (rc / "app", rc):
            if not cdir.is_dir():
                continue
            for script_name in ("infer.py", "inference.py", "generate.py", "predict.py"):
                cand = cdir / script_name
                if cand.is_file():
                    root_dir = rc
                    code_dir = cdir
                    infer_script = cand
                    break
            if infer_script:
                break
        if infer_script:
            break

    model_dirs: list[Path] = []
    if root_dir is not None:
        model_dirs.extend(
            [
                root_dir / "models" / "HunyuanVideo-Foley",
                root_dir / "models",
            ]
        )
    model_dirs.extend(
        [
            active_models_root / "hunyuanvideo-foley" / "HunyuanVideo-Foley",
            active_models_root / "hunyuanvideo-foley" / "xxl",
            active_models_root / "hunyuanvideo-foley" / "xl",
            active_models_root / "hunyuanvideo-foley",
        ]
    )

    variants = _detect_official_foley_variants(model_dirs)
    # 従来の単一ディレクトリ配置フォールバック
    if variants["XXL"] is None and variants["XL"] is None:
        fallback_dir = active_models_root / "hunyuanvideo-foley"
        if _has_weights(fallback_dir):
            variants["XXL"] = str(fallback_dir)

    python_exe = sys.executable
    hf_home: str | None = None
    if root_dir is not None:
        for py_cand in (
            root_dir / ".venv" / "Scripts" / "python.exe",
            root_dir / ".venv" / "bin" / "python",
        ):
            if py_cand.is_file():
                python_exe = str(py_cand)
                break
        hf_cand = root_dir / "hf-cache"
        if hf_cand.is_dir():
            hf_home = str(hf_cand)

    return {
        "available": bool(infer_script and (variants["XXL"] or variants["XL"])),
        "weights_found": bool(variants["XXL"] or variants["XL"]),
        "root": str(root_dir) if root_dir else None,
        "code_dir": str(code_dir) if code_dir else None,
        "infer_script": str(infer_script) if infer_script else None,
        "python_exe": python_exe,
        "hf_home": hf_home,
        "variants": variants,
    }


def _resolve_torch_device() -> tuple[str, str]:
    """config.get_torch_device() または torch バックエンドから (device, label) を返す。"""
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


def _detect_vram_gb() -> float:
    """利用可能メモリ容量 (CUDA VRAM または Apple Silicon ユニファイドメモリ GB) を返す。"""
    device, _ = _resolve_torch_device()
    try:
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                _, total = torch.cuda.mem_get_info(0)
                return round(total / (1024 ** 3), 2)
        elif device == "mps":
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0 and proc.stdout.strip().isdigit():
                return round(int(proc.stdout.strip()) / (1024 ** 3), 2)
            try:
                import psutil

                return round(psutil.virtual_memory().total / (1024 ** 3), 2)
            except Exception:
                return 16.0
    except Exception:
        pass
    return 0.0


def _run_hunyuan_foley_inference(
    video_path: Path,
    out_wav: Path,
    prompt: str,
    info: dict[str, Any],
    duration_sec: float,
    quality: str = "standard",
    chunk_duration_sec: float = 15.0,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """公式 HunyuanVideo-Foley をローカル推論（チャンク＋1秒クロスフェード対応）で実行する。"""
    device, device_label = _resolve_torch_device()
    vram_gb = _detect_vram_gb()
    variants = info["variants"]

    if device == "mps":
        # Apple Silicon (MPS) 環境ではフォールバック許可＆ユニファイドメモリに応じた安全な軽量選択
        if variants.get("XL") and (quality == "fast" or vram_gb < 24.0):
            variant = "XL"
            model_dir = variants["XL"]
        elif variants.get("XXL"):
            variant = "XXL"
            model_dir = variants["XXL"]
        else:
            variant = "XL"
            model_dir = variants["XL"]
        enable_offload = vram_gb < 32.0
        _emit_log(
            log_cb,
            f"{device_label} (ユニファイドメモリ約 {vram_gb:.1f}GB) 向けに PYTORCH_ENABLE_MPS_FALLBACK=1・"
            f"offload={enable_offload} で Foley ({variant}) を実行します。",
        )
    elif device == "cpu":
        variant = "XL" if variants.get("XL") else "XXL"
        model_dir = variants[variant]
        enable_offload = True
        _emit_log(
            log_cb,
            f"GPU/MPS 非検出のため CPU モード (offload={enable_offload}) で Foley ({variant}) を実行します（長尺時は時間がかかります）。",
        )
    else:
        if quality != "fast" and variants.get("XXL") and (vram_gb >= 14.0 or not variants.get("XL")):
            variant = "XXL"
            model_dir = variants["XXL"]
            enable_offload = vram_gb < 20.0
        elif variants.get("XL"):
            variant = "XL"
            model_dir = variants["XL"]
            enable_offload = vram_gb < 14.0
        else:
            variant = "XXL"
            model_dir = variants["XXL"]
            enable_offload = True

    _emit_log(
        log_cb,
        f"HunyuanVideo-Foley ({variant}, device={device}, offload={enable_offload}, MEM={vram_gb:.1f}GB) で音声生成を開始します。prompt='{prompt}'",
    )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません。")

    overlap_sec = 1.0
    chunk_len = max(5.0, float(chunk_duration_sec))
    if duration_sec <= chunk_len + 0.5:
        chunk_specs = [(0.0, duration_sec)]
    else:
        step = max(2.0, chunk_len - overlap_sec)
        chunk_specs = []
        t = 0.0
        while t < duration_sec - 0.25:
            end = min(duration_sec, t + chunk_len)
            chunk_specs.append((t, end))
            if end >= duration_sec:
                break
            t += step

    total_samples = max(1, int(round(duration_sec * SAMPLE_RATE)))
    acc = np.zeros((total_samples, 2), dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="ciness_foley_") as tmp_str:
        tmp_dir = Path(tmp_str)
        for ci, (cs, ce) in enumerate(chunk_specs):
            _emit_progress(
                progress_cb,
                int(10 + 80 * ci / max(len(chunk_specs), 1)),
                f"HunyuanVideo-Foley 推論中 ({ci + 1}/{len(chunk_specs)})...",
            )
            if len(chunk_specs) == 1:
                chunk_video = video_path
            else:
                chunk_video = tmp_dir / f"chunk_{ci:03d}.mp4"
                subprocess.run(
                    [
                        ffmpeg, "-y", "-v", "error",
                        "-ss", f"{cs:.3f}", "-t", f"{ce - cs:.3f}",
                        "-i", str(video_path),
                        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                        str(chunk_video),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=180,
                )

            chunk_out_dir = tmp_dir / f"out_{ci:03d}"
            chunk_out_dir.mkdir(parents=True, exist_ok=True)

            python_exe = str(info.get("python_exe") or sys.executable)
            steps = "30" if quality == "fast" else "50"
            cmd = [
                python_exe,
                "-X", "utf8",
                str(info["infer_script"]),
                "--model_path", str(model_dir),
                "--model_size", variant.lower(),
                "--single_video", str(chunk_video),
                "--single_prompt", prompt,
                "--output_dir", str(chunk_out_dir),
                "--num_inference_steps", steps,
                "--guidance_scale", "4.5",
            ]
            cfg_yaml = Path(str(info["code_dir"])) / "configs" / f"hunyuanvideo-foley-{variant.lower()}.yaml"
            if cfg_yaml.is_file():
                cmd += ["--config_path", str(cfg_yaml)]
            if enable_offload:
                cmd.append("--enable_offload")

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            if device == "mps":
                env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            if info.get("hf_home"):
                env["HF_HOME"] = str(info["hf_home"])
                env["HF_HUB_OFFLINE"] = "1"
            proc = subprocess.run(
                cmd,
                cwd=info["code_dir"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"HunyuanVideo-Foley 推論に失敗しました: {proc.stderr.strip()[-400:]}")

            wavs = sorted(chunk_out_dir.rglob("*.wav"))
            if not wavs:
                raise RuntimeError("HunyuanVideo-Foley から WAV 出力が見つかりませんでした。")

            seg, sr = sf.read(str(wavs[0]), dtype="float32", always_2d=True)
            if sr != SAMPLE_RATE:
                seg = _resample(seg, sr, SAMPLE_RATE)
            if seg.shape[1] == 1:
                seg = np.repeat(seg, 2, axis=1)
            elif seg.shape[1] > 2:
                seg = seg[:, :2]

            start_idx = int(round(cs * SAMPLE_RATE))
            seg_len = min(len(seg), total_samples - start_idx)
            if seg_len <= 0:
                continue
            seg = seg[:seg_len]

            w = np.ones(seg_len, dtype=np.float64)
            fade = int(round(overlap_sec * SAMPLE_RATE))
            if ci > 0:
                ramp = min(fade, seg_len)
                w[:ramp] = np.linspace(0.0, 1.0, ramp)
            if ci < len(chunk_specs) - 1:
                ramp = min(fade, seg_len)
                w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1.0, 0.0, ramp))

            acc[start_idx:start_idx + seg_len] += seg.astype(np.float64) * w[:, None]
            weight[start_idx:start_idx + seg_len] += w

    out_audio = (acc / np.maximum(weight[:, None], 1e-8)).astype(np.float32)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), out_audio, SAMPLE_RATE, subtype="FLOAT")
    _emit_progress(progress_cb, 100, "HunyuanVideo-Foley 音声生成が完了しました")

    return {
        "audio_wav": str(out_wav.resolve()),
        "sfx_mono_wav": None,
        "ambience_wav": None,
        "method": "hunyuan_video_foley",
        "variant": variant,
        "offload": enable_offload,
        "prompt": prompt,
        "chunks": len(chunk_specs),
        "is_preview_fallback": False,
        "sample_rate": SAMPLE_RATE,
    }


def _synthesize_motion_preview_foley(
    out_wav: Path,
    scene_analysis: dict[str, Any] | None,
    duration_sec: float,
    prompt: str,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """テスト・オフライン確認用 48kHz モーション同期プレビューFoley合成器。

    映像の明度・色調から自然な環境音ベッドを生成し、
    動きエネルギー（`motion_track`）およびカット点（`cut_times`）に同期した
    風切り・移動音・ソフトインパクト効果音を 48kHz で合成する。
    """
    _emit_log(
        log_cb,
        "HunyuanVideo-Foley 未取得のため、プレビュー合成器（motion_synth_preview）で48kHzモーション同期Foleyを生成します。",
    )
    _emit_progress(progress_cb, 20, "プレビューFoley環境音・モーション同期SFXを合成中...")

    n_samples = max(SAMPLE_RATE // 2, int(round(duration_sec * SAMPLE_RATE)))
    rng = np.random.default_rng(seed=20261124)

    summary = (scene_analysis or {}).get("scene_summary") or {}
    motion_track = (scene_analysis or {}).get("motion_track") or []
    brightness = float(summary.get("mean_brightness", 128.0))
    cut_times = summary.get("cut_times") or []

    # 1) 環境音ベッド（左右無相関なソフトフィルタノイズ：明度に応じて帯域を変化）
    white_l = rng.standard_normal(n_samples).astype(np.float64)
    white_r = rng.standard_normal(n_samples).astype(np.float64)
    cutoff_hz = 450.0 + (max(0.0, min(255.0, brightness)) / 255.0) * 1400.0
    sos_amb = butter(3, [65.0, cutoff_hz], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    amb_l = sosfilt(sos_amb, white_l)
    amb_r = sosfilt(sos_amb, white_r)
    # 緩やかな空気のうねり
    t_axis = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    lfo_l = 0.85 + 0.15 * np.sin(2.0 * math.pi * 0.22 * t_axis)
    lfo_r = 0.85 + 0.15 * np.cos(2.0 * math.pi * 0.19 * t_axis + 0.7)
    amb = np.stack([amb_l * lfo_l, amb_r * lfo_r], axis=1)
    amb_rms = float(np.sqrt(np.mean(amb ** 2))) + 1e-9
    amb = (amb / amb_rms * 0.045).astype(np.float32)  # 約 -27 dBFS の穏やかな空間ベッド

    _emit_progress(progress_cb, 55, "映像の動き重心・カット点に同期した効果音を合成中...")

    # 2) モーション同期 点音源SFX（モノラルで出力し、後段の空間音響エンジンで3D軌跡へ配置）
    if motion_track:
        mt_t = np.array([float(m["t"]) for m in motion_track], dtype=np.float64)
        mt_e = np.array([float(m.get("motion_energy", 0.0)) for m in motion_track], dtype=np.float64)
        env_motion = np.interp(t_axis, mt_t, mt_e, left=mt_e[0], right=mt_e[-1])
    else:
        env_motion = 0.12 + 0.08 * np.sin(2.0 * math.pi * 0.5 * t_axis) ** 2

    # 動きエネルギーの正規化（穏やかな動きでもはっきり聴こえるレンジへ）
    e_peak = float(np.max(env_motion))
    if e_peak > 1e-5:
        env_norm = np.clip(env_motion / max(e_peak, 0.08), 0.0, 1.0)
    else:
        env_norm = np.full(n_samples, 0.15, dtype=np.float64)

    # 動き速度変化に応じたウーッシュ／衣擦れ風帯域ノイズ＋低域モーション感
    sfx_noise = rng.standard_normal(n_samples).astype(np.float64)
    sos_whoosh = butter(3, [220.0, 2600.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_body = butter(2, [80.0, 380.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    whoosh = sosfilt(sos_whoosh, sfx_noise)
    body = sosfilt(sos_body, sfx_noise)
    whoosh /= float(np.sqrt(np.mean(whoosh ** 2))) + 1e-9
    body /= float(np.sqrt(np.mean(body ** 2))) + 1e-9

    sfx_mono = (whoosh * (0.14 * env_norm ** 1.2) + body * (0.10 * env_norm)).astype(np.float64)

    # 動き急増点およびカット点に柔らかなパーカッシブ・トランジション音を重ねる
    event_times = list(cut_times)
    if len(motion_track) >= 3:
        energies = [float(m.get("motion_energy", 0.0)) for m in motion_track]
        mean_e = float(np.mean(energies))
        for i in range(1, len(energies) - 1):
            if energies[i] > max(mean_e * 1.35, 0.05) and energies[i] >= energies[i - 1] and energies[i] >= energies[i + 1]:
                event_times.append(float(motion_track[i]["t"]))

    for ev_t in event_times[:24]:
        center = int(round(ev_t * SAMPLE_RATE))
        ev_len = int(round(0.22 * SAMPLE_RATE))
        start = max(0, center - int(0.02 * SAMPLE_RATE))
        end = min(n_samples, start + ev_len)
        if end - start < 64:
            continue
        lt = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE
        decay = np.exp(-lt * 18.0) * (1.0 - np.exp(-lt * 220.0))
        sweep_freq = 190.0 * np.exp(-lt * 6.0) + 65.0
        phase = 2.0 * math.pi * np.cumsum(sweep_freq) / SAMPLE_RATE
        hit = 0.22 * decay * np.sin(phase)
        sfx_mono[start:end] += hit

    # 開始・終了の軽いフェードでクリック防止
    fade_n = min(int(0.05 * SAMPLE_RATE), n_samples // 4)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)
        sfx_mono[:fade_n] *= ramp
        sfx_mono[-fade_n:] *= ramp[::-1]
        amb[:fade_n] *= ramp[:, None].astype(np.float32)
        amb[-fade_n:] *= ramp[::-1, None].astype(np.float32)

    sfx_mono_f32 = np.clip(sfx_mono, -0.95, 0.95).astype(np.float32)
    combined = np.clip(amb + sfx_mono_f32[:, None] * 0.85, -0.95, 0.95).astype(np.float32)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sfx_path = out_wav.with_name("foley_sfx_mono.wav")
    amb_path = out_wav.with_name("foley_ambience.wav")
    sf.write(str(out_wav), combined, SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(sfx_path), sfx_mono_f32, SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(amb_path), amb, SAMPLE_RATE, subtype="FLOAT")

    _emit_progress(progress_cb, 100, "プレビューFoley合成が完了しました")
    return {
        "audio_wav": str(out_wav.resolve()),
        "sfx_mono_wav": str(sfx_path.resolve()),
        "ambience_wav": str(amb_path.resolve()),
        "method": "motion_synth_preview",
        "variant": "preview_synth",
        "offload": False,
        "prompt": prompt,
        "chunks": 1,
        "is_preview_fallback": True,
        "sample_rate": SAMPLE_RATE,
    }


def generate_foley(
    video_path: str | Path,
    out_wav: str | Path,
    scene_analysis: dict[str, Any] | None = None,
    user_prompt: str = "",
    quality: str = "standard",
    allow_preview_fallback: bool = False,
    chunk_duration_sec: float = 15.0,
    progress_cb: Callable[..., Any] | None = None,
    log_cb: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """無音動画から 48kHz Foley 音声を生成する（未取得時は本番規定どおり明示エラー）。"""
    src = Path(video_path)
    dst = Path(out_wav)
    if not src.is_file():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {src}")

    summary = (scene_analysis or {}).get("scene_summary") if isinstance(scene_analysis, dict) else None
    duration_sec = float((scene_analysis or {}).get("duration_sec", 0.0) or 0.0)
    if duration_sec <= 0.0:
        from app.vision.analyzer import inspect_video

        duration_sec = float(inspect_video(src).get("duration_sec", 5.0) or 5.0)

    prompt = build_foley_prompt(summary, user_prompt=user_prompt)
    info = find_hunyuan_foley()

    if info["available"]:
        return _run_hunyuan_foley_inference(
            video_path=src,
            out_wav=dst,
            prompt=prompt,
            info=info,
            duration_sec=duration_sec,
            quality=quality,
            chunk_duration_sec=chunk_duration_sec,
            progress_cb=progress_cb,
            log_cb=log_cb,
        )

    if not allow_preview_fallback:
        raise RuntimeError(MISSING_FOLEY_MESSAGE)

    return _synthesize_motion_preview_foley(
        out_wav=dst,
        scene_analysis=scene_analysis,
        duration_sec=duration_sec,
        prompt=prompt,
        progress_cb=progress_cb,
        log_cb=log_cb,
    )

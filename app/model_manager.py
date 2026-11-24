# coding: utf-8
"""CineSS AIモデルカタログ管理と明示的ダウンロード。

設計方針（SoundSeparator 継承）:
- 推論時の暗黙ダウンロードは一切行わない。
- SoundSeparator の共有モデルが既存ならコピーせずそのまま参照（shared_soundsep）する。
- VRAM目安は提供元推奨・暫定目安・未確定を区別し、根拠のない誇張をしない。
"""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from app.config import MODELS_DIR, find_soundseparator_dir

ProgressCallback = Callable[[int, str], None]

_DIAR_REV = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
_DIAR_URL = f"https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/{_DIAR_REV}/"
_REUSE_REV = "31f82f2efad1095034492f236e8dac9239ba713b"
_REUSE_URL = f"https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/resolve/{_REUSE_REV}/"
_DBS_REV = "22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b"
_DBS_URL = f"https://huggingface.co/JusperLee/DBS-TIGER-DnR/resolve/{_DBS_REV}/"

CATEGORY_LABELS = {
    "separation": "音声3分離 (DME)",
    "dereverb": "残響除去",
    "diarization": "話者分離 (ダイアリゼーション)",
    "enhancement": "音声強調",
    "hrtf": "バイノーラル HRTF",
    "foley": "無音動画 Foley 生成",
    "speaker_overlap": "重なり発話の分離強化",
    "vision_depth": "映像深度推定",
}


def _file_spec(
    rel_path: str,
    url: str,
    *,
    fallback_urls: tuple[str, ...] = (),
    sha256: str | None = None,
    size: int | None = None,
) -> dict:
    return {
        "rel_path": rel_path,
        "url": url,
        "fallback_urls": list(fallback_urls),
        "sha256": sha256,
        "size": size,
    }


def _catalog_specs() -> list[dict]:
    """モデルカタログの定義一覧を返す。"""
    return [
        # ---- SoundSeparator 共有可能モデル ----
        {
            "id": "bandit",
            "label": "BandIt Plus",
            "category": "separation",
            "description": "映像の話し声・効果音・BGMをステレオで3分離する主力モデル。",
            "vram_summary": "推奨VRAM: 6 GB以上（暫定目安・最低要件は未検証）",
            "download_mb": 149,
            "source_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training",
            "downloadable": True,
            "soundsep_shareable": True,
            "files": [
                _file_spec(
                    "model_bandit_plus_dnr_sdr_11.47.chpt",
                    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt",
                )
            ],
        },
        {
            "id": "dbs_dnr",
            "label": "DBS-TIGER-DnR",
            "category": "separation",
            "description": "軽量・高速な映像3分離モデル（出力はモノラル、省VRAM環境向け）。",
            "vram_summary": "推奨VRAM: 4 GB以上（暫定目安・短尺合成音実測ベース）",
            "download_mb": 5,
            "source_url": "https://huggingface.co/JusperLee/DBS-TIGER-DnR",
            "downloadable": True,
            "soundsep_shareable": True,
            "files": [
                _file_spec(
                    "dbs_tiger_dnr.safetensors",
                    _DBS_URL + "model.safetensors",
                    size=4111836,
                    sha256="3d8205743f2a6f5a6704f5d4104d7cf5bf7bf02dab2fd728da31f4c69225cadb",
                ),
                _file_spec(
                    "dbs_tiger_dnr_config.json",
                    _DBS_URL + "config.json",
                    size=957,
                    sha256="c7f7893323e78f9b99747a13090558a13c4f4b77383a38acb9979709fd04d1d8",
                ),
            ],
        },
        {
            "id": "dereverb",
            "label": "MelBand RoFormer Dereverb",
            "category": "dereverb",
            "description": "話し声の元リバーブを抑えてドライ化し、3D再配置時の定位を鮮明にします。",
            "vram_summary": "推奨VRAM: 8 GB以上（暫定目安・最低要件は未検証）",
            "download_mb": 913,
            "source_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer",
            "downloadable": True,
            "soundsep_shareable": True,
            "files": [
                _file_spec(
                    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
                    "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
                ),
                _file_spec(
                    "dereverb_mel_band_roformer_anvuew.yaml",
                    "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml",
                ),
            ],
        },
        {
            "id": "nemotron_diarization",
            "label": "Nemotron-3-Diarization",
            "category": "diarization",
            "description": "最大8話者の発話区間（重なり発話含む）を推定するNVIDIA製話者分離モデル。",
            "vram_summary": "推奨VRAM: 未確定（RTX 4090 24GBで実音声98秒の動作を確認）",
            "download_mb": 397,
            "source_url": "https://huggingface.co/nvidia/Nemotron-3-Diarization",
            "downloadable": True,
            "soundsep_shareable": True,
            "files": [
                _file_spec("nemotron-3-diarization/config.json", _DIAR_URL + "config.json"),
                _file_spec("nemotron-3-diarization/processor_config.json", _DIAR_URL + "processor_config.json"),
                _file_spec(
                    "nemotron-3-diarization/model.safetensors",
                    _DIAR_URL + "model.safetensors",
                    size=396954592,
                    sha256="c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6",
                ),
            ],
        },
        {
            "id": "nemotron_reuse",
            "label": "NemotronLabs RE-USE",
            "category": "enhancement",
            "description": "モノラル音声のノイズ・残響を抑える音声強調モデル。",
            "vram_summary": "推奨VRAM: 未確定（RTX 4090 24GBで短尺合成音の動作を確認）",
            "download_mb": 39,
            "source_url": "https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE",
            "downloadable": True,
            "soundsep_shareable": True,
            "files": [
                _file_spec("nemotron-reuse/config.json", _REUSE_URL + "config.json", size=830),
                _file_spec(
                    "nemotron-reuse/model.safetensors",
                    _REUSE_URL + "model.safetensors",
                    size=38592940,
                    sha256="06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1",
                ),
            ],
        },
        # ---- CineSS 追加モデル ----
        {
            "id": "mit_kemar_sofa",
            "label": "MIT KEMAR SOFA HRTF",
            "category": "hrtf",
            "description": "MIT Media Lab実測ダミーヘッドHRTF（SOFA形式）。未取得時は内蔵物理HRTFで動作します。",
            "vram_summary": "GPU不要（CPU高速畳み込み・軽量18 MB）",
            "download_mb": 18,
            "source_url": "https://www.sofaconventions.org/mediawiki/index.php/Files",
            "downloadable": True,
            "soundsep_shareable": False,
            "files": [
                _file_spec(
                    "hrtf/mit_kemar_normal_pinna.sofa",
                    "https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa",
                    fallback_urls=(
                        "https://github.com/sofacoustics/API_MO/raw/master/HRTFs/MIT/mit_kemar_normal_pinna.sofa",
                    ),
                )
            ],
        },
        {
            "id": "hunyuan_foley_xxl",
            "label": "HunyuanVideo-Foley XXL",
            "category": "foley",
            "description": "無音動画から映像同期した48kHz高精細Foley効果音・環境音を生成する主力モデル。",
            "vram_summary": "推奨VRAM: 16 GB以上（--enable_offload併用で12 GB前後も可・条件により変動）",
            "download_mb": 16500,
            "source_url": "https://huggingface.co/tencent/HunyuanVideo-Foley",
            "downloadable": True,
            "soundsep_shareable": False,
            "local_dir": "hunyuanvideo-foley/xxl",
            "alt_local_dirs": ["hunyuanvideo-foley"],
            "marker_globs": ["*.safetensors", "*.pth", "*.pt", "*.ckpt"],
            "hf_repo": "tencent/HunyuanVideo-Foley",
        },
        {
            "id": "hunyuan_foley_xl",
            "label": "HunyuanVideo-Foley XL",
            "category": "foley",
            "description": "中VRAM環境向けの軽量版HunyuanVideo-Foley。オフロード併用で短尺動画に対応。",
            "vram_summary": "推奨VRAM: 12 GB以上（--enable_offload併用で8 GB前後も可・短尺推奨）",
            "download_mb": 9500,
            "source_url": "https://huggingface.co/tencent/HunyuanVideo-Foley",
            "downloadable": True,
            "soundsep_shareable": False,
            "local_dir": "hunyuanvideo-foley/xl",
            "marker_globs": ["*.safetensors", "*.pth", "*.pt", "*.ckpt"],
            "hf_repo": "tencent/HunyuanVideo-Foley",
        },
        {
            "id": "sepformer_whamr",
            "label": "SpeechBrain SepFormer",
            "category": "speaker_overlap",
            "description": "複数話者の発話が同時に重なる区間を音響的に2話者分離する強化モデル。",
            "vram_summary": "推奨VRAM: 4 GB以上（暫定目安・CPU実行も可）",
            "download_mb": 110,
            "source_url": "https://huggingface.co/speechbrain/sepformer-whamr",
            "downloadable": True,
            "soundsep_shareable": False,
            "local_dir": "sepformer/sepformer-whamr",
            "hf_repo": "speechbrain/sepformer-whamr",
            "files": [
                _file_spec(
                    "sepformer/sepformer-whamr/hyperparams.yaml",
                    "https://huggingface.co/speechbrain/sepformer-whamr/resolve/main/hyperparams.yaml",
                ),
                _file_spec(
                    "sepformer/sepformer-whamr/encoder.ckpt",
                    "https://huggingface.co/speechbrain/sepformer-whamr/resolve/main/encoder.ckpt",
                ),
                _file_spec(
                    "sepformer/sepformer-whamr/masknet.ckpt",
                    "https://huggingface.co/speechbrain/sepformer-whamr/resolve/main/masknet.ckpt",
                ),
                _file_spec(
                    "sepformer/sepformer-whamr/decoder.ckpt",
                    "https://huggingface.co/speechbrain/sepformer-whamr/resolve/main/decoder.ckpt",
                ),
            ],
        },
        {
            "id": "depth_anything_v2_small",
            "label": "Depth Anything V2 Small",
            "category": "vision_depth",
            "description": "映像の単眼深度マップを推定し、人物・音源の奥行き距離(m)精度を向上させます。",
            "vram_summary": "推奨VRAM: 2 GB以上（暫定目安・未取得時は顔サイズ比から距離推定）",
            "download_mb": 99,
            "source_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
            "downloadable": True,
            "soundsep_shareable": False,
            "local_dir": "depth-anything-v2-small",
            "hf_repo": "depth-anything/Depth-Anything-V2-Small-hf",
            "files": [
                _file_spec(
                    "depth-anything-v2-small/config.json",
                    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/config.json",
                ),
                _file_spec(
                    "depth-anything-v2-small/preprocessor_config.json",
                    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/preprocessor_config.json",
                ),
                _file_spec(
                    "depth-anything-v2-small/model.safetensors",
                    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/model.safetensors",
                ),
            ],
        },
    ]


def _file_present(base_dir: Path, spec: dict) -> bool:
    target = base_dir / spec["rel_path"]
    if not target.is_file():
        return False
    try:
        st_size = target.stat().st_size
    except OSError:
        return False
    if spec.get("size"):
        return st_size == spec["size"]
    return st_size > 0


def _dir_has_weights(directory: Path, globs: list[str]) -> bool:
    if not directory.is_dir():
        return False
    for pattern in globs:
        for matched in directory.glob(pattern):
            try:
                if matched.is_file() and matched.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def _resolve_model_status(
    spec: dict,
    models_root: Path | None = None,
    soundsep_dir: Path | None = None,
) -> tuple[str, Path | None]:
    """モデルの配置状態 (shared_soundsep | installed | missing) と実パスを返す。"""
    local_root = models_root or MODELS_DIR
    files = spec.get("files") or []

    # 1. CineSS 自身の models/ を確認
    if files:
        if all(_file_present(local_root, f) for f in files):
            first = local_root / files[0]["rel_path"]
            resolved = first.parent if "/" in files[0]["rel_path"] and len(files) > 1 else first
            return "installed", resolved
    elif spec.get("local_dir"):
        globs = spec.get("marker_globs") or ["*"]
        candidates = [local_root / spec["local_dir"]]
        for alt in spec.get("alt_local_dirs") or []:
            candidates.append(local_root / alt)
        for cand in candidates:
            if _dir_has_weights(cand, globs):
                return "installed", cand

    # 2. SoundSeparator 共有可能モデルであれば SoundSeparator/models/ を確認
    if spec.get("soundsep_shareable"):
        ss_dir = soundsep_dir if soundsep_dir is not None else find_soundseparator_dir()
        if ss_dir is not None:
            ss_models = ss_dir / "models"
            if files and all(_file_present(ss_models, f) for f in files):
                first = ss_models / files[0]["rel_path"]
                resolved = first.parent if "/" in files[0]["rel_path"] and len(files) > 1 else first
                return "shared_soundsep", resolved

    return "missing", None


def list_models(
    models_root: Path | None = None,
    soundsep_dir: Path | None = None,
) -> list[dict]:
    """利用モデルカタログと現在の導入状態一覧を返す（ネットワークアクセスなし）。"""
    ss_dir = soundsep_dir if soundsep_dir is not None else find_soundseparator_dir()
    results: list[dict] = []
    for spec in _catalog_specs():
        status, found_path = _resolve_model_status(spec, models_root=models_root, soundsep_dir=ss_dir)
        results.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "category": spec["category"],
                "category_label": CATEGORY_LABELS.get(spec["category"], spec["category"]),
                "description": spec["description"],
                "vram_summary": spec["vram_summary"],
                "download_mb": spec["download_mb"],
                "status": status,
                "path": str(found_path) if found_path is not None else None,
                "source_url": spec["source_url"],
                "downloadable": bool(spec.get("downloadable", True)),
            }
        )
    return results


def get_model_path(model_id: str) -> Path | None:
    """指定モデルが利用可能であればローカルまたは共有パスを返す。未取得なら None。"""
    for entry in list_models():
        if entry["id"] == model_id and entry["path"]:
            return Path(entry["path"])
    return None


def _emit(progress_cb: ProgressCallback | None, pct: int, message: str) -> None:
    if progress_cb is not None:
        progress_cb(max(0, min(100, int(pct))), message)


def _fetch_file(
    target: Path,
    urls: list[str],
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    chunk_cb: Callable[[int, int], None] | None = None,
) -> None:
    if target.is_file():
        st_size = target.stat().st_size
        if (expected_size and st_size == expected_size) or (not expected_size and st_size > 0):
            return
        raise RuntimeError(f"既存ファイルのサイズが不正です: {target.name}。破損の可能性があるため確認してください。")

    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for url in urls:
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.partial")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CineSS/1.0"})
            digest = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                header_len = resp.headers.get("Content-Length")
                total = expected_size or (int(header_len) if header_len and header_len.isdigit() else 0)
                downloaded = 0
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    downloaded += len(block)
                    if chunk_cb is not None:
                        chunk_cb(downloaded, total)

            if downloaded == 0 or (expected_size and downloaded != expected_size):
                raise RuntimeError(f"{target.name} のダウンロードが途中で終了しました（取得 {downloaded} bytes）。")
            if expected_sha256 and digest.hexdigest().lower() != expected_sha256.lower():
                raise RuntimeError(f"{target.name} の SHA-256 検証に失敗しました。")
            os.replace(tmp, target)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
        finally:
            tmp.unlink(missing_ok=True)

    raise RuntimeError(f"{target.name} の取得に失敗しました: {last_error}")


def download_model(model_id: str, progress_cb: ProgressCallback | None = None) -> dict:
    """指定モデルを明示的に取得する。推論処理から暗黙呼び出ししてはならない。"""
    spec = next((s for s in _catalog_specs() if s["id"] == model_id), None)
    if spec is None:
        raise ValueError(f"不明なモデルIDです: {model_id}")
    if not spec.get("downloadable", True):
        raise ValueError(f"モデル '{spec['label']}' は自動ダウンロード非対応です。")

    status, existing = _resolve_model_status(spec)
    if status in ("installed", "shared_soundsep") and existing is not None:
        label = "SoundSeparator共有モデルとして利用可能です" if status == "shared_soundsep" else "既に取得済みです"
        _emit(progress_cb, 100, label)
        return {
            "id": model_id,
            "label": spec["label"],
            "status": status,
            "path": str(existing),
            "message": label,
        }

    files = spec.get("files") or []
    if files:
        total_files = len(files)
        for idx, fspec in enumerate(files):
            target = MODELS_DIR / fspec["rel_path"]
            urls = [fspec["url"], *fspec.get("fallback_urls", [])]

            def _on_chunk(count: int, total: int, _idx: int = idx, _name: str = target.name) -> None:
                frac = min(count / total, 1.0) if total else 0.0
                pct = int(((_idx + frac) / total_files) * 100)
                mb = count / (1024 * 1024)
                if total:
                    total_mb = total / (1024 * 1024)
                    _emit(progress_cb, pct, f"{_name} を取得中 ({mb:.1f} / {total_mb:.1f} MB)")
                else:
                    _emit(progress_cb, pct, f"{_name} を取得中 ({mb:.1f} MB)")

            _emit(progress_cb, int((idx / total_files) * 100), f"{target.name} のダウンロードを開始します...")
            _fetch_file(
                target,
                urls,
                expected_size=fspec.get("size"),
                expected_sha256=fspec.get("sha256"),
                chunk_cb=_on_chunk,
            )
            _emit(progress_cb, int(((idx + 1) / total_files) * 100), f"{target.name} の配置と検証が完了しました")

        new_status, new_path = _resolve_model_status(spec)
        return {
            "id": model_id,
            "label": spec["label"],
            "status": new_status,
            "path": str(new_path) if new_path else None,
            "message": f"{spec['label']} の取得が完了しました。",
        }

    # HuggingFace リポジトリ単位の大型モデル (HunyuanVideo-Foley 等)
    hf_repo = spec.get("hf_repo")
    local_dir = spec.get("local_dir")
    if hf_repo and local_dir:
        dest_dir = MODELS_DIR / local_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        _emit(progress_cb, 5, f"HuggingFace ({hf_repo}) から取得を準備しています...")
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{spec['label']} の自動取得には huggingface_hub が必要です。\n"
                "既存のCUDA版PyTorchを壊さないよう以下を実行してから再試行してください:\n"
                "  uv pip install --python .venv huggingface-hub\n"
                f"または {spec['source_url']} から手動取得し、{dest_dir} に配置してください。"
            ) from exc

        try:
            _emit(progress_cb, 15, f"{spec['label']} をダウンロード中（大容量のため時間がかかります）...")
            snapshot_download(
                repo_id=hf_repo,
                local_dir=str(dest_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"HuggingFace からの {spec['label']} 取得に失敗しました: {exc}\n"
                f"ネットワーク接続・ディスク空き容量（目安 {spec['download_mb'] / 1024:.1f} GB以上）・"
                f"または HuggingFace トークン設定を確認してください。"
            ) from exc

        _emit(progress_cb, 100, f"{spec['label']} の取得が完了しました")
        new_status, new_path = _resolve_model_status(spec)
        return {
            "id": model_id,
            "label": spec["label"],
            "status": new_status,
            "path": str(new_path) if new_path else str(dest_dir),
            "message": f"{spec['label']} の取得が完了しました。",
        }

    raise RuntimeError(f"モデル '{spec['label']}' の取得先が定義されていません。")

# coding: utf-8
"""CineSS — 映像サラウンドMixMaster & バイノーラルスタジオ FastAPI サーバー。

主な役割:
- PC動作環境診断 (`/api/system/check`, `/api/system/recommend`)
- AIモデル管理・明示的ダウンロード (`/api/models`, `/api/models/{id}/download`)
- 動画アップロード＆自動検査 (`/api/videos/upload`)
- ジョブ管理＆AIエージェント連携ブリッジ (`/api/jobs`, `/api/agent/*`)
- 3D空間配置の再MixMaster (`/api/jobs/{id}/remix`)
- バイノーラル映像・音声・サラウンドマスター・個別ステムの Range 対応配信

重要方針:
- 重い推論・変換・同期I/Oはすべて `def`（同期ルート＝スレッドプール実行）または
  バックグラウンドスレッドで処理し、FastAPI のイベントループを一切ブロックしない。
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import (  # noqa: E402
    AGENT_STATE_PATH,
    DEFAULT_PORT,
    JOBS_DIR,
    OUTPUT_DIR,
    SHUTDOWN_FLAG_PATH,
)

STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = OUTPUT_DIR / "uploads"

# 起動時に作業ディレクトリを用意
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# エージェント生存判定の有効秒数（ハートビート切れ閾値）
AGENT_TTL_SECONDS = 30.0
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
MAX_LOG_ENTRIES = 500

# スレッドセーフなロック群
_state_lock = threading.RLock()
_agent_lock = threading.RLock()
_download_lock = threading.RLock()
_download_jobs: dict[str, dict[str, Any]] = {}


class JobCancelledError(RuntimeError):
    """ユーザーによるキャンセル要求を示す内部例外。"""


# ---------------------------------------------------------------------------
# Pydantic リクエストモデル
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    duration_sec: float = Field(..., ge=0.0)
    has_audio: bool = True
    mode: str = "auto"
    quality_preset: str = "auto"
    spatial_format: str = "atmos"
    ambisonics_order: Optional[int] = Field(default=None, ge=1, le=7)


class CreateJobRequest(BaseModel):
    upload_id: Optional[str] = None
    video_path: Optional[str] = None
    mode: str = "auto"
    spatial_format: str = "atmos"
    ambisonics_order: Optional[int] = Field(default=None, ge=1, le=7)
    quality_preset: str = "auto"
    target_lufs: Optional[float] = None
    loudness_mode: Optional[str] = None  # "match_original" | "fixed"
    dialog_pan_mode: str = "cinema"      # "cinema" | "immersive"
    enable_dialog_ducking: bool = True
    foley_prompt: str = ""
    allow_foley_fallback: bool = False
    preview_start: float = 0.0
    preview_duration: Optional[float] = None
    enable_dereverb: bool = True
    execution_mode: str = "auto"  # "auto" | "agent" | "local"


class RemixRequest(BaseModel):
    scene: Optional[dict[str, Any]] = None
    updated_scene: Optional[dict[str, Any]] = None
    spatial_format: Optional[str] = None
    ambisonics_order: Optional[int] = Field(default=None, ge=1, le=7)
    target_lufs: Optional[float] = None
    loudness_mode: Optional[str] = None
    dialog_pan_mode: Optional[str] = None
    enable_dialog_ducking: Optional[bool] = None


class AgentHeartbeatRequest(BaseModel):
    agent_name: str = "ciness-agent"
    pid: Optional[int] = None
    status: str = "idle"  # "idle" | "working" | "waiting" | "offline"
    active_job_id: Optional[str] = None


class AgentClaimRequest(BaseModel):
    agent_name: str = "ciness-agent"
    pid: Optional[int] = None
    message: Optional[str] = None


class AgentProgressRequest(BaseModel):
    progress: Optional[float] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    log_text: Optional[str] = None
    actor: str = "agent"


class AgentCompleteRequest(BaseModel):
    status: str = "done"  # "done" | "error" | "cancelled"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    actor: str = "agent"


class ShutdownRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# 共通ファイルI/O・ユーティリティ
# ---------------------------------------------------------------------------
def _now_ts() -> float:
    return time.time()


def _now_time_str() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Any, max_retries: int = 10) -> None:
    """一時ファイル経由で原子的に JSON を保存する（Windows ファイルロック競合の短時間リトライ付き）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        last_err: Optional[OSError] = None
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, path)
                return
            except (PermissionError, OSError) as exc:
                last_err = exc
                if attempt + 1 < max_retries:
                    time.sleep(0.015 * (attempt + 1))
        if last_err is not None:
            raise last_err
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: Any = None, max_retries: int = 4) -> Any:
    if not path.is_file():
        return default
    for attempt in range(max_retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt + 1 < max_retries:
                time.sleep(0.01 * (attempt + 1))
                continue
            return default
        except Exception:
            return default
    return default


def _sanitize_filename(filename: Optional[str]) -> str:
    name = Path(filename or "video.mp4").name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name:
        name = "video.mp4"
    if "." not in name:
        name += ".mp4"
    return name


def _validate_id(raw_id: str, label: str = "ID") -> str:
    if not raw_id or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", raw_id) or ".." in raw_id:
        raise HTTPException(status_code=400, detail=f"不正な{label}です")
    return raw_id


def _normalize_mode(mode: str) -> str:
    m = (mode or "auto").strip().lower()
    if m in ("voiced_remix", "voiced", "remix"):
        return "voiced"
    if m in ("silent_foley", "silent", "foley"):
        return "silent_foley"
    return "auto"


def _normalize_spatial_format(
    fmt: str, ambisonics_order: Optional[int] = None
) -> tuple[str, Optional[int]]:
    """サラウンド規格と Ambisonics 次数 (1〜7) を統一正規化する。"""
    f = (fmt or "atmos").strip().lower()
    if f in ("atmos", "dolby_atmos", "7.1.4"):
        return "atmos", None

    # フォーマット文字列からデフォルト次数を決定
    inferred_order: Optional[int] = None
    if f in ("ambisonics_foa", "foa", "ambisonics_1", "ambisonics1"):
        inferred_order = 1
    elif f in ("ambisonics_hoa", "hoa", "ambisonics_3", "ambisonics3", "ambisonics"):
        inferred_order = 3
    elif f in ("ambisonics_hoa5", "hoa5", "ambisonics_5", "ambisonics5"):
        inferred_order = 5
    elif f in ("ambisonics_hoa7", "hoa7", "ambisonics_7", "ambisonics7"):
        inferred_order = 7
    else:
        m = re.fullmatch(r"(?:ambisonics_?hoa|ambisonics_?|hoa)(\d)", f)
        if m:
            inferred_order = int(m.group(1))

    if inferred_order is None:
        return "atmos", None

    order = int(ambisonics_order) if ambisonics_order is not None else inferred_order
    order = max(1, min(7, order))

    if order == 1:
        return "ambisonics_foa", 1
    if order == 5:
        return "ambisonics_hoa5", 5
    if order == 7:
        return "ambisonics_hoa7", 7
    return "ambisonics_hoa", order


def _normalize_quality(preset: str) -> str:
    q = (preset or "auto").strip().lower()
    if q in ("auto", "high", "standard", "fast"):
        return q
    return "auto"


def _normalize_dialog_pan_mode(mode: Optional[str]) -> str:
    """セリフ定位モード（cinema / immersive）を正規化する。"""
    m = (mode or "cinema").strip().lower()
    return "immersive" if m == "immersive" else "cinema"


def _resolve_loudness_request(
    loudness_mode: Optional[str],
    target_lufs: Optional[float],
    fallback_mode: Optional[str] = None,
    fallback_lufs: Optional[float] = None,
) -> tuple[str, Optional[float]]:
    """loudness_mode と target_lufs を後方互換を保って解決する。"""
    if loudness_mode is not None:
        lm = str(loudness_mode).strip().lower()
        if lm in ("match_original", "match", "auto"):
            return "match_original", None
        if target_lufs is not None:
            return "fixed", float(target_lufs)
        if fallback_lufs is not None:
            return "fixed", float(fallback_lufs)
        return "fixed", -18.0

    # loudness_mode 未指定かつ target_lufs に数値がある既存呼び出しは "fixed" とする
    if target_lufs is not None:
        return "fixed", float(target_lufs)

    # 両方未指定時はフォールバック（新規ジョブでは match_original、再Mix時は元設定を維持）
    if fallback_mode is not None:
        fm = str(fallback_mode).strip().lower()
        if fm in ("match_original", "match", "auto"):
            return "match_original", None
        if fallback_lufs is not None:
            return "fixed", float(fallback_lufs)
        return "fixed", -18.0

    if fallback_lufs is not None:
        return "fixed", float(fallback_lufs)
    return "match_original", None


def _job_dir(job_id: str) -> Path:
    safe_id = _validate_id(job_id, "ジョブID")
    return JOBS_DIR / safe_id


def _job_state_path(job_id: str) -> Path:
    return _job_dir(job_id) / "state.json"


def _read_job_state(job_id: str) -> Optional[dict[str, Any]]:
    with _state_lock:
        return _read_json(_job_state_path(job_id), default=None)


def _write_job_state(job_id: str, state: dict[str, Any]) -> dict[str, Any]:
    with _state_lock:
        state["updated_at"] = _now_ts()
        _atomic_write_json(_job_state_path(job_id), state)
        return state


def _append_log(
    state: dict[str, Any],
    message: str,
    actor: str = "system",
    stage: Optional[str] = None,
) -> None:
    text = (message or "").strip()
    if not text:
        return
    logs = state.setdefault("logs", [])
    logs.append(
        {
            "ts": _now_ts(),
            "time_str": _now_time_str(),
            "actor": actor or "system",
            "stage": stage or state.get("stage") or "",
            "message": text,
        }
    )
    if len(logs) > MAX_LOG_ENTRIES:
        del logs[:-MAX_LOG_ENTRIES]


# ---------------------------------------------------------------------------
# エージェント接続状態管理
# ---------------------------------------------------------------------------
def _get_agent_info() -> dict[str, Any]:
    """AGENT_STATE_PATH から最新のエージェント接続情報を取得する。"""
    with _agent_lock:
        raw = _read_json(AGENT_STATE_PATH, default={}) or {}
    last_seen = raw.get("last_seen")
    status_raw = str(raw.get("status") or "offline")
    now = _now_ts()
    connected = False
    if isinstance(last_seen, (int, float)) and (now - float(last_seen)) <= AGENT_TTL_SECONDS:
        if status_raw not in ("offline", "stopped", "disconnected"):
            connected = True

    active_job_id = raw.get("active_job_id") if connected else None
    # state.json 側に agent_running なジョブがあれば補完する
    if connected and not active_job_id:
        for st in _list_all_job_states():
            if st.get("status") == "agent_running":
                active_job_id = st.get("job_id")
                break

    if not connected:
        effective_status = "offline"
    elif active_job_id or status_raw in ("working", "running", "busy"):
        effective_status = "working"
    else:
        effective_status = "idle"

    return {
        "agent_connected": connected,
        "agent_last_seen": last_seen,
        "agent_last_seen_iso": raw.get("last_seen_iso"),
        "agent_name": raw.get("agent_name") if connected else raw.get("agent_name"),
        "agent_pid": raw.get("pid") if connected else None,
        "agent_status": effective_status,
        "active_job_id": active_job_id,
    }


def _update_agent_state(
    agent_name: Optional[str] = None,
    pid: Optional[int] = None,
    status: Optional[str] = None,
    active_job_id: Any = ...,
) -> dict[str, Any]:
    """AGENT_STATE_PATH を原子的に更新する。"""
    with _agent_lock:
        cur = _read_json(AGENT_STATE_PATH, default={}) or {}
        if agent_name is not None:
            cur["agent_name"] = agent_name
        elif not cur.get("agent_name"):
            cur["agent_name"] = "ciness-agent"
        if pid is not None:
            cur["pid"] = pid
        if status is not None:
            cur["status"] = status
        elif not cur.get("status"):
            cur["status"] = "idle"
        if active_job_id is not ...:
            cur["active_job_id"] = active_job_id
        cur["last_seen"] = _now_ts()
        cur["last_seen_iso"] = _now_iso()
        _atomic_write_json(AGENT_STATE_PATH, cur)
    return _get_agent_info()


def _list_all_job_states() -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    if not JOBS_DIR.is_dir():
        return states
    with _state_lock:
        for child in JOBS_DIR.iterdir():
            if not child.is_dir():
                continue
            st = _read_json(child / "state.json", default=None)
            if isinstance(st, dict) and st.get("job_id"):
                states.append(st)
    states.sort(key=lambda s: float(s.get("created_at") or 0.0), reverse=True)
    return states


# ---------------------------------------------------------------------------
# 成果物ファイル解決・ステム列挙
# ---------------------------------------------------------------------------
def _extract_result_path(result: Optional[dict[str, Any]], keys: tuple[str, ...]) -> Optional[Path]:
    if not isinstance(result, dict):
        return None
    containers = [result]
    for sub in ("files", "artifacts", "outputs", "spatial", "media"):
        val = result.get(sub)
        if isinstance(val, dict):
            containers.append(val)
    for container in containers:
        for key in keys:
            raw = container.get(key)
            if isinstance(raw, str) and raw.strip():
                p = Path(raw)
                if p.is_file():
                    return p
    return None


def _resolve_media_path(job_id: str, state: dict[str, Any], kind: str) -> Optional[Path]:
    """kind に応じた成果物ファイルパスを柔軟かつ安全に解決する。"""
    jdir = _job_dir(job_id)
    result = state.get("result") or _read_json(jdir / "result.json", default={}) or {}
    req = state.get("request") or {}

    def _first_existing(candidates: list[Path]) -> Optional[Path]:
        for c in candidates:
            if c and c.is_file():
                return c
        return None

    if kind == "original_video":
        found = _extract_result_path(
            result, ("preview_video", "input_preview_video", "original_video", "source_video")
        )
        if found:
            return found
        fallback_list = [jdir / "preview_input.mp4", jdir / "input_preview.mp4"]
        if req.get("video_path"):
            fallback_list.append(Path(req["video_path"]))
        return _first_existing(fallback_list)

    if kind == "binaural_video":
        found = _extract_result_path(
            result, ("binaural_video", "binaural_mp4", "output_video", "video_path")
        )
        if found:
            return found
        return _first_existing(
            [jdir / "binaural_output.mp4", jdir / "binaural.mp4", jdir / "output.mp4"]
        )

    if kind == "original_audio":
        found = _extract_result_path(
            result, ("original_audio", "original_preview_wav", "original_wav", "extracted_audio")
        )
        if found:
            return found
        return _first_existing(
            [
                jdir / "original_preview.wav",
                jdir / "original.wav",
                jdir / "extracted_audio.wav",
                jdir / "audio" / "original.wav",
            ]
        )

    if kind == "binaural_audio":
        found = _extract_result_path(
            result, ("binaural_audio", "binaural_wav", "binaural_path")
        )
        if found:
            return found
        return _first_existing(
            [
                jdir / "binaural.wav",
                jdir / "binaural_output.wav",
                jdir / "spatial" / "binaural.wav",
            ]
        )

    if kind == "master_spatial":
        found = _extract_result_path(
            result, ("master_spatial", "master_spatial_wav", "adm_bwf_wav", "ambisonics_wav")
        )
        if found:
            return found
        candidates = [
            jdir / "master_spatial.wav",
            jdir / "spatial" / "master_spatial.wav",
            jdir / "spatial" / "master_atmos_adm.wav",
            jdir / "spatial" / "master_ambisonics_hoa.wav",
            jdir / "spatial" / "master_ambisonics_foa.wav",
            jdir / "master_atmos_adm.wav",
        ]
        found = _first_existing(candidates)
        if found:
            return found
        for pattern in ("*adm*.wav", "*ambisonics*.wav", "*ambix*.wav", "*master*.wav"):
            for folder in (jdir / "spatial", jdir):
                if folder.is_dir():
                    matches = sorted(folder.glob(pattern))
                    if matches:
                        return matches[0]
        return None

    if kind == "surround_714":
        found = _extract_result_path(result, ("surround_714", "surround_714_wav"))
        if found:
            return found
        return _first_existing(
            [
                jdir / "surround_714.wav",
                jdir / "spatial" / "surround_714.wav",
                jdir / "spatial" / "atmos_714.wav",
            ]
        )

    if kind == "stems_zip":
        found = _extract_result_path(result, ("stems_zip", "stems_archive"))
        if found:
            return found
        return _first_existing([jdir / "stems.zip"])

    return None


def _collect_stems(job_id: str, scene: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """ジョブ内の個別ステムWAVを一覧化する。"""
    jdir = _job_dir(job_id)
    found: dict[str, dict[str, Any]] = {}

    if isinstance(scene, dict):
        for obj in scene.get("objects") or []:
            raw_path = obj.get("audio_path")
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.is_absolute():
                p = jdir / p
            if p.is_file():
                found[p.name] = {
                    "name": p.name,
                    "object_id": obj.get("id") or p.stem,
                    "label": obj.get("label") or p.stem,
                    "role": obj.get("role") or "stem",
                    "url": f"/api/jobs/{job_id}/stems/{p.name}",
                    "size_bytes": p.stat().st_size,
                }

    for sub in ("stems", "speakers", "separated", "audio"):
        folder = jdir / sub
        if not folder.is_dir():
            continue
        for wav in sorted(folder.glob("*.wav")):
            if wav.name not in found:
                found[wav.name] = {
                    "name": wav.name,
                    "object_id": wav.stem,
                    "label": wav.stem,
                    "role": "stem",
                    "url": f"/api/jobs/{job_id}/stems/{wav.name}",
                    "size_bytes": wav.stat().st_size,
                }
    return list(found.values())


def _enrich_job_payload(job_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """GET /api/jobs/{job_id} 用に scene.json / result.json / media URL を結合する。"""
    jdir = _job_dir(job_id)
    out = dict(state)
    out["job_dir"] = str(jdir.resolve())
    if not out.get("result"):
        disk_result = _read_json(jdir / "result.json", default=None)
        if disk_result is not None:
            out["result"] = disk_result

    scene = _read_json(jdir / "scene.json", default=None)
    if scene is not None:
        out["scene"] = scene

    remix_ver = int(out.get("remix_version") or 0)
    qs = f"?v={remix_ver}" if remix_ver else ""
    media_kinds = (
        "original_video",
        "binaural_video",
        "original_audio",
        "binaural_audio",
        "master_spatial",
        "surround_714",
        "stems_zip",
    )
    available_media: dict[str, bool] = {}
    media_urls: dict[str, str] = {}
    for kind in media_kinds:
        p = _resolve_media_path(job_id, out, kind)
        exists = bool(p and p.is_file())
        available_media[kind] = exists
        if exists:
            media_urls[kind] = f"/api/jobs/{job_id}/media/{kind}{qs}"
    out["available_media"] = available_media
    out["media_urls"] = media_urls
    out["stems"] = _collect_stems(job_id, scene)
    return out


# ---------------------------------------------------------------------------
# ローカルパイプライン実行ランナー（スレッド実行）
# ---------------------------------------------------------------------------
def _make_callbacks(job_id: str, default_actor: str = "local"):
    """pipeline.run_job / remix_job 向けの柔軟な進捗・ログコールバックを生成する。"""

    def progress_cb(*args: Any, **kwargs: Any) -> None:
        pct: Optional[float] = kwargs.get("progress")
        stage: Optional[str] = kwargs.get("stage")
        msg: Optional[str] = kwargs.get("message")

        if len(args) == 1:
            if isinstance(args[0], dict):
                pct = args[0].get("progress", pct)
                stage = args[0].get("stage", stage)
                msg = args[0].get("message", msg)
            elif isinstance(args[0], (int, float)):
                pct = float(args[0])
            elif isinstance(args[0], str):
                msg = args[0]
        elif len(args) == 2:
            if isinstance(args[0], (int, float)):
                pct = float(args[0])
                msg = str(args[1]) if args[1] is not None else msg
            else:
                stage = str(args[0])
                msg = str(args[1])
        elif len(args) >= 3:
            if isinstance(args[0], (int, float)):
                pct = float(args[0])
            stage = str(args[1]) if args[1] is not None else stage
            msg = str(args[2]) if args[2] is not None else msg

        with _state_lock:
            st = _read_job_state(job_id)
            if not st:
                return
            if st.get("cancel_requested"):
                raise JobCancelledError("ユーザーによりキャンセルされました")
            if pct is not None:
                st["progress"] = round(max(0.0, min(100.0, float(pct))), 1)
            if stage:
                st["stage"] = stage
            if msg:
                st["message"] = msg
                _append_log(st, msg, actor=default_actor, stage=st.get("stage"))
            _write_job_state(job_id, st)

    def log_cb(*args: Any, **kwargs: Any) -> None:
        actor = kwargs.get("actor") or default_actor
        stage = kwargs.get("stage")
        msg = kwargs.get("message") or kwargs.get("log_text")
        if msg is None:
            if len(args) == 1:
                msg = str(args[0])
            elif len(args) >= 2:
                stage = stage or str(args[0])
                msg = str(args[1])
        if not msg:
            return
        with _state_lock:
            st = _read_job_state(job_id)
            if not st:
                return
            _append_log(st, str(msg), actor=str(actor), stage=stage or st.get("stage"))
            _write_job_state(job_id, st)

    return progress_cb, log_cb


def _run_local_job(job_id: str, job_dir: Path, request_payload: dict[str, Any]) -> None:
    """サーバー内蔵スレッドで app.pipeline.run_job を実行する。"""
    t0 = _now_ts()
    progress_cb, log_cb = _make_callbacks(job_id, default_actor="local")
    try:
        with _state_lock:
            st = _read_job_state(job_id)
            if not st:
                return
            st["status"] = "local_running"
            st["actor"] = "local"
            st["started_at"] = st.get("started_at") or t0
            st["stage"] = "inspect"
            st["message"] = "ローカルパイプラインを起動しています..."
            _append_log(st, "サーバー内蔵ランナーで処理を開始しました。", actor="local", stage="inspect")
            _write_job_state(job_id, st)

        from app.pipeline import run_job

        result = run_job(job_dir, request_payload, progress_cb=progress_cb, log_cb=log_cb)
        with _state_lock:
            st = _read_job_state(job_id) or {"job_id": job_id}
            st["status"] = "done"
            st["stage"] = "done"
            st["progress"] = 100.0
            st["message"] = "サラウンドMixMaster＆バイノーラル出力が完了しました"
            st["result"] = result
            st["error"] = None
            st["finished_at"] = _now_ts()
            st["elapsed_sec"] = round(st["finished_at"] - float(st.get("started_at") or t0), 2)
            _append_log(
                st,
                f"全工程が完了しました（所要時間 {st['elapsed_sec']} 秒）。",
                actor="local",
                stage="done",
            )
            _write_job_state(job_id, st)
    except JobCancelledError as exc:
        with _state_lock:
            st = _read_job_state(job_id) or {"job_id": job_id}
            st["status"] = "cancelled"
            st["stage"] = "cancelled"
            st["message"] = str(exc)
            st["finished_at"] = _now_ts()
            st["elapsed_sec"] = round(st["finished_at"] - float(st.get("started_at") or t0), 2)
            _append_log(st, str(exc), actor="local", stage="cancelled")
            _write_job_state(job_id, st)
    except Exception as exc:
        err_msg = str(exc) or exc.__class__.__name__
        tb = traceback.format_exc()
        with _state_lock:
            st = _read_job_state(job_id) or {"job_id": job_id}
            st["status"] = "error"
            st["stage"] = "error"
            st["message"] = f"エラーが発生しました: {err_msg}"
            st["error"] = err_msg
            st["error_detail"] = tb[-2000:]
            st["finished_at"] = _now_ts()
            st["elapsed_sec"] = round(st["finished_at"] - float(st.get("started_at") or t0), 2)
            _append_log(st, f"処理エラー: {err_msg}", actor="local", stage="error")
            _write_job_state(job_id, st)


def _download_model_worker(model_id: str) -> None:
    """バックグラウンドでモデルをダウンロードする。"""

    def progress_cb(*args: Any, **kwargs: Any) -> None:
        pct = kwargs.get("progress")
        msg = kwargs.get("message")
        if len(args) == 1:
            if isinstance(args[0], dict):
                pct = args[0].get("progress", pct)
                msg = args[0].get("message", msg)
            elif isinstance(args[0], (int, float)):
                pct = float(args[0])
            elif isinstance(args[0], str):
                msg = args[0]
        elif len(args) >= 2:
            if isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                # (downloaded_bytes, total_bytes, [msg]) 形式にも対応
                total = float(args[1])
                pct = (float(args[0]) / total * 100.0) if total > 0 else pct
                if len(args) >= 3:
                    msg = str(args[2])
            elif isinstance(args[0], (int, float)):
                pct = float(args[0])
                msg = str(args[1])
        with _download_lock:
            entry = _download_jobs.setdefault(model_id, {})
            if pct is not None:
                entry["progress"] = round(max(0.0, min(100.0, float(pct))), 1)
            if msg:
                entry["message"] = str(msg)
            entry["updated_at"] = _now_ts()

    try:
        from app.model_manager import download_model

        res = download_model(model_id, progress_cb=progress_cb)
        with _download_lock:
            _download_jobs[model_id] = {
                "model_id": model_id,
                "status": "done",
                "progress": 100.0,
                "message": "ダウンロード完了",
                "result": res,
                "error": None,
                "updated_at": _now_ts(),
                "finished_at": _now_ts(),
            }
    except Exception as exc:
        err_msg = str(exc) or exc.__class__.__name__
        with _download_lock:
            _download_jobs[model_id] = {
                "model_id": model_id,
                "status": "error",
                "progress": 0.0,
                "message": f"ダウンロード失敗: {err_msg}",
                "error": err_msg,
                "updated_at": _now_ts(),
                "finished_at": _now_ts(),
            }


# ---------------------------------------------------------------------------
# FastAPI アプリケーション定義
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        SHUTDOWN_FLAG_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    yield


app = FastAPI(
    title="CineSS — 映像サラウンドMixMaster & バイノーラルスタジオ",
    version="1.0.0",
    lifespan=_lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index_page() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="index.html が見つかりません")
    return FileResponse(str(index_file), headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    agent = _get_agent_info()
    return {
        "status": "ok",
        "app": "CineSS",
        "port": DEFAULT_PORT,
        "agent_connected": agent["agent_connected"],
        "agent_status": agent["agent_status"],
        "agent_name": agent["agent_name"],
        "active_job_id": agent["active_job_id"],
        "server_time": _now_ts(),
    }


@app.post("/api/system/shutdown")
def api_system_shutdown(req: Optional[ShutdownRequest] = Body(default=None)):
    """Web UI からの停止操作を受け、待機エージェントとサーバー自身を安全に終了する。"""
    payload = req or ShutdownRequest()
    running = [
        st
        for st in _list_all_job_states()
        if st.get("status") in ("agent_running", "local_running", "remixing")
    ]
    if running and not payload.force:
        return JSONResponse(
            status_code=409,
            content={
                "detail": f"処理中のジョブが {len(running)} 件あります。強制停止しますか？",
                "running_jobs": [str(r.get("job_id")) for r in running],
            },
        )

    _atomic_write_json(
        SHUTDOWN_FLAG_PATH,
        {"requested_at": _now_ts(), "requested_at_iso": _now_iso(), "force": bool(payload.force)},
    )
    if AGENT_STATE_PATH.is_file():
        _update_agent_state(status="offline", active_job_id=None)

    if os.environ.get("CINESS_DISABLE_EXIT") != "1":
        def _delayed_exit() -> None:
            time.sleep(0.5)
            os._exit(0)

        threading.Thread(target=_delayed_exit, name="ciness-shutdown", daemon=True).start()

    return {
        "status": "shutting_down",
        "message": "CineSS サーバーと待機エージェントを停止しています...",
    }


# ---------------------------------------------------------------------------
# 1. ハードウェア診断＆モデル管理
# ---------------------------------------------------------------------------
@app.get("/api/system/check")
def api_system_check() -> dict[str, Any]:
    """実PCのCPU/GPU/RAM/FFmpeg/ディスク診断＋エージェント接続状態を返す。"""
    from app.hardware_advisor import check_system

    report = dict(check_system())
    agent = _get_agent_info()
    report["agent_connected"] = agent["agent_connected"]
    report["agent_last_seen"] = agent["agent_last_seen"]
    report["agent_name"] = agent["agent_name"]
    report["agent_status"] = agent["agent_status"]
    report["active_job_id"] = agent["active_job_id"]
    report["agent"] = agent
    return report


@app.post("/api/system/recommend")
def api_system_recommend(req: RecommendRequest) -> dict[str, Any]:
    """動画尺・有音/無音・品質プリセット・サラウンド規格に応じた実行プラン診断を返す。"""
    from app.hardware_advisor import evaluate_video_plan

    spatial_format, ambi_order = _normalize_spatial_format(
        req.spatial_format, req.ambisonics_order
    )
    kwargs: dict[str, Any] = {
        "duration_sec": req.duration_sec,
        "has_audio": req.has_audio,
        "mode": _normalize_mode(req.mode),
        "quality_preset": _normalize_quality(req.quality_preset),
        "spatial_format": spatial_format,
    }
    if ambi_order is not None and "ambisonics_order" in inspect.signature(evaluate_video_plan).parameters:
        kwargs["ambisonics_order"] = ambi_order
    return evaluate_video_plan(**kwargs)


@app.get("/api/models")
def api_models_list() -> dict[str, Any]:
    """利用モデル一覧と現在のダウンロード進捗を返す。"""
    from app.model_manager import list_models

    models = list_models()
    with _download_lock:
        downloads = {k: dict(v) for k, v in _download_jobs.items()}

    enriched = []
    for m in models:
        item = dict(m)
        mid = item.get("id")
        if mid in downloads:
            item["download_state"] = downloads[mid]
        enriched.append(item)
    return {"models": enriched, "downloads": downloads}


@app.post("/api/models/{model_id}/download")
def api_model_download(model_id: str) -> dict[str, Any]:
    """指定モデルのバックグラウンドダウンロードを開始する（暗黙DLは行わない）。"""
    safe_mid = _validate_id(model_id, "モデルID")
    from app.model_manager import list_models

    catalog = {m.get("id"): m for m in list_models()}
    if safe_mid not in catalog:
        raise HTTPException(status_code=404, detail=f"未知のモデルIDです: {safe_mid}")

    model_info = catalog[safe_mid]
    if not model_info.get("downloadable", True) and model_info.get("status") == "missing":
        raise HTTPException(
            status_code=400,
            detail="このモデルは手動導入または SoundSeparator 側での取得が必要です",
        )

    with _download_lock:
        existing = _download_jobs.get(safe_mid)
        if existing and existing.get("status") == "downloading":
            return {"ok": True, "model_id": safe_mid, "download": dict(existing)}
        entry = {
            "model_id": safe_mid,
            "status": "downloading",
            "progress": 0.0,
            "message": "ダウンロードを開始しています...",
            "error": None,
            "started_at": _now_ts(),
            "updated_at": _now_ts(),
        }
        _download_jobs[safe_mid] = entry

    th = threading.Thread(
        target=_download_model_worker,
        args=(safe_mid,),
        daemon=True,
        name=f"ciness-dl-{safe_mid}",
    )
    th.start()
    return {"ok": True, "model_id": safe_mid, "download": dict(entry)}


# ---------------------------------------------------------------------------
# 2. 動画アップロード＆検査
# ---------------------------------------------------------------------------
@app.post("/api/videos/upload")
def api_video_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """動画を保存し、映像/音声メタ検査とPCスペック照合プランを返す。"""
    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"未対応の動画形式です ({ext})。MP4 / MOV / MKV / WebM に対応しています。",
        )

    upload_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    upload_dir = UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dst_path = upload_dir / safe_name

    try:
        with dst_path.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f, length=1024 * 1024)
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"動画ファイルの保存に失敗しました: {exc}") from exc
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    size_bytes = dst_path.stat().st_size
    if size_bytes <= 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="アップロードされた動画ファイルが空です")

    try:
        from app.vision.analyzer import inspect_video

        inspection = inspect_video(str(dst_path))
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"動画の解析に失敗しました: {exc}"
        ) from exc

    has_audio = not bool(inspection.get("is_silent", not inspection.get("has_audio_stream", True)))
    duration_sec = float(inspection.get("duration_sec") or 0.0)

    from app.hardware_advisor import evaluate_video_plan

    plan = evaluate_video_plan(
        duration_sec=duration_sec,
        has_audio=has_audio,
        mode="auto",
        quality_preset="auto",
        spatial_format="atmos",
    )

    meta = {
        "upload_id": upload_id,
        "filename": safe_name,
        "video_path": str(dst_path),
        "size_bytes": size_bytes,
        "inspection": inspection,
        "created_at": _now_ts(),
    }
    _atomic_write_json(upload_dir / "meta.json", meta)

    return {
        "upload_id": upload_id,
        "filename": safe_name,
        "size_bytes": size_bytes,
        "preview_url": f"/api/videos/{upload_id}/preview",
        "inspection": inspection,
        "plan": plan,
    }


@app.get("/api/videos/{upload_id}/preview")
def api_video_preview(upload_id: str) -> FileResponse:
    safe_uid = _validate_id(upload_id, "アップロードID")
    meta = _read_json(UPLOADS_DIR / safe_uid / "meta.json", default=None)
    if not meta or not meta.get("video_path"):
        raise HTTPException(status_code=404, detail="アップロード動画が見つかりません")
    vpath = Path(meta["video_path"])
    if not vpath.is_file():
        raise HTTPException(status_code=404, detail="動画ファイルが存在しません")
    mime, _ = mimetypes.guess_type(str(vpath))
    return FileResponse(
        str(vpath),
        media_type=mime or "video/mp4",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# 3. ジョブ管理＆エージェント連携ブリッジ
# ---------------------------------------------------------------------------
def _resolve_upload_video(upload_id: Optional[str], video_path: Optional[str]) -> tuple[Path, str]:
    if upload_id:
        safe_uid = _validate_id(upload_id, "アップロードID")
        upload_dir = UPLOADS_DIR / safe_uid
        meta = _read_json(upload_dir / "meta.json", default=None)
        if meta and meta.get("video_path"):
            p = Path(meta["video_path"])
            if p.is_file():
                return p, str(meta.get("filename") or p.name)
        if upload_dir.is_dir():
            for f in upload_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_EXTS:
                    return f, f.name
        raise HTTPException(status_code=404, detail=f"upload_id '{safe_uid}' の動画が見つかりません")

    if video_path:
        p = Path(video_path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"指定された動画が見つかりません: {video_path}")
        return p, p.name

    raise HTTPException(status_code=400, detail="upload_id または video_path を指定してください")


@app.post("/api/jobs")
def api_create_job(req: CreateJobRequest) -> dict[str, Any]:
    """新規処理ジョブを作成し、エージェント待機キューまたはローカル即時実行を開始する。"""
    src_video, filename = _resolve_upload_video(req.upload_id, req.video_path)

    exec_mode = (req.execution_mode or "auto").strip().lower()
    if exec_mode not in ("auto", "agent", "local"):
        exec_mode = "auto"

    mode = _normalize_mode(req.mode)
    spatial_format, ambi_order = _normalize_spatial_format(
        req.spatial_format, req.ambisonics_order
    )
    quality_preset = _normalize_quality(req.quality_preset)
    loudness_mode, target_lufs = _resolve_loudness_request(
        req.loudness_mode, req.target_lufs
    )
    dialog_pan_mode = _normalize_dialog_pan_mode(req.dialog_pan_mode)
    enable_dialog_ducking = bool(req.enable_dialog_ducking)
    preview_start = max(0.0, float(req.preview_start or 0.0))
    preview_duration = (
        max(1.0, float(req.preview_duration))
        if req.preview_duration is not None and float(req.preview_duration) > 0
        else None
    )

    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    request_payload: dict[str, Any] = {
        "job_id": job_id,
        "upload_id": req.upload_id,
        "filename": filename,
        "video_path": str(src_video.resolve()),
        "mode": mode,
        "spatial_format": spatial_format,
        "ambisonics_order": ambi_order,
        "quality_preset": quality_preset,
        "loudness_mode": loudness_mode,
        "target_lufs": target_lufs,
        "dialog_pan_mode": dialog_pan_mode,
        "enable_dialog_ducking": enable_dialog_ducking,
        "foley_prompt": (req.foley_prompt or "").strip(),
        "allow_foley_fallback": bool(req.allow_foley_fallback),
        "preview_start": preview_start,
        "preview_duration": preview_duration,
        "enable_dereverb": bool(req.enable_dereverb),
        "execution_mode": exec_mode,
    }
    _atomic_write_json(jdir / "request.json", request_payload)

    agent = _get_agent_info()
    use_agent = (exec_mode == "agent") or (exec_mode == "auto" and agent["agent_connected"])

    now = _now_ts()
    initial_state: dict[str, Any] = {
        "job_id": job_id,
        "upload_id": req.upload_id,
        "filename": filename,
        "status": "queued_for_agent" if use_agent else "local_running",
        "execution_mode": exec_mode,
        "actor": None if use_agent else "local",
        "stage": "queued" if use_agent else "inspect",
        "progress": 0.0,
        "message": (
            "エージェントの受注を待っています..."
            if use_agent
            else "ローカル処理を開始しています..."
        ),
        "cancel_requested": False,
        "remix_version": 0,
        "created_at": now,
        "updated_at": now,
        "started_at": None if use_agent else now,
        "finished_at": None,
        "elapsed_sec": None,
        "request": request_payload,
        "logs": [],
        "result": None,
        "error": None,
    }

    if use_agent:
        if agent["agent_connected"]:
            _append_log(
                initial_state,
                f"ジョブを登録しました。接続中エージェント（{agent.get('agent_name') or 'ciness-agent'}）の受注を待機しています。",
                actor="server",
                stage="queued",
            )
        else:
            _append_log(
                initial_state,
                "エージェント実行キューに登録しました（現在未接続です。エージェントを起動するか、ローカル即時実行へ切り替えられます）。",
                actor="server",
                stage="queued",
            )
    else:
        _append_log(
            initial_state,
            "ジョブを登録しました。ローカル即時実行ランナーを起動します。",
            actor="server",
            stage="inspect",
        )

    _write_job_state(job_id, initial_state)

    if not use_agent:
        th = threading.Thread(
            target=_run_local_job,
            args=(job_id, jdir, request_payload),
            daemon=True,
            name=f"ciness-job-{job_id}",
        )
        th.start()

    return _enrich_job_payload(job_id, initial_state)


@app.get("/api/jobs")
def api_list_jobs(limit: int = Query(30, ge=1, le=200)) -> dict[str, Any]:
    """ジョブ一覧を新しい順で返す。"""
    states = _list_all_job_states()[:limit]
    items = []
    for st in states:
        req = st.get("request") or {}
        items.append(
            {
                "job_id": st.get("job_id"),
                "filename": st.get("filename"),
                "status": st.get("status"),
                "stage": st.get("stage"),
                "progress": st.get("progress", 0.0),
                "message": st.get("message", ""),
                "execution_mode": st.get("execution_mode"),
                "actor": st.get("actor"),
                "mode": req.get("mode"),
                "spatial_format": req.get("spatial_format"),
                "quality_preset": req.get("quality_preset"),
                "created_at": st.get("created_at"),
                "updated_at": st.get("updated_at"),
                "elapsed_sec": st.get("elapsed_sec"),
                "error": st.get("error"),
                "has_result": bool(st.get("result")),
            }
        )
    return {"jobs": items, "agent": _get_agent_info()}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> dict[str, Any]:
    """特定ジョブの最新 state.json・scene.json・メディアURL一覧を返す。"""
    st = _read_job_state(job_id)
    if not st:
        raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
    return _enrich_job_payload(job_id, st)


@app.post("/api/jobs/{job_id}/start-local")
def api_job_start_local(job_id: str) -> dict[str, Any]:
    """エージェント待機中のジョブをローカル即時実行へ切り替える。"""
    with _state_lock:
        st = _read_job_state(job_id)
        if not st:
            raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
        if st.get("status") != "queued_for_agent":
            raise HTTPException(
                status_code=409,
                detail=f"現在のステータス ({st.get('status')}) ではローカル切替できません",
            )
        jdir = _job_dir(job_id)
        req_payload = st.get("request") or _read_json(jdir / "request.json", default={})
        st["status"] = "local_running"
        st["actor"] = "local"
        st["stage"] = "inspect"
        st["started_at"] = _now_ts()
        st["message"] = "ローカル即時実行へ切り替えました..."
        _append_log(
            st,
            "ユーザー操作によりエージェント待機からローカル即時実行へ切り替えました。",
            actor="user",
            stage="inspect",
        )
        _write_job_state(job_id, st)

    th = threading.Thread(
        target=_run_local_job,
        args=(job_id, jdir, req_payload),
        daemon=True,
        name=f"ciness-job-{job_id}",
    )
    th.start()
    return _enrich_job_payload(job_id, st)


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str) -> dict[str, Any]:
    """ジョブのキャンセル要求フラグを立てる。待機中なら即座にキャンセル完了。"""
    with _state_lock:
        st = _read_job_state(job_id)
        if not st:
            raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
        st["cancel_requested"] = True
        if st.get("status") == "queued_for_agent":
            st["status"] = "cancelled"
            st["stage"] = "cancelled"
            st["message"] = "待機中のジョブをキャンセルしました"
            st["finished_at"] = _now_ts()
            _append_log(st, "ユーザーにより待機ジョブがキャンセルされました。", actor="user", stage="cancelled")
        elif st.get("status") in ("agent_running", "local_running", "remixing"):
            st["message"] = "キャンセル要求を送信しました（処理の区切りで停止します）..."
            _append_log(st, "キャンセル要求を受け付けました。", actor="user", stage=st.get("stage"))
        _write_job_state(job_id, st)
    return _enrich_job_payload(job_id, st)


@app.post("/api/jobs/{job_id}/remix")
def api_job_remix(job_id: str, req: RemixRequest) -> dict[str, Any]:
    """3D空間配置やフォーマット・ラウドネスを変更し、分離済みステムから即座に再MixMasterする。"""
    jdir = _job_dir(job_id)
    updated_scene = req.scene or req.updated_scene
    if not isinstance(updated_scene, dict) or not updated_scene.get("objects"):
        raise HTTPException(status_code=400, detail="有効な scene データ（objects を含む）を指定してください")

    with _state_lock:
        st = _read_job_state(job_id)
        if not st:
            raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
        if st.get("status") in ("agent_running", "local_running", "remixing"):
            raise HTTPException(
                status_code=409,
                detail="ジョブが現在処理中のため、完了後に再Mixを実行してください",
            )
        orig_req = st.get("request") or {}
        orig_mix = (
            updated_scene.get("mix_settings")
            if isinstance(updated_scene.get("mix_settings"), dict)
            else {}
        )
        if req.spatial_format is not None:
            raw_fmt = req.spatial_format
            raw_order = req.ambisonics_order
        else:
            raw_fmt = orig_req.get("spatial_format") or "atmos"
            raw_order = (
                req.ambisonics_order
                if req.ambisonics_order is not None
                else orig_req.get("ambisonics_order")
            )
        spatial_format, ambi_order = _normalize_spatial_format(raw_fmt, raw_order)

        fallback_lufs = (
            orig_req.get("target_lufs")
            if orig_req.get("target_lufs") is not None
            else orig_mix.get("target_lufs")
        )
        loudness_mode, target_lufs = _resolve_loudness_request(
            req.loudness_mode,
            req.target_lufs,
            fallback_mode=orig_req.get("loudness_mode") or orig_mix.get("loudness_mode"),
            fallback_lufs=fallback_lufs,
        )
        raw_pan_mode = (
            req.dialog_pan_mode
            if req.dialog_pan_mode is not None
            else (orig_req.get("dialog_pan_mode") or orig_mix.get("dialog_pan_mode") or "cinema")
        )
        dialog_pan_mode = _normalize_dialog_pan_mode(raw_pan_mode)
        if req.enable_dialog_ducking is not None:
            enable_dialog_ducking = bool(req.enable_dialog_ducking)
        elif orig_req.get("enable_dialog_ducking") is not None:
            enable_dialog_ducking = bool(orig_req.get("enable_dialog_ducking"))
        elif orig_mix.get("enable_dialog_ducking") is not None:
            enable_dialog_ducking = bool(orig_mix.get("enable_dialog_ducking"))
        else:
            enable_dialog_ducking = True

        lufs_label = (
            "元音声に合わせる (Match Original)"
            if loudness_mode == "match_original" or target_lufs is None
            else f"{target_lufs:.1f} LKFS"
        )

        st["status"] = "remixing"
        st["stage"] = "remix"
        st["progress"] = 10.0
        st["message"] = "新しい3D配置で再MixMaster＆MP4再結合を実行中..."
        st["cancel_requested"] = False
        _append_log(
            st,
            f"3D空間配置の再Mixを開始しました（形式: {spatial_format} / 目標: {lufs_label} / セリフ定位: {dialog_pan_mode} / ダッキング: {'ON' if enable_dialog_ducking else 'OFF'}）。",
            actor="user",
            stage="remix",
        )
        _write_job_state(job_id, st)

    t0 = _now_ts()
    progress_cb, log_cb = _make_callbacks(job_id, default_actor="remix")
    try:
        from app.pipeline import remix_job

        sig_params = inspect.signature(remix_job).parameters
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
        )
        remix_kwargs: dict[str, Any] = {
            "spatial_format": spatial_format,
            "progress_cb": progress_cb,
            "log_cb": log_cb,
        }
        if "loudness_mode" in sig_params or has_var_kw:
            remix_kwargs["loudness_mode"] = loudness_mode
            remix_kwargs["target_lufs"] = target_lufs
        else:
            remix_kwargs["target_lufs"] = (
                float(target_lufs) if target_lufs is not None else -18.0
            )
        if ambi_order is not None and ("ambisonics_order" in sig_params or has_var_kw):
            remix_kwargs["ambisonics_order"] = ambi_order
        if "dialog_pan_mode" in sig_params or has_var_kw:
            remix_kwargs["dialog_pan_mode"] = dialog_pan_mode
        if "enable_dialog_ducking" in sig_params or has_var_kw:
            remix_kwargs["enable_dialog_ducking"] = enable_dialog_ducking

        scene_file = jdir / "scene.json"
        prev_scene_mtime = scene_file.stat().st_mtime_ns if scene_file.is_file() else None
        new_result = remix_job(jdir, updated_scene, **remix_kwargs)
        cur_scene_mtime = scene_file.stat().st_mtime_ns if scene_file.is_file() else None
        # pipeline.remix_job が scene.json を更新済みなら保持し、未生成/未更新時のみ書き込む
        if cur_scene_mtime is None or cur_scene_mtime == prev_scene_mtime:
            _atomic_write_json(scene_file, updated_scene)

        with _state_lock:
            st = _read_job_state(job_id) or {"job_id": job_id}
            st["status"] = "done"
            st["stage"] = "done"
            st["progress"] = 100.0
            st["remix_version"] = int(st.get("remix_version") or 0) + 1
            if isinstance(st.get("request"), dict):
                st["request"]["spatial_format"] = spatial_format
                st["request"]["ambisonics_order"] = ambi_order
                st["request"]["loudness_mode"] = loudness_mode
                if (
                    loudness_mode == "match_original"
                    and req.loudness_mode is None
                    and req.target_lufs is None
                ):
                    # 明示指定なしの再Mix時は元の target_lufs 設定（None 等）を壊さず維持する
                    pass
                else:
                    st["request"]["target_lufs"] = target_lufs
                st["request"]["dialog_pan_mode"] = dialog_pan_mode
                st["request"]["enable_dialog_ducking"] = enable_dialog_ducking
            st["result"] = new_result
            st["error"] = None
            elapsed = round(_now_ts() - t0, 2)
            st["message"] = f"再MixMasterが完了しました（{elapsed}秒）"
            _append_log(st, f"再MixMaster＆動画再結合が完了しました（{elapsed}秒）。", actor="remix", stage="done")
            _write_job_state(job_id, st)

        enriched = _enrich_job_payload(job_id, st)
        return {
            "ok": True,
            "job_id": job_id,
            "status": "done",
            "remix_version": enriched.get("remix_version", 1),
            "elapsed_sec": elapsed,
            "result": enriched.get("result"),
            "scene": enriched.get("scene"),
            "media_urls": enriched.get("media_urls"),
            "state": enriched,
        }
    except Exception as exc:
        err_msg = str(exc) or exc.__class__.__name__
        with _state_lock:
            st = _read_job_state(job_id) or {"job_id": job_id}
            # 既存の成果物が壊れていなければエラー情報を記録して done に戻すか error にする
            st["status"] = "error"
            st["stage"] = "error"
            st["error"] = err_msg
            st["message"] = f"再Mixエラー: {err_msg}"
            _append_log(st, f"再Mixに失敗しました: {err_msg}", actor="remix", stage="error")
            _write_job_state(job_id, st)
        raise HTTPException(status_code=500, detail=f"再MixMasterに失敗しました: {err_msg}") from exc


# ---------------------------------------------------------------------------
# エージェント専用エンドポイント
# ---------------------------------------------------------------------------
@app.post("/api/agent/heartbeat")
def api_agent_heartbeat(req: Optional[AgentHeartbeatRequest] = Body(default=None)) -> dict[str, Any]:
    """常駐エージェントからの生存通知を受け取り AGENT_STATE_PATH を更新する。"""
    payload = req or AgentHeartbeatRequest()
    status_norm = (payload.status or "idle").strip().lower()
    if payload.active_job_id is not None:
        next_active: Any = payload.active_job_id
    elif "active_job_id" in payload.model_fields_set or status_norm in ("idle", "waiting", "stopped", "offline"):
        next_active = None
    else:
        next_active = ...
    info = _update_agent_state(
        agent_name=payload.agent_name,
        pid=payload.pid,
        status=payload.status,
        active_job_id=next_active,
    )
    queued_count = sum(
        1
        for st in _list_all_job_states()
        if st.get("status") == "queued_for_agent" and not st.get("cancel_requested")
    )
    return {
        "ok": True,
        "agent_connected": info["agent_connected"],
        "agent_status": info["agent_status"],
        "queued_count": queued_count,
        "server_time": _now_ts(),
    }


@app.get("/api/agent/next")
def api_agent_next() -> dict[str, Any]:
    """最も古い queued_for_agent ジョブを1件返す（なければ {"job": null}）。"""
    if SHUTDOWN_FLAG_PATH.is_file():
        return {"job": None, "shutdown": True}
    with _state_lock:
        queued = [
            st
            for st in _list_all_job_states()
            if st.get("status") == "queued_for_agent" and not st.get("cancel_requested")
        ]
        if not queued:
            return {"job": None}
        queued.sort(key=lambda s: float(s.get("created_at") or 0.0))
        oldest = queued[0]
        job_id = str(oldest["job_id"])
        jdir = _job_dir(job_id)
        req_data = oldest.get("request") or _read_json(jdir / "request.json", default={})
        return {
            "job": {
                "job_id": job_id,
                "job_dir": str(jdir.resolve()),
                "filename": oldest.get("filename"),
                "created_at": oldest.get("created_at"),
                "request": req_data,
            }
        }


@app.post("/api/agent/jobs/{job_id}/claim")
def api_agent_claim(
    job_id: str, req: Optional[AgentClaimRequest] = Body(default=None)
) -> dict[str, Any]:
    """エージェントが待機ジョブを受注し、ステータスを agent_running に遷移する。"""
    payload = req or AgentClaimRequest()
    with _state_lock:
        st = _read_job_state(job_id)
        if not st:
            raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
        if st.get("status") != "queued_for_agent":
            raise HTTPException(
                status_code=409,
                detail=f"ジョブは既に {st.get('status')} 状態のため受注できません",
            )
        if st.get("cancel_requested"):
            raise HTTPException(status_code=409, detail="ジョブはキャンセルされています")

        now = _now_ts()
        st["status"] = "agent_running"
        st["actor"] = payload.agent_name or "agent"
        st["stage"] = "inspect"
        st["started_at"] = now
        st["message"] = payload.message or f"エージェント（{st['actor']}）が処理を開始しました..."
        _append_log(
            st,
            payload.message or f"エージェント {st['actor']} がジョブを受注しました。",
            actor=st["actor"],
            stage="inspect",
        )
        _write_job_state(job_id, st)

    _update_agent_state(
        agent_name=payload.agent_name,
        pid=payload.pid,
        status="working",
        active_job_id=job_id,
    )
    jdir = _job_dir(job_id)
    req_data = st.get("request") or _read_json(jdir / "request.json", default={})
    return {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(jdir.resolve()),
        "request": req_data,
        "state": st,
    }


@app.post("/api/agent/jobs/{job_id}/progress")
def api_agent_progress(job_id: str, req: AgentProgressRequest) -> dict[str, Any]:
    """エージェントからの進捗・ステージ・思考ログを state.json へ追記する。"""
    with _state_lock:
        st = _read_job_state(job_id)
        if not st:
            raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
        if req.progress is not None:
            st["progress"] = round(max(0.0, min(100.0, float(req.progress))), 1)
        if req.stage:
            st["stage"] = req.stage
        if req.message:
            st["message"] = req.message
        log_body = req.log_text or req.message
        if log_body:
            _append_log(st, log_body, actor=req.actor or "agent", stage=req.stage or st.get("stage"))
        _write_job_state(job_id, st)
        cancel_requested = bool(st.get("cancel_requested", False))
        cur_status = st.get("status")

    _update_agent_state(status="working", active_job_id=job_id)
    return {
        "ok": True,
        "job_id": job_id,
        "status": cur_status,
        "cancel_requested": cancel_requested,
    }


@app.post("/api/agent/jobs/{job_id}/complete")
def api_agent_complete(job_id: str, req: AgentCompleteRequest) -> dict[str, Any]:
    """エージェントからのジョブ完了／エラー報告を反映する。"""
    status = (req.status or "done").strip().lower()
    if status not in ("done", "error", "cancelled"):
        status = "done"

    jdir = _job_dir(job_id)
    with _state_lock:
        st = _read_job_state(job_id)
        if not st:
            raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
        now = _now_ts()
        st["status"] = status
        st["stage"] = status
        st["finished_at"] = now
        started = st.get("started_at") or st.get("created_at") or now
        st["elapsed_sec"] = round(max(0.0, now - float(started)), 2)

        if status == "done":
            st["progress"] = 100.0
            result_data = req.result or _read_json(jdir / "result.json", default=None)
            st["result"] = result_data
            st["error"] = None
            st["message"] = req.message or "エージェントによるサラウンドMixMasterが完了しました"
            _append_log(
                st,
                req.message or f"エージェント処理が完了しました（所要時間 {st['elapsed_sec']} 秒）。",
                actor=req.actor or "agent",
                stage="done",
            )
        elif status == "cancelled":
            st["message"] = req.message or "ジョブがキャンセルされました"
            _append_log(st, st["message"], actor=req.actor or "agent", stage="cancelled")
        else:
            st["error"] = req.error or req.message or "エージェント実行中にエラーが発生しました"
            st["message"] = f"エラー: {st['error']}"
            _append_log(st, f"エラー終了: {st['error']}", actor=req.actor or "agent", stage="error")

        _write_job_state(job_id, st)

    _update_agent_state(status="idle", active_job_id=None)
    return {"ok": True, "job_id": job_id, "state": _enrich_job_payload(job_id, st)}


# ---------------------------------------------------------------------------
# 4. メディア配信・ダウンロード
# ---------------------------------------------------------------------------
_MEDIA_MIME_MAP = {
    "original_video": "video/mp4",
    "binaural_video": "video/mp4",
    "original_audio": "audio/wav",
    "binaural_audio": "audio/wav",
    "master_spatial": "audio/wav",
    "surround_714": "audio/wav",
    "stems_zip": "application/zip",
}

_MEDIA_DOWNLOAD_NAME = {
    "original_video": "original_video.mp4",
    "binaural_video": "ciness_binaural.mp4",
    "original_audio": "ciness_original_48k.wav",
    "binaural_audio": "ciness_binaural_48k.wav",
    "master_spatial": "ciness_master_spatial.wav",
    "surround_714": "ciness_surround_714.wav",
    "stems_zip": "ciness_stems.zip",
}


@app.get("/api/jobs/{job_id}/media/{kind}")
def api_job_media(
    job_id: str,
    kind: str,
    download: int = Query(0),
) -> FileResponse:
    """ジョブの映像・音声・サラウンドマスター・ZIPを Range 対応で配信する。"""
    if kind not in _MEDIA_MIME_MAP:
        raise HTTPException(status_code=404, detail=f"未対応のメディア種別です: {kind}")

    st = _read_job_state(job_id)
    if not st:
        raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")

    path = _resolve_media_path(job_id, st, kind)
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail=f"成果物ファイル ({kind}) が見つかりません")

    guessed, _ = mimetypes.guess_type(str(path))
    media_type = guessed or _MEDIA_MIME_MAP[kind]
    headers = {"Cache-Control": "no-store"}

    if download:
        stem_base = Path(st.get("filename") or job_id).stem
        dl_name = f"{stem_base}_{path.name}" if path.name else _MEDIA_DOWNLOAD_NAME[kind]
        return FileResponse(str(path), media_type=media_type, filename=dl_name, headers=headers)
    return FileResponse(str(path), media_type=media_type, headers=headers)


@app.get("/api/jobs/{job_id}/stems/{stem_name}")
def api_job_stem(
    job_id: str,
    stem_name: str,
    download: int = Query(0),
) -> FileResponse:
    """個別分離ステム／話者別WAVを試聴・ダウンロード配信する。"""
    safe_name = Path(stem_name).name
    if safe_name != stem_name or ".." in stem_name or not safe_name:
        raise HTTPException(status_code=400, detail="不正なステムファイル名です")

    jdir = _job_dir(job_id)
    if not jdir.is_dir():
        raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")

    names_to_try = [safe_name]
    if not safe_name.lower().endswith(".wav"):
        names_to_try.append(safe_name + ".wav")

    target: Optional[Path] = None
    for cand_name in names_to_try:
        for sub in ("stems", "speakers", "separated", "audio", ""):
            cand = (jdir / sub / cand_name) if sub else (jdir / cand_name)
            if cand.is_file():
                target = cand
                break
        if target:
            break

    if not target:
        scene = _read_json(jdir / "scene.json", default=None)
        if isinstance(scene, dict):
            for obj in scene.get("objects") or []:
                raw_path = obj.get("audio_path")
                if not raw_path:
                    continue
                p = Path(raw_path)
                if not p.is_absolute():
                    p = jdir / p
                if p.is_file() and (p.name in names_to_try or p.stem == safe_name):
                    target = p
                    break

    if not target or not target.is_file():
        raise HTTPException(status_code=404, detail=f"ステムが見つかりません: {safe_name}")

    headers = {"Cache-Control": "no-store"}
    if download:
        return FileResponse(str(target), media_type="audio/wav", filename=target.name, headers=headers)
    return FileResponse(str(target), media_type="audio/wav", headers=headers)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CineSS Web サーバーを起動します")
    parser.add_argument("--host", default="127.0.0.1", help="バインドホスト (既定: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"ポート番号 (既定: {DEFAULT_PORT})")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

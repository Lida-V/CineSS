# coding: utf-8
"""CineSS AIダビング監督（自動演出プラン生成・安全検証・冪等シーン合成）モジュール。

映像解析から得たシーン温度カーブ（`intensity_track`）・状態区間（`scene_segments`）と、
SFX 4層分解の衝撃イベント（`impact_events`）を組み合わせ、映画のダビングステージのような
時間変化ミックス演出（ゲイン変化・距離演出・スマートダッキング免除）を構築・適用する。
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

MIN_DIRECTOR_GAIN_DB = -18.0
MAX_DIRECTOR_GAIN_DB = 12.0
MIN_DIRECTOR_DISTANCE_M = 0.6
MAX_DIRECTOR_DISTANCE_M = 12.0

_SEGMENT_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "quiet": {
        "sfx_impact": {"gain_db": -1.8, "distance_m": 3.8, "duck_exempt": False},
        "sfx": {"gain_db": -1.8, "distance_m": 3.8, "duck_exempt": False},
        "foley": {"gain_db": -1.5, "distance_m": 3.8, "duck_exempt": False},
        "sfx_sub": {"gain_db": -2.0, "distance_m": None, "duck_exempt": True},
        "ambience": {"gain_db": -2.5, "distance_m": None, "duck_exempt": False},
        "music": {"gain_db": -1.2, "distance_m": None, "duck_exempt": False},
    },
    "normal": {
        "sfx_impact": {"gain_db": 0.0, "distance_m": None, "duck_exempt": False},
        "sfx": {"gain_db": 0.0, "distance_m": None, "duck_exempt": False},
        "foley": {"gain_db": 0.0, "distance_m": None, "duck_exempt": False},
        "sfx_sub": {"gain_db": 0.0, "distance_m": None, "duck_exempt": True},
        "ambience": {"gain_db": 0.0, "distance_m": None, "duck_exempt": False},
        "music": {"gain_db": 0.0, "distance_m": None, "duck_exempt": False},
    },
    "rising": {
        "sfx_impact": {"gain_db": 1.6, "distance_m": 2.0, "duck_exempt": True},
        "sfx": {"gain_db": 1.6, "distance_m": 2.0, "duck_exempt": True},
        "foley": {"gain_db": 1.4, "distance_m": 2.0, "duck_exempt": True},
        "sfx_sub": {"gain_db": 1.8, "distance_m": None, "duck_exempt": True},
        "ambience": {"gain_db": 0.5, "distance_m": None, "duck_exempt": False},
        "music": {"gain_db": 0.8, "distance_m": None, "duck_exempt": False},
    },
    "climax": {
        "sfx_impact": {"gain_db": 3.2, "distance_m": 1.55, "duck_exempt": True},
        "sfx": {"gain_db": 3.0, "distance_m": 1.60, "duck_exempt": True},
        "foley": {"gain_db": 2.8, "distance_m": 1.60, "duck_exempt": True},
        "sfx_sub": {"gain_db": 3.5, "distance_m": None, "duck_exempt": True},
        "ambience": {"gain_db": 1.0, "distance_m": None, "duck_exempt": True},
        "music": {"gain_db": 1.2, "distance_m": None, "duck_exempt": False},
    },
}


def _finite_float(val: Any, default: float) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp_gain(val: Any) -> float:
    return round(max(MIN_DIRECTOR_GAIN_DB, min(MAX_DIRECTOR_GAIN_DB, _finite_float(val, 0.0))), 2)


def _clamp_distance(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = _finite_float(val, float("nan"))
    if not math.isfinite(f):
        return None
    return round(max(MIN_DIRECTOR_DISTANCE_M, min(MAX_DIRECTOR_DISTANCE_M, f)), 2)


def _kf_time(kf: dict[str, Any]) -> float:
    if "t_sec" in kf and kf["t_sec"] is not None:
        return _finite_float(kf["t_sec"], 0.0)
    return _finite_float(kf.get("t", 0.0), 0.0)


def _dedupe_sort_track(kfs: list[dict[str, Any]], duration_sec: float) -> list[dict[str, Any]]:
    by_t: dict[float, dict[str, Any]] = {}
    for raw in kfs:
        if not isinstance(raw, dict):
            continue
        t_val = round(max(0.0, min(duration_sec, _kf_time(raw))), 3)
        entry: dict[str, Any] = {
            "t_sec": t_val,
            "gain_db": _clamp_gain(raw.get("gain_db", 0.0)),
            "distance_m": _clamp_distance(raw.get("distance_m")),
            "duck_exempt": bool(raw.get("duck_exempt", False)),
        }
        if raw.get("reason"):
            entry["reason"] = str(raw["reason"])
        by_t[t_val] = entry
    return [by_t[t] for t in sorted(by_t.keys())]


def _segment_note_ja(label_code: str, start_sec: float, end_sec: float, mean_temp: float) -> str:
    span = f"{start_sec:.1f}s〜{end_sec:.1f}s"
    if label_code == "climax":
        return f"[{span}] クライマックス (温度 {mean_temp:.2f}): 衝撃SFXとSubを+3.2〜+3.5dB前に押し出し、セリフ中もダッキング免除で迫力を維持。"
    if label_code == "rising":
        return f"[{span}] 高揚区間 (温度 {mean_temp:.2f}): 効果音を+1.6dB・近距離(2.0m)へ寄せ、低域の厚みを増加。"
    if label_code == "quiet":
        return f"[{span}] 静寂区間 (温度 {mean_temp:.2f}): 環境音を-2.5dB控えめ・効果音を遠景(3.8m)へ引いて静けさと明瞭度を確保。"
    return f"[{span}] 通常区間 (温度 {mean_temp:.2f}): 映像追従の標準空間バランスを維持。"


def build_auto_director_plan(
    vision_result: dict[str, Any] | None,
    sfx_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """シーン温度カーブ・状態区間・衝撃イベントから AI ダビング監督プランを自動生成する。"""
    vr = vision_result or {}
    duration_sec = max(0.1, _finite_float(vr.get("duration_sec"), 1.0))
    segments = list(vr.get("scene_segments") or [])
    if not segments:
        segments = [{
            "segment_id": 0,
            "start_sec": 0.0,
            "end_sec": round(duration_sec, 3),
            "duration_sec": round(duration_sec, 3),
            "mean_temperature": 0.3,
            "peak_temperature": 0.3,
            "label": "通常",
            "label_code": "normal",
        }]

    tracks: dict[str, list[dict[str, Any]]] = {
        role: [] for role in ("sfx_impact", "sfx", "foley", "sfx_sub", "ambience", "music")
    }
    notes: list[dict[str, Any]] = []

    ramp = min(0.22, duration_sec * 0.08)
    for seg_idx, seg in enumerate(segments):
        s_t = max(0.0, min(duration_sec, _finite_float(seg.get("start_sec"), 0.0)))
        e_t = max(s_t, min(duration_sec, _finite_float(seg.get("end_sec"), duration_sec)))
        code = str(seg.get("label_code") or "normal").lower()
        if code not in _SEGMENT_PRESETS:
            code = "normal"
        preset = _SEGMENT_PRESETS[code]
        mean_temp = _finite_float(seg.get("mean_temperature"), 0.3)

        notes.append({
            "t_sec": round(s_t, 3),
            "end_sec": round(e_t, 3),
            "label": str(seg.get("label") or code),
            "label_code": code,
            "mean_temperature": round(mean_temp, 3),
            "note": _segment_note_ja(code, s_t, e_t, mean_temp),
        })

        sample_times = [s_t]
        if (e_t - s_t) > 2.2 * ramp and seg_idx > 0:
            sample_times.append(round(s_t + ramp, 3))
        if (e_t - s_t) > 2.2 * ramp and seg_idx < len(segments) - 1:
            sample_times.append(round(e_t - ramp, 3))
        if seg_idx == len(segments) - 1:
            sample_times.append(e_t)

        for role, p in preset.items():
            for idx_t, t_pt in enumerate(sample_times):
                # 境界の開始点は直前区間との補間用に滑らかに遷移させる
                tracks[role].append({
                    "t_sec": round(t_pt, 3),
                    "gain_db": p["gain_db"],
                    "distance_m": p["distance_m"],
                    "duck_exempt": bool(p["duck_exempt"]),
                    "reason": f"{seg.get('label', code)}区間" if idx_t == 0 else None,
                })

    # SFX 衝撃イベントに基づく瞬間パンチ・近接強調ポイントを重畳
    raw_events = list((sfx_info or {}).get("impact_events") or [])
    strong_events = [
        ev for ev in raw_events
        if _finite_float(ev.get("peak_db"), -120.0) >= -28.0
    ]
    if not strong_events and raw_events:
        strong_events = sorted(
            raw_events,
            key=lambda e: _finite_float(e.get("peak_db"), -120.0),
            reverse=True,
        )[:4]
    strong_events = sorted(strong_events[:12], key=lambda e: _finite_float(e.get("t_sec"), 0.0))

    applied_events: list[dict[str, Any]] = []
    for ev in strong_events:
        ev_t = max(0.0, min(duration_sec, _finite_float(ev.get("t_sec"), 0.0)))
        ev_peak = _finite_float(ev.get("peak_db"), -20.0)
        side = str(ev.get("dominant_side") or "center")

        # イベント時刻が属する区間の基準ゲインを確認し、自然なパンチ量を加味
        seg_match = next(
            (s for s in segments if _finite_float(s.get("start_sec"), 0.0) <= ev_t <= _finite_float(s.get("end_sec"), duration_sec)),
            segments[-1],
        )
        seg_code = str(seg_match.get("label_code") or "normal")
        base_imp_gain = _SEGMENT_PRESETS.get(seg_code, _SEGMENT_PRESETS["normal"])["sfx_impact"]["gain_db"]
        boost_gain = min(4.0, max(base_imp_gain, 2.4 if ev_peak >= -16.0 else 1.8))
        boost_sub = min(4.0, max(_SEGMENT_PRESETS.get(seg_code, _SEGMENT_PRESETS["normal"])["sfx_sub"]["gain_db"], 2.2))

        for role, g_val, d_val in (
            ("sfx_impact", boost_gain, 1.65),
            ("sfx", boost_gain, 1.65),
            ("sfx_sub", boost_sub, None),
        ):
            tracks[role].append({
                "t_sec": round(ev_t, 3),
                "gain_db": round(g_val, 2),
                "distance_m": d_val,
                "duck_exempt": True,
                "reason": f"衝撃ピーク強調 ({side}, {ev_peak:.1f} dBFS)",
            })

        applied_events.append({
            "t_sec": round(ev_t, 3),
            "peak_db": round(ev_peak, 2),
            "dominant_side": side,
            "boost_gain_db": round(boost_gain, 2),
        })

    climax_cnt = sum(1 for s in segments if str(s.get("label_code")) == "climax")
    quiet_cnt = sum(1 for s in segments if str(s.get("label_code")) == "quiet")
    summary_ja = (
        f"全{len(segments)}区間（クライマックス{climax_cnt}区間・静寂{quiet_cnt}区間）と"
        f"衝撃ピーク{len(applied_events)}点を解析し、時間変化ダイナミック演出を自動設計しました。"
    )

    plan = {
        "version": 1,
        "source": "auto_director",
        "duration_sec": round(duration_sec, 3),
        "summary_ja": summary_ja,
        "notes": notes,
        "events": applied_events,
        "tracks": tracks,
    }
    return validate_director_plan(plan, duration_sec)


def validate_director_plan(plan: dict[str, Any] | None, duration_sec: float) -> dict[str, Any]:
    """任意の監督プランを安全検証・正規化し、ゲイン・距離・時刻範囲を保証する。"""
    if plan is None:
        raw_plan: dict[str, Any] = {}
    elif isinstance(plan, dict):
        raw_plan = copy.deepcopy(plan)
    else:
        raise ValueError("director_plan は辞書 (dict) である必要があります。")

    dur = max(0.1, _finite_float(duration_sec, _finite_float(raw_plan.get("duration_sec"), 1.0)))

    raw_tracks = raw_plan.get("tracks")
    if not isinstance(raw_tracks, dict):
        # トップレベルに直接役割キーや単一 keyframes が渡された形式も柔軟に正規化
        raw_tracks = {}
        known_roles = ("sfx_impact", "sfx", "foley", "sfx_sub", "ambience", "music", "dialog",
                       "sfx_impact_left", "sfx_impact_center", "sfx_impact_right", "sfx_ambience")
        for key in known_roles:
            if isinstance(raw_plan.get(key), list):
                raw_tracks[key] = raw_plan[key]
        if not raw_tracks and isinstance(raw_plan.get("keyframes"), list):
            raw_tracks["sfx_impact"] = raw_plan["keyframes"]
            raw_tracks["sfx"] = raw_plan["keyframes"]

    clean_tracks: dict[str, list[dict[str, Any]]] = {}
    for track_key, kfs in raw_tracks.items():
        if not isinstance(kfs, list):
            continue
        cleaned = _dedupe_sort_track(kfs, dur)
        if cleaned:
            clean_tracks[str(track_key)] = cleaned

    raw_notes = raw_plan.get("notes")
    clean_notes: list[dict[str, Any]] = []
    if isinstance(raw_notes, list):
        for n in raw_notes:
            if isinstance(n, dict):
                clean_notes.append({
                    "t_sec": round(max(0.0, min(dur, _finite_float(n.get("t_sec"), 0.0))), 3),
                    "end_sec": round(max(0.0, min(dur, _finite_float(n.get("end_sec"), dur))), 3),
                    "label": str(n.get("label") or ""),
                    "label_code": str(n.get("label_code") or "normal"),
                    "mean_temperature": round(_finite_float(n.get("mean_temperature"), 0.3), 3),
                    "note": str(n.get("note") or ""),
                })
            elif isinstance(n, str) and n.strip():
                clean_notes.append({"t_sec": 0.0, "end_sec": round(dur, 3), "label": "", "label_code": "normal", "mean_temperature": 0.3, "note": n.strip()})

    return {
        "version": int(raw_plan.get("version") or 1),
        "source": str(raw_plan.get("source") or "custom_director"),
        "duration_sec": round(dur, 3),
        "summary_ja": str(raw_plan.get("summary_ja") or "カスタム監督演出プラン"),
        "notes": clean_notes,
        "events": list(raw_plan.get("events") or []),
        "tracks": clean_tracks,
    }


def _select_track_for_object(obj: dict[str, Any], tracks: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]] | None:
    obj_id = str(obj.get("id") or "").strip().lower()
    role = str(obj.get("role") or "").strip().lower()
    candidates = [obj_id, role]
    if role == "sfx_impact":
        candidates.append("sfx")
    elif role == "foley":
        candidates.append("sfx")
    elif obj_id == "foley_ambience" and "ambience" in tracks:
        candidates.append("ambience")

    for key in candidates:
        if key and isinstance(tracks.get(key), list) and tracks[key]:
            return tracks[key]
    return None


def _interpolate_base_keyframes(
    base_kfs: list[dict[str, Any]],
    base_gain_db: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not base_kfs:
        n = times.shape[0]
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.full(n, 2.5, dtype=np.float64),
            np.full(n, base_gain_db, dtype=np.float64),
        )
    sorted_kf = sorted((k for k in base_kfs if isinstance(k, dict)), key=_kf_time)
    t_b = np.array([_kf_time(k) for k in sorted_kf], dtype=np.float64)
    az_b = np.array([_finite_float(k.get("azimuth_deg"), 0.0) for k in sorted_kf], dtype=np.float64)
    el_b = np.array([_finite_float(k.get("elevation_deg"), 0.0) for k in sorted_kf], dtype=np.float64)
    d_b = np.array([_finite_float(k.get("distance_m"), 2.5) for k in sorted_kf], dtype=np.float64)
    g_b = np.array([
        base_gain_db if k.get("gain_db") is None else _finite_float(k.get("gain_db"), base_gain_db)
        for k in sorted_kf
    ], dtype=np.float64)

    if len(t_b) == 1:
        n = times.shape[0]
        return (
            np.full(n, az_b[0], dtype=np.float64),
            np.full(n, el_b[0], dtype=np.float64),
            np.full(n, d_b[0], dtype=np.float64),
            np.full(n, g_b[0], dtype=np.float64),
        )

    az_unwrapped = np.unwrap(np.radians(az_b))
    az_out = ((np.degrees(np.interp(times, t_b, az_unwrapped)) + 180.0) % 360.0) - 180.0
    el_out = np.interp(times, t_b, el_b)
    d_out = np.interp(times, t_b, d_b)
    g_out = np.interp(times, t_b, g_b)
    return az_out, el_out, d_out, g_out


def apply_director_plan_to_scene(
    scene: dict[str, Any],
    director_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """シーンへ監督プランを冪等に適用する（初回 base_keyframes を保持し累積させない）。"""
    new_scene = copy.deepcopy(scene)
    duration_sec = max(0.1, _finite_float(new_scene.get("duration_sec"), 1.0))
    plan = validate_director_plan(director_plan, duration_sec)
    tracks = plan.get("tracks") or {}

    for obj in new_scene.get("objects") or []:
        if not isinstance(obj, dict):
            continue

        # 初回基準状態を保存（既存なら絶対に上書きせず、毎回そこから再計算する）
        if "base_gain_db" not in obj:
            obj["base_gain_db"] = round(_finite_float(obj.get("gain_db"), 0.0), 3)
        if "base_duck_exempt" not in obj:
            obj["base_duck_exempt"] = bool(obj.get("duck_exempt", False))
        if "base_keyframes" not in obj:
            raw_kfs = obj.get("keyframes") or [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.5}]
            obj["base_keyframes"] = copy.deepcopy(raw_kfs)

        base_gain_db = _finite_float(obj["base_gain_db"], 0.0)
        base_duck_exempt = bool(obj["base_duck_exempt"])
        base_kfs = list(obj["base_keyframes"] or [])

        dir_kfs = _select_track_for_object(obj, tracks)
        if not dir_kfs:
            # 対象指示が無いトラックはクリーンに初回基準へ復元
            restored_kfs = copy.deepcopy(base_kfs)
            for rk in restored_kfs:
                t_v = round(_kf_time(rk), 3)
                rk["t"] = t_v
                rk["t_sec"] = t_v
            obj["gain_db"] = base_gain_db
            obj["duck_exempt"] = base_duck_exempt
            obj["keyframes"] = restored_kfs
            obj["director_applied"] = False
            continue

        # ベース時刻と監督プラン時刻の和集合で評価時刻軸を生成
        t_set = {round(max(0.0, min(duration_sec, _kf_time(k))), 3) for k in base_kfs if isinstance(k, dict)}
        t_set.update(round(float(k["t_sec"]), 3) for k in dir_kfs)
        if not t_set:
            t_set.add(0.0)
        grid_t = np.array(sorted(t_set), dtype=np.float64)

        az_arr, el_arr, d_base_arr, g_base_arr = _interpolate_base_keyframes(base_kfs, base_gain_db, grid_t)

        t_dir = np.array([float(k["t_sec"]) for k in dir_kfs], dtype=np.float64)
        g_dir = np.array([float(k["gain_db"]) for k in dir_kfs], dtype=np.float64)
        dir_gain_interp = np.interp(grid_t, t_dir, g_dir)

        # 距離指示がある時刻では監督指定距離を補間採用（None の箇所はベース距離をフォールバック）
        has_any_dist = any(k.get("distance_m") is not None for k in dir_kfs)
        if has_any_dist:
            _, _, d_base_at_dir, _ = _interpolate_base_keyframes(base_kfs, base_gain_db, t_dir)
            d_dir_vals = np.array(
                [
                    float(d_base_at_dir[i]) if k.get("distance_m") is None else float(k["distance_m"])
                    for i, k in enumerate(dir_kfs)
                ],
                dtype=np.float64,
            )
            dist_interp = np.interp(grid_t, t_dir, d_dir_vals)
        else:
            dist_interp = d_base_arr

        # duck_exempt は直前監督キーフレームの状態を区間保持
        dir_exempt_vals = [bool(k.get("duck_exempt", False)) for k in dir_kfs]
        exempt_indices = np.clip(np.searchsorted(t_dir, grid_t, side="right") - 1, 0, len(dir_kfs) - 1)

        is_bed = bool(obj.get("is_bed", False))
        merged_kfs: list[dict[str, Any]] = []
        for i, t_val in enumerate(grid_t):
            t_r = round(float(t_val), 3)
            final_gain = _clamp_gain(g_base_arr[i] + dir_gain_interp[i])
            final_dist = round(max(MIN_DIRECTOR_DISTANCE_M, min(MAX_DIRECTOR_DISTANCE_M, float(dist_interp[i]))), 2)
            kf_exempt = bool(base_duck_exempt or dir_exempt_vals[int(exempt_indices[i])])
            merged_kfs.append({
                "t": t_r,
                "t_sec": t_r,
                "azimuth_deg": 0.0 if is_bed else round(float(az_arr[i]), 2),
                "elevation_deg": 0.0 if is_bed else round(float(el_arr[i]), 2),
                "distance_m": round(float(d_base_arr[i]), 2) if is_bed else final_dist,
                "gain_db": final_gain,
                "duck_exempt": kf_exempt,
            })

        obj["gain_db"] = base_gain_db
        obj["duck_exempt"] = base_duck_exempt
        obj["keyframes"] = merged_kfs
        obj["director_applied"] = True

    new_scene["director_plan"] = plan
    return new_scene

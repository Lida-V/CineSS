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
MIN_DIRECTOR_AZIMUTH_DEG = -180.0
MAX_DIRECTOR_AZIMUTH_DEG = 180.0
MIN_DIRECTOR_ELEVATION_DEG = -90.0
MAX_DIRECTOR_ELEVATION_DEG = 90.0
MIN_REAR_LEVEL = 0.4
MAX_REAR_LEVEL = 1.6
MIN_HEIGHT_LEVEL = 0.3
MAX_HEIGHT_LEVEL = 1.6
MIN_RT60_SEC = 0.15
MAX_RT60_SEC = 1.5

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


def _clamp_azimuth(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = _finite_float(val, float("nan"))
    if not math.isfinite(f):
        return None
    wrapped = ((f + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and f > 0.0:
        wrapped = 180.0
    return round(max(MIN_DIRECTOR_AZIMUTH_DEG, min(MAX_DIRECTOR_AZIMUTH_DEG, wrapped)), 2)


def _clamp_azimuth_offset(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = _finite_float(val, float("nan"))
    if not math.isfinite(f):
        return None
    return round(max(MIN_DIRECTOR_AZIMUTH_DEG, min(MAX_DIRECTOR_AZIMUTH_DEG, f)), 2)


def _clamp_elevation(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = _finite_float(val, float("nan"))
    if not math.isfinite(f):
        return None
    return round(max(MIN_DIRECTOR_ELEVATION_DEG, min(MAX_DIRECTOR_ELEVATION_DEG, f)), 2)


def _clamp_spread(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = _finite_float(val, float("nan"))
    if not math.isfinite(f):
        return None
    return round(max(0.0, min(1.0, f)), 3)


def _kf_time(kf: dict[str, Any]) -> float:
    if "t_sec" in kf and kf["t_sec"] is not None:
        return _finite_float(kf["t_sec"], 0.0)
    return _finite_float(kf.get("t", 0.0), 0.0)


def _normalize_world_360(raw_world: Any) -> dict[str, Any] | None:
    if not isinstance(raw_world, dict):
        return None
    enabled = bool(raw_world.get("enabled", True))
    env_type = str(raw_world.get("environment_type") or "indoor_intimate")
    env_ja = str(raw_world.get("environment_label_ja") or raw_world.get("label_ja") or "360°拡張空間")
    rear_default = 1.0 if enabled else 0.65
    height_default = 1.0 if enabled else 0.55
    rear_lvl = round(max(MIN_REAR_LEVEL, min(MAX_REAR_LEVEL, _finite_float(raw_world.get("rear_level"), rear_default))), 2)
    height_lvl = round(max(MIN_HEIGHT_LEVEL, min(MAX_HEIGHT_LEVEL, _finite_float(raw_world.get("height_level"), height_default))), 2)
    wrap_lvl = round(max(0.0, min(1.0, _finite_float(raw_world.get("surround_wrap"), 0.65 if enabled else 0.0))), 2)
    rt60_raw = raw_world.get("rt60_sec")
    rt60_val: float | None = None
    if rt60_raw not in (None, ""):
        f_rt = _finite_float(rt60_raw, float("nan"))
        if math.isfinite(f_rt):
            rt60_val = round(max(MIN_RT60_SEC, min(MAX_RT60_SEC, f_rt)), 2)
    return {
        "enabled": enabled,
        "environment_type": env_type,
        "environment_label_ja": env_ja,
        "rear_level": rear_lvl,
        "height_level": height_lvl,
        "surround_wrap": wrap_lvl,
        "rt60_sec": rt60_val,
    }


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
        az = _clamp_azimuth(raw.get("azimuth_deg"))
        az_off = _clamp_azimuth_offset(raw.get("azimuth_offset_deg"))
        el = _clamp_elevation(raw.get("elevation_deg"))
        el_off = _clamp_elevation(raw.get("elevation_offset_deg"))
        spread = _clamp_spread(raw.get("spread"))
        if az is not None:
            entry["azimuth_deg"] = az
        if az_off is not None:
            entry["azimuth_offset_deg"] = az_off
        if el is not None:
            entry["elevation_deg"] = el
        if el_off is not None:
            entry["elevation_offset_deg"] = el_off
        if spread is not None:
            entry["spread"] = spread
        if raw.get("reason"):
            entry["reason"] = str(raw["reason"])

        prev = by_t.get(t_val)
        if prev is not None:
            # 同一時刻に区間基準点と通過/衝撃イベントが重なった場合、明示座標を維持しつつ演出を統合
            for opt_k in ("distance_m", "azimuth_deg", "azimuth_offset_deg", "elevation_deg", "elevation_offset_deg", "spread", "reason"):
                if opt_k not in entry or entry[opt_k] is None:
                    if prev.get(opt_k) is not None:
                        entry[opt_k] = prev[opt_k]
            if prev.get("duck_exempt"):
                entry["duck_exempt"] = True
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


def _preset_gain_at(
    segments: list[dict[str, Any]],
    t_sec: float,
    role: str,
    duration_sec: float,
) -> float:
    seg_match = next(
        (
            s for s in segments
            if _finite_float(s.get("start_sec"), 0.0) <= t_sec <= _finite_float(s.get("end_sec"), duration_sec)
        ),
        segments[-1],
    )
    seg_code = str(seg_match.get("label_code") or "normal").lower()
    if seg_code not in _SEGMENT_PRESETS:
        seg_code = "normal"
    return float(_SEGMENT_PRESETS[seg_code].get(role, _SEGMENT_PRESETS[seg_code]["sfx_impact"])["gain_db"])


def build_auto_director_plan(
    vision_result: dict[str, Any] | None,
    sfx_info: dict[str, Any] | None = None,
    enable_world_360: bool = True,
) -> dict[str, Any]:
    """シーン温度カーブ・状態区間・360°ワールド情報・衝撃イベントから AI ダビング監督プランを自動生成する。"""
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

    wp = dict(vr.get("world_profile") or {})
    env_type = str(wp.get("environment_type") or "indoor_intimate")
    env_ja = str(wp.get("environment_label_ja") or "室内・標準空間")
    rear_strength = max(0.50, min(1.0, _finite_float(wp.get("rear_immersion_strength"), 0.65)))
    sky_presence = max(0.25, min(1.0, _finite_float(wp.get("sky_overhead_presence"), 0.40)))
    ground_act = max(0.0, min(1.0, _finite_float(wp.get("ground_activity"), 0.25)))
    rec_rt60 = wp.get("recommended_rt60_sec")

    if enable_world_360:
        world_360: dict[str, Any] = {
            "enabled": True,
            "environment_type": env_type,
            "environment_label_ja": env_ja,
            "rear_level": round(0.85 + 0.55 * rear_strength, 2),
            "height_level": round(0.75 + 0.60 * sky_presence, 2),
            "surround_wrap": round(0.45 + 0.45 * rear_strength, 2),
            "rt60_sec": round(float(rec_rt60), 2) if rec_rt60 is not None else None,
        }
    else:
        world_360 = {
            "enabled": False,
            "environment_type": env_type,
            "environment_label_ja": env_ja,
            "rear_level": 0.65,
            "height_level": 0.55,
            "surround_wrap": 0.0,
            "rt60_sec": None,
        }

    tracks: dict[str, list[dict[str, Any]]] = {
        role: []
        for role in (
            "sfx_impact",
            "sfx_impact_left",
            "sfx_impact_center",
            "sfx_impact_right",
            "sfx",
            "foley",
            "sfx_sub",
            "ambience",
            "music",
        )
    }
    notes: list[dict[str, Any]] = []

    if enable_world_360:
        notes.append({
            "t_sec": 0.0,
            "end_sec": round(duration_sec, 3),
            "label": "360°世界",
            "label_code": "world_360",
            "mean_temperature": round(rear_strength, 3),
            "note": (
                f"[全区間] 360°想定空間 ({env_ja}): 後方包囲 {world_360['rear_level']:.2f}x / "
                f"頭上ハイト {world_360['height_level']:.2f}x / 残響 RT60={world_360['rt60_sec'] or '--'}s"
            ),
        })

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

        # クライマックス＆高揚区間の 360° サラウンド・ラップ（左右衝撃音を側面・ハイト方向へ展開）
        if enable_world_360:
            if code == "climax":
                az_off_mag = 36.0
                el_off_val = round(14.0 * sky_presence, 1)
            elif code == "rising":
                az_off_mag = 16.0
                el_off_val = round(8.0 * sky_presence, 1)
            else:
                az_off_mag = 0.0
                el_off_val = 0.0
        else:
            az_off_mag = 0.0
            el_off_val = 0.0

        for role, p in preset.items():
            for idx_t, t_pt in enumerate(sample_times):
                entry = {
                    "t_sec": round(t_pt, 3),
                    "gain_db": p["gain_db"],
                    "distance_m": p["distance_m"],
                    "duck_exempt": bool(p["duck_exempt"]),
                    "reason": f"{seg.get('label', code)}区間" if idx_t == 0 else None,
                }
                tracks[role].append(entry)
                if role == "sfx_impact":
                    tracks["sfx_impact_center"].append(dict(entry))
                    left_entry = dict(entry)
                    right_entry = dict(entry)
                    if enable_world_360:
                        left_entry["azimuth_offset_deg"] = -az_off_mag
                        left_entry["elevation_offset_deg"] = el_off_val
                        right_entry["azimuth_offset_deg"] = az_off_mag
                        right_entry["elevation_offset_deg"] = el_off_val
                    tracks["sfx_impact_left"].append(left_entry)
                    tracks["sfx_impact_right"].append(right_entry)

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
            ("sfx_impact_left", boost_gain, 1.65),
            ("sfx_impact_right", boost_gain, 1.65),
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

        center_ev: dict[str, Any] = {
            "t_sec": round(ev_t, 3),
            "gain_db": round(boost_gain, 2),
            "distance_m": 1.65,
            "duck_exempt": True,
            "reason": f"衝撃ピーク強調 ({side}, {ev_peak:.1f} dBFS)",
        }
        if enable_world_360 and ground_act >= 0.45 and seg_code != "climax":
            center_ev["elevation_offset_deg"] = -10.0
        tracks["sfx_impact_center"].append(center_ev)

        applied_events.append({
            "t_sec": round(ev_t, 3),
            "peak_db": round(ev_peak, 2),
            "dominant_side": side,
            "boost_gain_db": round(boost_gain, 2),
        })

    # 画面外通過（フライバイ）＆全方位衝撃波による 360° 立体軌跡注入
    flyby_events = list(vr.get("flyby_events") or []) if enable_world_360 else []
    applied_flybys: list[dict[str, Any]] = []
    for f_ev in flyby_events:
        if not isinstance(f_ev, dict):
            continue
        f_t = max(0.0, min(duration_sec, _finite_float(f_ev.get("t_sec"), 0.0)))
        kind = str(f_ev.get("kind") or "")
        base_g = _preset_gain_at(segments, f_t, "sfx_impact", duration_sec)
        seg_match = next(
            (s for s in segments if _finite_float(s.get("start_sec"), 0.0) <= f_t <= _finite_float(s.get("end_sec"), duration_sec)),
            segments[-1],
        )
        ret_off = 36.0 if str(seg_match.get("label_code")) == "climax" else (16.0 if str(seg_match.get("label_code")) == "rising" else 0.0)

        if kind == "side_rear_left":
            tracks["sfx_impact_left"].extend([
                {"t_sec": round(max(0.0, f_t - 0.15), 3), "azimuth_deg": -42.0, "distance_m": 1.8, "gain_db": round(min(4.0, base_g + 1.2), 2), "duck_exempt": True, "reason": "左画面外通過 (進入)"},
                {"t_sec": round(min(duration_sec, f_t + 0.12), 3), "azimuth_deg": -95.0, "distance_m": 1.35, "gain_db": round(min(4.0, base_g + 3.0), 2), "duck_exempt": True, "reason": "左側面通過 (-95°)"},
                {"t_sec": round(min(duration_sec, f_t + 0.42), 3), "azimuth_deg": -138.0, "distance_m": 2.1, "gain_db": round(min(4.0, base_g + 1.5), 2), "duck_exempt": True, "reason": "左後方へ抜ける (-138°)"},
                {"t_sec": round(min(duration_sec, f_t + 0.72), 3), "azimuth_offset_deg": -ret_off, "gain_db": round(base_g, 2), "duck_exempt": False},
            ])
        elif kind == "side_rear_right":
            tracks["sfx_impact_right"].extend([
                {"t_sec": round(max(0.0, f_t - 0.15), 3), "azimuth_deg": 42.0, "distance_m": 1.8, "gain_db": round(min(4.0, base_g + 1.2), 2), "duck_exempt": True, "reason": "右画面外通過 (進入)"},
                {"t_sec": round(min(duration_sec, f_t + 0.12), 3), "azimuth_deg": 95.0, "distance_m": 1.35, "gain_db": round(min(4.0, base_g + 3.0), 2), "duck_exempt": True, "reason": "右側面通過 (+95°)"},
                {"t_sec": round(min(duration_sec, f_t + 0.42), 3), "azimuth_deg": 138.0, "distance_m": 2.1, "gain_db": round(min(4.0, base_g + 1.5), 2), "duck_exempt": True, "reason": "右後方へ抜ける (+138°)"},
                {"t_sec": round(min(duration_sec, f_t + 0.72), 3), "azimuth_offset_deg": ret_off, "gain_db": round(base_g, 2), "duck_exempt": False},
            ])
        elif kind == "overhead_pass":
            peak_el = round(max(50.0, min(78.0, _finite_float(f_ev.get("peak_elevation_deg"), 64.0))), 1)
            for tname in ("sfx_impact_center", "sfx", "foley"):
                tracks[tname].extend([
                    {"t_sec": round(max(0.0, f_t - 0.12), 3), "elevation_deg": 24.0, "distance_m": 1.9, "gain_db": round(min(4.0, base_g + 1.2), 2), "duck_exempt": True, "reason": "頭上通過 (上昇)"},
                    {"t_sec": round(min(duration_sec, f_t + 0.12), 3), "elevation_deg": peak_el, "distance_m": 1.5, "gain_db": round(min(4.0, base_g + 2.6), 2), "duck_exempt": True, "reason": f"頭上ハイト通過 (+{peak_el:.0f}°)"},
                    {"t_sec": round(min(duration_sec, f_t + 0.45), 3), "elevation_deg": 22.0, "distance_m": 2.2, "gain_db": round(min(4.0, base_g + 1.0), 2), "duck_exempt": True},
                    {"t_sec": round(min(duration_sec, f_t + 0.75), 3), "elevation_offset_deg": 0.0, "gain_db": round(base_g, 2), "duck_exempt": False},
                ])
        elif kind == "surround_burst":
            tracks["sfx_impact_left"].extend([
                {"t_sec": round(f_t, 3), "azimuth_deg": -115.0, "elevation_offset_deg": 22.0, "distance_m": 1.45, "gain_db": 3.6, "duck_exempt": True, "reason": "全方位衝撃波展開 (-115° / 頭上)"},
                {"t_sec": round(min(duration_sec, f_t + 0.55), 3), "azimuth_offset_deg": -ret_off, "elevation_offset_deg": 0.0, "gain_db": round(base_g, 2)},
            ])
            tracks["sfx_impact_right"].extend([
                {"t_sec": round(f_t, 3), "azimuth_deg": 115.0, "elevation_offset_deg": 22.0, "distance_m": 1.45, "gain_db": 3.6, "duck_exempt": True, "reason": "全方位衝撃波展開 (+115° / 頭上)"},
                {"t_sec": round(min(duration_sec, f_t + 0.55), 3), "azimuth_offset_deg": ret_off, "elevation_offset_deg": 0.0, "gain_db": round(base_g, 2)},
            ])
            for tname in ("sfx_impact_center", "sfx", "foley"):
                tracks[tname].extend([
                    {"t_sec": round(f_t, 3), "elevation_offset_deg": 18.0, "distance_m": 1.45, "gain_db": 3.5, "duck_exempt": True, "reason": "全方位衝撃波バースト"},
                    {"t_sec": round(min(duration_sec, f_t + 0.55), 3), "elevation_offset_deg": 0.0, "gain_db": round(base_g, 2)},
                ])
        else:
            continue

        applied_flybys.append(dict(f_ev))
        if f_ev.get("note_ja"):
            notes.append({
                "t_sec": round(f_t, 3),
                "end_sec": round(min(duration_sec, f_t + 0.72), 3),
                "label": "360°通過",
                "label_code": kind,
                "mean_temperature": round(_finite_float(f_ev.get("intensity"), 0.8), 3),
                "note": str(f_ev["note_ja"]),
            })

    climax_cnt = sum(1 for s in segments if str(s.get("label_code")) == "climax")
    quiet_cnt = sum(1 for s in segments if str(s.get("label_code")) == "quiet")
    w360_tag = f" / 360°({env_ja}, 通過{len(applied_flybys)}件)" if enable_world_360 else ""
    summary_ja = (
        f"全{len(segments)}区間（クライマックス{climax_cnt}区間・静寂{quiet_cnt}区間）と"
        f"衝撃ピーク{len(applied_events)}点を解析{w360_tag}し、360°ダイナミック演出を自動設計しました。"
    )

    plan = {
        "version": 1,
        "source": "auto_director",
        "duration_sec": round(duration_sec, 3),
        "summary_ja": summary_ja,
        "world_360": world_360,
        "notes": notes,
        "events": applied_events,
        "flyby_events": applied_flybys,
        "tracks": tracks,
    }
    return validate_director_plan(plan, duration_sec)


def validate_director_plan(plan: dict[str, Any] | None, duration_sec: float) -> dict[str, Any]:
    """任意の監督プランを安全検証・正規化し、ゲイン・距離・360°座標・時刻範囲を保証する。"""
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

    out_plan: dict[str, Any] = {
        "version": int(raw_plan.get("version") or 1),
        "source": str(raw_plan.get("source") or "custom_director"),
        "duration_sec": round(dur, 3),
        "summary_ja": str(raw_plan.get("summary_ja") or "カスタム監督演出プラン"),
        "notes": clean_notes,
        "events": list(raw_plan.get("events") or []),
        "tracks": clean_tracks,
    }
    norm_world = _normalize_world_360(raw_plan.get("world_360"))
    if norm_world is not None:
        out_plan["world_360"] = norm_world
    if isinstance(raw_plan.get("flyby_events"), list):
        out_plan["flyby_events"] = list(raw_plan.get("flyby_events") or [])
    return out_plan


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
    """シーンへ監督プラン（ゲイン・距離・360°方位角/仰角・world_360）を冪等に適用する。"""
    new_scene = copy.deepcopy(scene)
    duration_sec = max(0.1, _finite_float(new_scene.get("duration_sec"), 1.0))
    plan = validate_director_plan(director_plan, duration_sec)
    tracks = plan.get("tracks") or {}

    if "base_room" not in new_scene:
        new_scene["base_room"] = copy.deepcopy(new_scene.get("room") or {"rt60_sec": 0.45, "size": "medium"})
    if "base_world_360" not in new_scene:
        new_scene["base_world_360"] = copy.deepcopy(
            new_scene.get("world_360") or {"enabled": True, "rear_level": 1.0, "height_level": 1.0, "surround_wrap": 0.65}
        )

    plan_world = plan.get("world_360")
    if isinstance(plan_world, dict):
        new_scene["world_360"] = copy.deepcopy(plan_world)
    else:
        new_scene["world_360"] = copy.deepcopy(new_scene["base_world_360"])

    restored_room = copy.deepcopy(new_scene["base_room"])
    rt60_override = (new_scene.get("world_360") or {}).get("rt60_sec")
    if rt60_override is not None and math.isfinite(float(rt60_override)):
        restored_room["rt60_sec"] = round(max(MIN_RT60_SEC, min(MAX_RT60_SEC, float(rt60_override))), 2)
    new_scene["room"] = restored_room

    for obj in new_scene.get("objects") or []:
        if not isinstance(obj, dict):
            continue

        # 初回基準状態を保存（既存なら絶対に上書きせず、毎回そこから再計算する）
        if "base_gain_db" not in obj:
            obj["base_gain_db"] = round(_finite_float(obj.get("gain_db"), 0.0), 3)
        if "base_duck_exempt" not in obj:
            obj["base_duck_exempt"] = bool(obj.get("duck_exempt", False))
        if "base_spread" not in obj:
            obj["base_spread"] = round(max(0.0, min(1.0, _finite_float(obj.get("spread"), 0.15))), 3)
        if "base_keyframes" not in obj:
            raw_kfs = obj.get("keyframes") or [{"t": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_m": 2.5}]
            obj["base_keyframes"] = copy.deepcopy(raw_kfs)

        base_gain_db = _finite_float(obj["base_gain_db"], 0.0)
        base_duck_exempt = bool(obj["base_duck_exempt"])
        base_spread = float(obj["base_spread"])
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
            obj["spread"] = base_spread
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

        az_base_at_dir, el_base_at_dir, d_base_at_dir, _ = _interpolate_base_keyframes(base_kfs, base_gain_db, t_dir)

        # 距離指示がある時刻では監督指定距離を補間採用（None の箇所はベース距離をフォールバック）
        has_any_dist = any(k.get("distance_m") is not None for k in dir_kfs)
        if has_any_dist:
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

        # 360° 方位角指示（絶対 azimuth_deg または相対 azimuth_offset_deg）を最短角で補間
        has_az_ctrl = any(
            k.get("azimuth_deg") is not None or k.get("azimuth_offset_deg") is not None
            for k in dir_kfs
        )
        if has_az_ctrl:
            off_az_raw = []
            for i, k in enumerate(dir_kfs):
                if k.get("azimuth_deg") is not None:
                    target_az = float(k["azimuth_deg"]) + _finite_float(k.get("azimuth_offset_deg"), 0.0)
                    diff_az = ((target_az - float(az_base_at_dir[i]) + 180.0) % 360.0) - 180.0
                    off_az_raw.append(diff_az)
                else:
                    off_az_raw.append(_finite_float(k.get("azimuth_offset_deg"), 0.0))
            off_az_unwrapped = np.unwrap(np.radians(np.asarray(off_az_raw, dtype=np.float64)))
            eff_off_az = np.degrees(np.interp(grid_t, t_dir, off_az_unwrapped))
            az_final = ((az_arr + eff_off_az + 180.0) % 360.0) - 180.0
        else:
            az_final = az_arr

        # 仰角指示（絶対 elevation_deg または相対 elevation_offset_deg）を [-90, 90] で補間
        has_el_ctrl = any(
            k.get("elevation_deg") is not None or k.get("elevation_offset_deg") is not None
            for k in dir_kfs
        )
        if has_el_ctrl:
            off_el_raw = []
            for i, k in enumerate(dir_kfs):
                if k.get("elevation_deg") is not None:
                    target_el = float(k["elevation_deg"]) + _finite_float(k.get("elevation_offset_deg"), 0.0)
                    off_el_raw.append(max(-90.0, min(90.0, target_el)) - float(el_base_at_dir[i]))
                else:
                    off_el_raw.append(_finite_float(k.get("elevation_offset_deg"), 0.0))
            eff_off_el = np.interp(grid_t, t_dir, np.asarray(off_el_raw, dtype=np.float64))
            el_final = np.clip(el_arr + eff_off_el, -90.0, 90.0)
        else:
            el_final = el_arr

        spread_vals = [float(k["spread"]) for k in dir_kfs if k.get("spread") is not None]
        obj["spread"] = round(max(0.0, min(1.0, float(np.mean(spread_vals)))), 3) if spread_vals else base_spread

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
            az_r = 0.0 if is_bed else round(float(az_final[i]), 2)
            if not is_bed and abs(az_r + 180.0) < 1e-6:
                az_r = 180.0
            merged_kfs.append({
                "t": t_r,
                "t_sec": t_r,
                "azimuth_deg": az_r,
                "elevation_deg": 0.0 if is_bed else round(float(el_final[i]), 2),
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

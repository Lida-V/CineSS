"""ITU-R BS.2076 / BS.2088 準拠 Dolby Atmos 互換 ADM BWF (.wav) ライター＆バリデーター.

完全フリー＆オープン国際規格として 7.1.4 (12ch) または 7.1.2 (10ch) DirectSpeakers Bed ＋
複数 Dynamic Audio Objects をマルチチャンネル PCM（24-bit int または 32-bit float）で
RIFF/RF64 WAVE コンテナへ格納し、標準 `bext`・`chna`・`axml` チャンク
（`ebuCoreMain` / `audioFormatExtended`）を書き出す（DaVinci Resolve / EAR / Reaper 互換）。
"""

from __future__ import annotations

import datetime as _dt
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .atmos import ATMOS_714_SPEAKERS, Speaker714
from .hrtf import SAMPLE_RATE

# WAVE_FORMAT_EXTENSIBLE サブフォーマット GUID
_KSDATAFORMAT_SUBTYPE_PCM = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

_ADM_NS = "urn:ebu:metadata-schema:ebuCore_2016"
_RF64_THRESHOLD_BYTES = 0xE0000000  # 約 3.5 GiB 超で自動的に RF64 へ切替


def format_adm_timecode(seconds: float) -> str:
    """秒数を ITU-R BS.2076 ADM タイムコード形式 HH:MM:SS.SSSSS へフォーマットする。"""
    sec = max(0.0, float(seconds))
    hours = int(sec // 3600)
    rem = sec - hours * 3600
    minutes = int(rem // 60)
    rem -= minutes * 60
    return f"{hours:02d}:{minutes:02d}:{rem:08.5f}"


def _select_bed_speakers(bed_channels: int) -> tuple[tuple[Speaker714, ...], str, str]:
    """ベッドチャンネル数 (12=7.1.4, 10=7.1.2) に応じたスピーカー定義と PackFormatID を返す。"""
    if bed_channels == 12:
        return ATMOS_714_SPEAKERS, "AP_00010017", "CommonDefinition_7.1.4"
    if bed_channels == 10:
        # 7.1.2: L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf
        return ATMOS_714_SPEAKERS[:10], "AP_00010016", "CommonDefinition_7.1.2"
    raise ValueError(f"ADM ベッドは 10ch (7.1.2) または 12ch (7.1.4) のみ対応です（指定: {bed_channels}ch）")


def _encode_pcm_bytes(audio_2d: np.ndarray, bit_depth: int = 24) -> bytes:
    """float32/float64 (N, C) 音声を 24-bit signed PCM または 32-bit IEEE float バイト列へ変換する。"""
    arr = np.asarray(audio_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"2次元配列 (N, C) が必要です: {arr.shape}")

    if bit_depth == 32:
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if bit_depth == 24:
        clipped = np.clip(arr, -1.0, 1.0)
        scaled = np.round(clipped * 8388607.0).astype("<i4")
        raw_u8 = scaled.view(np.uint8).reshape(-1, 4)
        return np.ascontiguousarray(raw_u8[:, :3]).tobytes()
    raise ValueError(f"bit_depth は 24 または 32 のみ対応です（指定: {bit_depth}）")


def _build_bext_chunk(
    description: str = "CineSS Dolby Atmos ADM BWF Master",
    originator: str = "CineSS Spatial Engine",
    integrated_lufs: float | None = None,
    true_peak_dbtp: float | None = None,
) -> bytes:
    """EBU Tech 3285 v2 準拠 `bext` チャンクペイロードを生成する。"""
    now = _dt.datetime.now()
    desc_b = description.encode("ascii", errors="replace")[:256].ljust(256, b"\x00")
    orig_b = originator.encode("ascii", errors="replace")[:32].ljust(32, b"\x00")
    ref_b = f"CINESS{now.strftime('%Y%m%d%H%M%S')}".encode("ascii")[:32].ljust(32, b"\x00")
    date_b = now.strftime("%Y-%m-%d").encode("ascii")[:10].ljust(10, b"\x00")
    time_b = now.strftime("%H:%M:%S").encode("ascii")[:8].ljust(8, b"\x00")
    umid_b = b"\x00" * 64
    reserved_b = b"\x00" * 180

    def _to_bext_loudness(val: float | None) -> int:
        if val is None or not np.isfinite(val):
            return 0x7FFF
        return max(-32768, min(32767, int(round(val * 100.0))))

    l_val = _to_bext_loudness(integrated_lufs)
    tp_val = _to_bext_loudness(true_peak_dbtp)
    coding_history = b"A=PCM,F=48000,W=24,M=multi,T=CineSS\r\n\x00"

    header = struct.pack(
        "<256s32s32s10s8sQH64shhhhh180s",
        desc_b,
        orig_b,
        ref_b,
        date_b,
        time_b,
        0,  # TimeReference
        2,  # BWF Version 2
        umid_b,
        l_val,
        0x7FFF,  # LoudnessRange
        tp_val,
        0x7FFF,  # MaxMomentaryLoudness
        0x7FFF,  # MaxShortTermLoudness
        reserved_b,
    )
    return header + coding_history


def _build_fmt_chunk(num_channels: int, sample_rate: int, bit_depth: int) -> bytes:
    """厳密な 40 バイト WAVE_FORMAT_EXTENSIBLE `fmt ` ペイロードを生成する。"""
    bytes_per_sample = bit_depth // 8
    block_align = num_channels * bytes_per_sample
    avg_bytes_per_sec = sample_rate * block_align
    subformat = _KSDATAFORMAT_SUBTYPE_IEEE_FLOAT if bit_depth == 32 else _KSDATAFORMAT_SUBTYPE_PCM
    return (
        struct.pack(
            "<HHIIHHHHHI",
            0xFFFE,            # wFormatTag (2)
            num_channels,      # nChannels (2)
            sample_rate,       # nSamplesPerSec (4)
            avg_bytes_per_sec, # nAvgBytesPerSec (4)
            block_align,       # nBlockAlign (2)
            bit_depth,         # wBitsPerSample (2)
            22,                # cbSize (2)
            bit_depth,         # wValidBitsPerSample (2)
            0,                 # dwChannelMask (4) — 合計 24 バイト
            0,                 # 未使用
        )[:24]
        + subformat  # 16 バイト → 合計 40 バイト
    )


def _build_adm_xml_and_chna(
    bed_speakers: tuple[Speaker714, ...],
    bed_pack_id: str,
    bed_pack_name: str,
    dynamic_objects: list[dict[str, Any]],
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    bit_depth: int = 24,
    programme_name: str = "CineSS Dolby Atmos Mix",
    integrated_lufs: float | None = None,
    true_peak_dbtp: float | None = None,
) -> tuple[bytes, bytes]:
    """ITU-R BS.2076 ADM `axml` (UTF-8 XML) と BS.2088 `chna` バイナリを整合性を保って同時生成する。"""
    dur_tc = format_adm_timecode(duration_sec)
    n_bed = len(bed_speakers)
    n_dyn = len(dynamic_objects)
    total_tracks = n_bed + n_dyn

    ET.register_namespace("", _ADM_NS)
    root = ET.Element(f"{{{_ADM_NS}}}ebuCoreMain")
    core_meta = ET.SubElement(root, f"{{{_ADM_NS}}}coreMetadata")
    fmt_el = ET.SubElement(core_meta, f"{{{_ADM_NS}}}format")
    afe = ET.SubElement(fmt_el, f"{{{_ADM_NS}}}audioFormatExtended")

    # 1. audioProgramme
    prog = ET.SubElement(
        afe,
        f"{{{_ADM_NS}}}audioProgramme",
        {
            "audioProgrammeID": "APR_1001",
            "audioProgrammeName": programme_name,
            "start": "00:00:00.00000",
            "end": dur_tc,
        },
    )
    ET.SubElement(prog, f"{{{_ADM_NS}}}audioContentIDRef").text = "ACO_1001"
    if n_dyn > 0:
        ET.SubElement(prog, f"{{{_ADM_NS}}}audioContentIDRef").text = "ACO_1002"
    if integrated_lufs is not None or true_peak_dbtp is not None:
        lm = ET.SubElement(prog, f"{{{_ADM_NS}}}loudnessMetadata")
        if integrated_lufs is not None and np.isfinite(integrated_lufs):
            ET.SubElement(lm, f"{{{_ADM_NS}}}integratedLoudness").text = f"{integrated_lufs:.2f}"
        if true_peak_dbtp is not None and np.isfinite(true_peak_dbtp):
            ET.SubElement(lm, f"{{{_ADM_NS}}}maxTruePeak").text = f"{true_peak_dbtp:.2f}"

    # 2. audioContent (Bed + Dynamic Objects)
    content_bed = ET.SubElement(
        afe,
        f"{{{_ADM_NS}}}audioContent",
        {"audioContentID": "ACO_1001", "audioContentName": f"{bed_pack_name}_Bed"},
    )
    ET.SubElement(content_bed, f"{{{_ADM_NS}}}audioObjectIDRef").text = "AO_1001"

    content_obj: ET.Element | None = None
    if n_dyn > 0:
        content_obj = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioContent",
            {"audioContentID": "ACO_1002", "audioContentName": "Dynamic_Audio_Objects"},
        )

    # chna レコード格納リスト
    chna_records: list[tuple[int, str, str, str]] = []

    # 3. Bed audioObject (AO_1001)
    ao_bed = ET.SubElement(
        afe,
        f"{{{_ADM_NS}}}audioObject",
        {
            "audioObjectID": "AO_1001",
            "audioObjectName": f"Bed_{n_bed}ch",
            "start": "00:00:00.00000",
            "duration": dur_tc,
        },
    )
    ET.SubElement(ao_bed, f"{{{_ADM_NS}}}audioPackFormatIDRef").text = bed_pack_id

    bed_meta_list: list[dict[str, Any]] = []
    for idx, spk in enumerate(bed_speakers, start=1):
        ac_id = f"AC_0001{idx:04X}"
        as_id = f"AS_0001{idx:04X}"
        at_id = f"AT_0001{idx:04X}_01"
        atu_id = f"ATU_{idx:08X}"
        ET.SubElement(ao_bed, f"{{{_ADM_NS}}}audioTrackUIDRef").text = atu_id
        chna_records.append((idx, atu_id, at_id, bed_pack_id))
        bed_meta_list.append(
            {
                "spk": spk,
                "ac_id": ac_id,
                "as_id": as_id,
                "at_id": at_id,
                "atu_id": atu_id,
                "idx": idx,
            }
        )

    # 4. Dynamic Objects の audioObject 定義
    dyn_meta_list: list[dict[str, Any]] = []
    for obj_i, dyn in enumerate(dynamic_objects, start=1):
        track_idx = n_bed + obj_i
        ao_id = f"AO_{1001 + obj_i:04d}"
        ap_id = f"AP_0003{obj_i:04X}"
        ac_id = f"AC_0003{obj_i:04X}"
        as_id = f"AS_0003{obj_i:04X}"
        at_id = f"AT_0003{obj_i:04X}_01"
        atu_id = f"ATU_{track_idx:08X}"

        if content_obj is not None:
            ET.SubElement(content_obj, f"{{{_ADM_NS}}}audioObjectIDRef").text = ao_id

        label = str(dyn.get("label") or dyn.get("id") or f"Object_{obj_i}")
        ao_el = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioObject",
            {
                "audioObjectID": ao_id,
                "audioObjectName": label,
                "start": "00:00:00.00000",
                "duration": dur_tc,
            },
        )
        ET.SubElement(ao_el, f"{{{_ADM_NS}}}audioPackFormatIDRef").text = ap_id
        ET.SubElement(ao_el, f"{{{_ADM_NS}}}audioTrackUIDRef").text = atu_id

        chna_records.append((track_idx, atu_id, at_id, ap_id))
        dyn_meta_list.append(
            {
                "dyn": dyn,
                "label": label,
                "obj_i": obj_i,
                "ap_id": ap_id,
                "ac_id": ac_id,
                "as_id": as_id,
                "at_id": at_id,
                "atu_id": atu_id,
            }
        )

    # 5. audioPackFormat (Bed + Dynamic Objects)
    ap_bed = ET.SubElement(
        afe,
        f"{{{_ADM_NS}}}audioPackFormat",
        {
            "audioPackFormatID": bed_pack_id,
            "audioPackFormatName": bed_pack_name,
            "typeLabel": "0001",
            "typeDefinition": "DirectSpeakers",
        },
    )
    for bm in bed_meta_list:
        ET.SubElement(ap_bed, f"{{{_ADM_NS}}}audioChannelFormatIDRef").text = bm["ac_id"]

    for dm in dyn_meta_list:
        ap_obj = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioPackFormat",
            {
                "audioPackFormatID": dm["ap_id"],
                "audioPackFormatName": dm["label"],
                "typeLabel": "0003",
                "typeDefinition": "Objects",
            },
        )
        ET.SubElement(ap_obj, f"{{{_ADM_NS}}}audioChannelFormatIDRef").text = dm["ac_id"]

    # 6. audioChannelFormat + audioBlockFormat (Bed DirectSpeakers)
    for bm in bed_meta_list:
        spk: Speaker714 = bm["spk"]
        acf = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioChannelFormat",
            {
                "audioChannelFormatID": bm["ac_id"],
                "audioChannelFormatName": f"RoomCentric_{spk.name}",
                "typeLabel": "0001",
                "typeDefinition": "DirectSpeakers",
            },
        )
        if spk.is_lfe:
            ET.SubElement(acf, f"{{{_ADM_NS}}}frequency", {"typeDefinition": "lowPass"}).text = "80"

        abf = ET.SubElement(
            acf,
            f"{{{_ADM_NS}}}audioBlockFormat",
            {
                "audioBlockFormatID": f"AB_0001{bm['idx']:04X}_00000001",
                "rtime": "00:00:00.00000",
                "duration": dur_tc,
            },
        )
        ET.SubElement(abf, f"{{{_ADM_NS}}}speakerLabel").text = spk.itu_label
        ET.SubElement(abf, f"{{{_ADM_NS}}}cartesian").text = "1"
        ET.SubElement(abf, f"{{{_ADM_NS}}}position", {"coordinate": "X"}).text = f"{spk.adm_x:.4f}"
        ET.SubElement(abf, f"{{{_ADM_NS}}}position", {"coordinate": "Y"}).text = f"{spk.adm_y:.4f}"
        ET.SubElement(abf, f"{{{_ADM_NS}}}position", {"coordinate": "Z"}).text = f"{spk.adm_z:.4f}"

    # 7. audioChannelFormat + 時間変化 audioBlockFormat (Dynamic Objects)
    for dm in dyn_meta_list:
        dyn = dm["dyn"]
        obj_i = dm["obj_i"]
        acf = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioChannelFormat",
            {
                "audioChannelFormatID": dm["ac_id"],
                "audioChannelFormatName": dm["label"],
                "typeLabel": "0003",
                "typeDefinition": "Objects",
            },
        )
        kfs: list[dict[str, float]] = list(dyn.get("adm_keyframes") or [{"t": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}])
        kfs.sort(key=lambda item: float(item.get("t", 0.0)))

        for b_idx, kf in enumerate(kfs, start=1):
            t_start = max(0.0, float(kf.get("t", 0.0)))
            if b_idx < len(kfs):
                t_end = max(t_start, float(kfs[b_idx].get("t", duration_sec)))
            else:
                t_end = max(t_start, duration_sec)
            b_dur = max(0.0, t_end - t_start)

            abf = ET.SubElement(
                acf,
                f"{{{_ADM_NS}}}audioBlockFormat",
                {
                    "audioBlockFormatID": f"AB_0003{obj_i:04X}_{b_idx:08X}",
                    "rtime": format_adm_timecode(t_start),
                    "duration": format_adm_timecode(b_dur),
                },
            )
            ET.SubElement(abf, f"{{{_ADM_NS}}}cartesian").text = "1"
            ET.SubElement(abf, f"{{{_ADM_NS}}}position", {"coordinate": "X"}).text = f"{float(kf.get('x', 0.0)):.4f}"
            ET.SubElement(abf, f"{{{_ADM_NS}}}position", {"coordinate": "Y"}).text = f"{float(kf.get('y', 1.0)):.4f}"
            ET.SubElement(abf, f"{{{_ADM_NS}}}position", {"coordinate": "Z"}).text = f"{float(kf.get('z', 0.0)):.4f}"
            ET.SubElement(abf, f"{{{_ADM_NS}}}gain").text = "1.0000"
            ET.SubElement(abf, f"{{{_ADM_NS}}}jumpPosition").text = "0"

    # 8. audioStreamFormat / audioTrackFormat / audioTrackUID
    all_streams = [
        (bm["as_id"], bm["ac_id"], bm["at_id"], bm["atu_id"], bed_pack_id, f"Bed_{bm['spk'].name}")
        for bm in bed_meta_list
    ] + [
        (dm["as_id"], dm["ac_id"], dm["at_id"], dm["atu_id"], dm["ap_id"], dm["label"])
        for dm in dyn_meta_list
    ]

    for as_id, ac_id, at_id, atu_id, ap_id, name in all_streams:
        asf = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioStreamFormat",
            {
                "audioStreamFormatID": as_id,
                "audioStreamFormatName": f"PCM_{name}",
                "formatLabel": "0001",
                "formatDefinition": "PCM",
            },
        )
        ET.SubElement(asf, f"{{{_ADM_NS}}}audioChannelFormatIDRef").text = ac_id
        ET.SubElement(asf, f"{{{_ADM_NS}}}audioTrackFormatIDRef").text = at_id

    for as_id, ac_id, at_id, atu_id, ap_id, name in all_streams:
        atf = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioTrackFormat",
            {
                "audioTrackFormatID": at_id,
                "audioTrackFormatName": f"PCM_{name}",
                "formatLabel": "0001",
                "formatDefinition": "PCM",
            },
        )
        ET.SubElement(atf, f"{{{_ADM_NS}}}audioStreamFormatIDRef").text = as_id

    for as_id, ac_id, at_id, atu_id, ap_id, name in all_streams:
        atu = ET.SubElement(
            afe,
            f"{{{_ADM_NS}}}audioTrackUID",
            {
                "UID": atu_id,
                "sampleRate": str(sample_rate),
                "bitDepth": str(bit_depth),
            },
        )
        ET.SubElement(atu, f"{{{_ADM_NS}}}audioTrackFormatIDRef").text = at_id
        ET.SubElement(atu, f"{{{_ADM_NS}}}audioPackFormatIDRef").text = ap_id

    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover
        pass
    xml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="utf-8")

    # chna バイナリ構築: uint16 numTracks, uint16 numUIDs, [40-byte entries]
    chna_parts = [struct.pack("<HH", total_tracks, len(chna_records))]
    for track_idx, atu_id, at_id, ap_id in chna_records:
        chna_parts.append(
            struct.pack(
                "<H12s14s11sB",
                track_idx,
                atu_id.encode("ascii")[:12].ljust(12, b"\x00"),
                at_id.encode("ascii")[:14].ljust(14, b"\x00"),
                ap_id.encode("ascii")[:11].ljust(11, b"\x00"),
                0,
            )
        )
    chna_bytes = b"".join(chna_parts)
    return xml_bytes, chna_bytes


def _pack_riff_chunk(fourcc: bytes, payload: bytes, force_size_u32: int | None = None) -> bytes:
    """RIFF/RF64 チャンク（8バイトヘッダ＋データ＋奇数長パディング）を組み立てる。"""
    size_u32 = len(payload) if force_size_u32 is None else force_size_u32
    pad = b"\x00" if (len(payload) % 2 == 1) else b""
    return fourcc + struct.pack("<I", size_u32) + payload + pad


def write_adm_bwf(
    out_path: str | Path,
    bed_audio: np.ndarray,
    dynamic_objects: list[dict[str, Any]],
    sample_rate: int = SAMPLE_RATE,
    bit_depth: int = 24,
    programme_name: str = "CineSS Dolby Atmos Mix",
    integrated_lufs: float | None = None,
    true_peak_dbtp: float | None = None,
    force_rf64: bool = False,
) -> Path:
    """7.1.4 (または 7.1.2) Bed ＋ Dynamic Objects を ITU-R BS.2076 準拠 ADM BWF (.wav) へ書き出す。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bed = np.asarray(bed_audio, dtype=np.float32)
    if bed.ndim != 2:
        raise ValueError(f"bed_audio は (N, 12) または (N, 10) が必要です: {bed.shape}")

    n_samples, n_bed = bed.shape
    bed_speakers, bed_pack_id, bed_pack_name = _select_bed_speakers(n_bed)

    # 全チャンネル連結: [Bed (12ch or 10ch), Dynamic Object 1, ..., Dynamic Object M]
    channels_list: list[np.ndarray] = [bed]
    for dyn in dynamic_objects:
        sig = np.asarray(dyn["audio"], dtype=np.float32).reshape(-1)
        if sig.shape[0] < n_samples:
            sig = np.pad(sig, (0, n_samples - sig.shape[0]), mode="constant")
        elif sig.shape[0] > n_samples:
            sig = sig[:n_samples]
        channels_list.append(sig[:, np.newaxis])

    all_audio = np.concatenate(channels_list, axis=1)
    total_channels = int(all_audio.shape[1])
    duration_sec = n_samples / float(sample_rate)

    fmt_payload = _build_fmt_chunk(total_channels, sample_rate, bit_depth)
    bext_payload = _build_bext_chunk(
        description=programme_name,
        integrated_lufs=integrated_lufs,
        true_peak_dbtp=true_peak_dbtp,
    )
    data_payload = _encode_pcm_bytes(all_audio, bit_depth=bit_depth)
    axml_payload, chna_payload = _build_adm_xml_and_chna(
        bed_speakers=bed_speakers,
        bed_pack_id=bed_pack_id,
        bed_pack_name=bed_pack_name,
        dynamic_objects=dynamic_objects,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        programme_name=programme_name,
        integrated_lufs=integrated_lufs,
        true_peak_dbtp=true_peak_dbtp,
    )

    bext_chunk = _pack_riff_chunk(b"bext", bext_payload)
    fmt_chunk = _pack_riff_chunk(b"fmt ", fmt_payload)
    chna_chunk = _pack_riff_chunk(b"chna", chna_payload)
    axml_chunk = _pack_riff_chunk(b"axml", axml_payload)

    data_pad = b"\x00" if (len(data_payload) % 2 == 1) else b""
    estimated_body = 4 + len(bext_chunk) + len(fmt_chunk) + 8 + len(data_payload) + len(data_pad) + len(chna_chunk) + len(axml_chunk)
    use_rf64 = bool(force_rf64 or estimated_body >= _RF64_THRESHOLD_BYTES)

    with open(out, "wb") as f:
        if use_rf64:
            ds64_payload_len = 28
            riff_size_64 = estimated_body + 8 + ds64_payload_len
            ds64_payload = struct.pack("<QQQI", riff_size_64, len(data_payload), n_samples, 0)
            ds64_chunk = _pack_riff_chunk(b"ds64", ds64_payload)
            f.write(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE")
            f.write(ds64_chunk)
            f.write(bext_chunk)
            f.write(fmt_chunk)
            f.write(b"data" + struct.pack("<I", 0xFFFFFFFF))
            f.write(data_payload)
            f.write(data_pad)
            f.write(chna_chunk)
            f.write(axml_chunk)
        else:
            f.write(b"RIFF" + struct.pack("<I", estimated_body) + b"WAVE")
            f.write(bext_chunk)
            f.write(fmt_chunk)
            f.write(b"data" + struct.pack("<I", len(data_payload)))
            f.write(data_payload)
            f.write(data_pad)
            f.write(chna_chunk)
            f.write(axml_chunk)

    return out


# ---------------------------------------------------------------------------
# ADM BWF バリデーター
# ---------------------------------------------------------------------------


def _local_name(tag: str) -> str:
    """XML 名前空間付きタグからローカル要素名を取り出す。"""
    return tag.split("}", 1)[1] if "}" in tag else tag


def validate_adm_bwf(path: str | Path) -> dict[str, Any]:
    """ADM BWF (.wav) ファイルの RIFF/RF64 構造、`fmt `、`data`、`chna`、`axml` 整合性を検証する。"""
    p = Path(path)
    errors: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "errors": errors,
        "path": str(p),
        "container": None,
        "sample_rate": 0,
        "channels": 0,
        "bit_depth": 0,
        "duration_sec": 0.0,
        "num_tracks_chna": 0,
        "num_uids_chna": 0,
        "num_objects": 0,
        "num_channel_formats": 0,
        "num_block_formats": 0,
        "has_bext": False,
        "has_axml": False,
        "has_chna": False,
    }

    if not p.exists():
        errors.append(f"ファイルが存在しません: {p}")
        return result

    file_size = p.stat().st_size
    if file_size < 12:
        errors.append("ファイルサイズが RIFF ヘッダ最小長未満です")
        return result

    fmt_bytes: bytes | None = None
    chna_bytes: bytes | None = None
    axml_bytes: bytes | None = None
    data_size: int = 0
    ds64_data_size: int | None = None

    with open(p, "rb") as f:
        header = f.read(12)
        magic = header[0:4]
        wave_id = header[8:12]
        if magic not in (b"RIFF", b"RF64") or wave_id != b"WAVE":
            errors.append(f"RIFF/RF64 WAVE ヘッダ不正: magic={magic!r}, wave={wave_id!r}")
            return result
        result["container"] = magic.decode("ascii")

        while f.tell() + 8 <= file_size:
            ck_hdr = f.read(8)
            if len(ck_hdr) < 8:
                break
            fourcc = ck_hdr[0:4]
            ck_size = struct.unpack("<I", ck_hdr[4:8])[0]

            if fourcc == b"data" and ck_size == 0xFFFFFFFF and ds64_data_size is not None:
                actual_size = ds64_data_size
            else:
                actual_size = ck_size

            pos_start = f.tell()
            if pos_start + actual_size > file_size:
                errors.append(f"チャンク {fourcc!r} のサイズ ({actual_size}) がファイル終端を超過しています")
                break

            if fourcc == b"ds64" and actual_size >= 28:
                ds64_raw = f.read(28)
                _, ds64_data_size, _, _ = struct.unpack("<QQQI", ds64_raw)
                f.seek(pos_start + actual_size)
            elif fourcc == b"fmt ":
                fmt_bytes = f.read(actual_size)
            elif fourcc == b"bext":
                result["has_bext"] = True
                f.seek(pos_start + actual_size)
            elif fourcc == b"data":
                data_size = actual_size
                f.seek(pos_start + actual_size)
            elif fourcc == b"chna":
                result["has_chna"] = True
                chna_bytes = f.read(actual_size)
            elif fourcc == b"axml":
                result["has_axml"] = True
                axml_bytes = f.read(actual_size)
            else:
                f.seek(pos_start + actual_size)

            # RIFF 2バイト境界パディングをスキップ
            if actual_size % 2 == 1 and f.tell() < file_size:
                f.seek(1, 1)

    # 1. fmt チャンク検証
    if fmt_bytes is None or len(fmt_bytes) < 16:
        errors.append("`fmt ` チャンクが見つからないか不正です")
    else:
        w_fmt, n_ch, sr, _avg, block_align, bits = struct.unpack("<HHIIHH", fmt_bytes[:16])
        result["channels"] = int(n_ch)
        result["sample_rate"] = int(sr)
        result["bit_depth"] = int(bits)
        if w_fmt not in (1, 3, 0xFFFE):
            errors.append(f"未対応の WAVE フォーマットタグ: 0x{w_fmt:04X}")
        if n_ch <= 0:
            errors.append("チャンネル数が 0 以下です")
        if block_align <= 0 or (data_size % block_align != 0):
            errors.append(f"`data` チャンク長 ({data_size}) が blockAlign ({block_align}) の整数倍ではありません")
        elif sr > 0:
            n_frames = data_size // block_align
            result["duration_sec"] = n_frames / float(sr)

    if data_size == 0:
        errors.append("`data` チャンクが空または欠損しています")

    # 2. chna チャンク検証
    chna_uids: set[str] = set()
    if chna_bytes is None or len(chna_bytes) < 4:
        errors.append("`chna` チャンクが見つからないか不正です")
    else:
        num_tracks, num_uids = struct.unpack("<HH", chna_bytes[:4])
        result["num_tracks_chna"] = int(num_tracks)
        result["num_uids_chna"] = int(num_uids)
        expected_len = 4 + 40 * num_uids
        if len(chna_bytes) != expected_len:
            errors.append(f"`chna` サイズ不一致: 期待 {expected_len}, 実際 {len(chna_bytes)}")
        if result["channels"] > 0 and num_tracks != result["channels"]:
            errors.append(f"`chna` numTracks ({num_tracks}) と `fmt` channels ({result['channels']}) が不一致です")
        for i in range(num_uids):
            offset = 4 + i * 40
            if offset + 40 > len(chna_bytes):
                break
            t_idx, uid_b, tref_b, pref_b, _pad = struct.unpack("<H12s14s11sB", chna_bytes[offset : offset + 40])
            uid_s = uid_b.rstrip(b"\x00").decode("ascii", errors="replace")
            tref_s = tref_b.rstrip(b"\x00").decode("ascii", errors="replace")
            pref_s = pref_b.rstrip(b"\x00").decode("ascii", errors="replace")
            if t_idx < 1 or (result["channels"] > 0 and t_idx > result["channels"]):
                errors.append(f"`chna` トラック番号範囲外: {t_idx}")
            if not uid_s or not tref_s or not pref_s:
                errors.append(f"`chna` エントリ {i} の ID が空です")
            chna_uids.add(uid_s)

    # 3. axml チャンク検証
    if axml_bytes is None or not axml_bytes.strip():
        errors.append("`axml` チャンクが見つからないか空です")
    else:
        try:
            xml_text = axml_bytes.rstrip(b"\x00").decode("utf-8")
            xml_root = ET.fromstring(xml_text)
        except Exception as exc:
            errors.append(f"`axml` XML パース失敗: {exc}")
            xml_root = None

        if xml_root is not None:
            if _local_name(xml_root.tag) != "ebuCoreMain":
                errors.append(f"`axml` ルート要素が ebuCoreMain ではありません: {xml_root.tag}")

            counts: dict[str, int] = {}
            xml_uids: set[str] = set()
            for el in xml_root.iter():
                lname = _local_name(el.tag)
                counts[lname] = counts.get(lname, 0) + 1
                if lname == "audioTrackUID":
                    uid_attr = el.attrib.get("UID", "")
                    if uid_attr:
                        xml_uids.add(uid_attr)

            required_tags = (
                "audioFormatExtended",
                "audioProgramme",
                "audioContent",
                "audioObject",
                "audioPackFormat",
                "audioChannelFormat",
                "audioBlockFormat",
                "audioTrackUID",
            )
            for tag in required_tags:
                if counts.get(tag, 0) == 0:
                    errors.append(f"`axml` に必須要素 <{tag}> が存在しません")

            result["num_objects"] = counts.get("audioObject", 0)
            result["num_channel_formats"] = counts.get("audioChannelFormat", 0)
            result["num_block_formats"] = counts.get("audioBlockFormat", 0)

            missing_uids = chna_uids - xml_uids
            if missing_uids:
                errors.append(f"`chna` の TrackUID が `axml` に存在しません: {sorted(missing_uids)}")

    result["valid"] = len(errors) == 0
    return result

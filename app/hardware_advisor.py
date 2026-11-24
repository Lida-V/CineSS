# coding: utf-8
"""PC動作環境診断・スペック別推奨動画長＆使用モデル選定アドバイザー。

実機の CPU / GPU / RAM / FFmpeg / 空きディスク / SoundSeparator共有状況 を計測し、
「有音動画リミックス」「無音動画Foley生成」それぞれについて
✅(ok) / ⚠️(warn) / ✕(error) 判定と現実的な推奨動画長・モデル構成を提示する。
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import APP_ROOT, OUTPUT_DIR, SAMPLE_RATE, find_soundseparator_dir
from app.model_manager import list_models


# ---------------------------------------------------------------------------
# Ambisonics 次数・WAV容量見積りユーティリティ (1〜7次対応)
# ---------------------------------------------------------------------------

_AMBISONICS_FORMAT_ORDER = {
    "ambisonics_foa": 1,
    "foa": 1,
    "ambisonics1": 1,
    "ambisonics_1": 1,
    "ambisonics_hoa": 3,
    "hoa": 3,
    "ambisonics3": 3,
    "ambisonics_3": 3,
    "ambisonics_hoa2": 2,
    "ambisonics_hoa3": 3,
    "ambisonics_hoa4": 4,
    "ambisonics_hoa5": 5,
    "hoa5": 5,
    "ambisonics5": 5,
    "ambisonics_5": 5,
    "ambisonics_hoa6": 6,
    "ambisonics_hoa7": 7,
    "hoa7": 7,
    "ambisonics7": 7,
    "ambisonics_7": 7,
}


def resolve_spatial_spec(
    spatial_format: str = "atmos",
    ambisonics_order: int | None = None,
) -> dict[str, Any]:
    """spatial_format と ambisonics_order から正規化規格・次数・チャンネル数を返す。"""
    raw_fmt = (spatial_format or "atmos").strip().lower()
    if raw_fmt in ("atmos", "dolby_atmos", "7.1.4"):
        return {
            "spatial_format": "atmos",
            "format_label": "Dolby Atmos 互換（ADM BWF + 7.1.4 + 直接バイノーラル・完全フリー）",
            "ambisonics_order": None,
            "channels": 16,  # ADM BWF 概算目安（7.1.4 Bed 12ch + Dynamic Objects）
        }

    order = _AMBISONICS_FORMAT_ORDER.get(raw_fmt)
    if ambisonics_order is not None:
        try:
            order = max(1, min(7, int(ambisonics_order)))
        except (TypeError, ValueError):
            pass
    if order is None:
        order = 3

    canonical_fmt = {
        1: "ambisonics_foa",
        3: "ambisonics_hoa",
        5: "ambisonics_hoa5",
        7: "ambisonics_hoa7",
    }.get(order, "ambisonics_hoa")
    channels = (order + 1) ** 2
    tier_tag = "FOA" if order == 1 else "HOA"
    return {
        "spatial_format": canonical_fmt,
        "format_label": f"Ambisonics {order}次 {tier_tag}（{channels}ch AmbiX）",
        "ambisonics_order": order,
        "channels": channels,
    }


def estimate_spatial_wav_size(
    duration_sec: float,
    spatial_format: str = "atmos",
    ambisonics_order: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """48kHz 32bit float 書き出し時のサラウンドマスターWAV推定容量を計算する。"""
    spec = resolve_spatial_spec(spatial_format, ambisonics_order)
    dur = max(0.0, float(duration_sec))
    channels = int(spec["channels"])
    bytes_per_sec = int(sample_rate) * 4 * channels
    total_bytes = int(round(dur * bytes_per_sec))
    mb = round(total_bytes / (1024**2), 2)
    gb = round(total_bytes / (1024**3), 3)
    mb_per_min = round((bytes_per_sec * 60.0) / (1024**2), 1)
    return {
        **spec,
        "estimated_bytes": total_bytes,
        "estimated_mb": mb,
        "estimated_gb": gb,
        "mb_per_min": mb_per_min,
        "requires_rf64": total_bytes > int(3.5 * (1024**3)),
    }


# ---------------------------------------------------------------------------
# 低レベル ハードウェア実測プローブ群（ユニットテストで個別にモック可能）
# ---------------------------------------------------------------------------


def _run_sysctl(key: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        proc = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            val = proc.stdout.strip()
            return val if val else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _read_windows_cpu_name() -> str | None:
    if os.name != "nt" or platform.system() != "Windows":
        return None
    try:
        import winreg  # noqa: PLC0415

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if isinstance(name, str) and name.strip():
                return re.sub(r"\s+", " ", name.strip())
    except OSError:
        return None
    return None


def _read_linux_cpu_name() -> str | None:
    if platform.system() != "Linux":
        return None
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if not cpuinfo.is_file():
            return None
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                name = line.split(":", 1)[1].strip()
                if name:
                    return re.sub(r"\s+", " ", name)
    except OSError:
        return None
    return None


def _probe_cpu() -> dict[str, Any]:
    sys_name = platform.system()
    name = (
        _read_windows_cpu_name()
        or (_run_sysctl("machdep.cpu.brand_string") if sys_name == "Darwin" else None)
        or (_run_sysctl("hw.model") if sys_name == "Darwin" else None)
        or _read_linux_cpu_name()
        or platform.processor()
        or platform.machine()
        or "不明なCPU"
    )
    cores = 0
    threads = 0
    try:
        import psutil  # noqa: PLC0415

        cores = int(psutil.cpu_count(logical=False) or 0)
        threads = int(psutil.cpu_count(logical=True) or 0)
    except ImportError:
        pass
    if sys_name == "Darwin":
        if cores <= 0:
            raw_cores = _run_sysctl("hw.physicalcpu")
            if raw_cores and raw_cores.isdigit():
                cores = int(raw_cores)
        if threads <= 0:
            raw_threads = _run_sysctl("hw.logicalcpu")
            if raw_threads and raw_threads.isdigit():
                threads = int(raw_threads)
    if threads <= 0:
        threads = int(os.cpu_count() or 1)
    if cores <= 0:
        cores = max(1, threads // 2) if threads > 1 else 1
    return {"name": name, "cores": cores, "threads": threads}


def _probe_nvidia_smi() -> dict[str, Any] | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 4:
            return None
        total_mib = float(parts[1])
        free_mib = float(parts[2])
        return {
            "name": parts[0],
            "vram_total_gb": round(total_mib / 1024.0, 2),
            "vram_free_gb": round(free_mib / 1024.0, 2),
            "driver": parts[3],
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _probe_torch_cuda() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415

        available = bool(torch.cuda.is_available())
        if not available:
            return {"torch_installed": True, "cuda": False}
        name = str(torch.cuda.get_device_name(0))
        free_b, total_b = torch.cuda.mem_get_info(0)
        return {
            "torch_installed": True,
            "cuda": True,
            "name": name,
            "vram_total_gb": round(total_b / (1024**3), 2),
            "vram_free_gb": round(free_b / (1024**3), 2),
        }
    except Exception:
        return {"torch_installed": False, "cuda": False}


def _probe_torch_mps() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415

        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        available = bool(mps_backend is not None and mps_backend.is_available())
        return {"torch_installed": True, "mps": available}
    except Exception:
        return {"torch_installed": False, "mps": False}


def _probe_non_nvidia_gpu() -> str | None:
    """Windows / macOS / Linux で AMD / Intel 等の非 NVIDIA GPU 名を検出する。"""
    sys_name = platform.system()
    ignore_tokens = ("nvidia", "microsoft basic", "virtual", "remote", "parsec", "vmware", "citrix")

    if sys_name == "Windows":
        try:
            import winreg  # noqa: PLC0415

            base_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_key) as root_key:
                for idx in range(16):
                    sub_name = f"{idx:04d}"
                    try:
                        with winreg.OpenKey(root_key, sub_name) as sub_key:
                            desc, _ = winreg.QueryValueEx(sub_key, "DriverDesc")
                            if isinstance(desc, str):
                                cleaned = re.sub(r"\s+", " ", desc.strip())
                                low = cleaned.lower()
                                if cleaned and not any(tok in low for tok in ignore_tokens):
                                    return cleaned
                    except OSError:
                        continue
        except OSError:
            pass
        return None

    if sys_name == "Darwin":
        try:
            proc = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                for line in proc.stdout.splitlines():
                    if "Chipset Model:" in line:
                        chip = line.split(":", 1)[1].strip()
                        if chip and "nvidia" not in chip.lower():
                            return chip
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    if sys_name == "Linux":
        lspci = shutil.which("lspci")
        if lspci:
            try:
                proc = subprocess.run([lspci], capture_output=True, text=True, timeout=3, check=False)
                if proc.returncode == 0 and proc.stdout:
                    for line in proc.stdout.splitlines():
                        low = line.lower()
                        if any(k in low for k in ("vga compatible controller", "3d controller", "display controller")):
                            if "nvidia" not in low and ":" in line:
                                return line.split(":", 2)[-1].strip()
            except (OSError, subprocess.SubprocessError):
                pass
    return None


def _probe_gpu() -> dict[str, Any]:
    smi_info = _probe_nvidia_smi()
    torch_info = _probe_torch_cuda()

    if smi_info:
        # PyTorchが入っていてCUDA無効なら明示的に検知する
        if torch_info.get("torch_installed"):
            cuda_ok = bool(torch_info.get("cuda"))
        else:
            cuda_ok = False
        return {
            "name": smi_info["name"],
            "cuda": cuda_ok,
            "mps": False,
            "backend": "cuda" if cuda_ok else "cpu",
            "unified_memory": False,
            "non_nvidia_gpu": False,
            "vram_total_gb": smi_info["vram_total_gb"],
            "vram_free_gb": smi_info["vram_free_gb"],
            "driver": smi_info["driver"],
            "nvidia_detected": True,
        }

    if torch_info.get("cuda"):
        return {
            "name": torch_info.get("name", "NVIDIA CUDA GPU"),
            "cuda": True,
            "mps": False,
            "backend": "cuda",
            "unified_memory": False,
            "non_nvidia_gpu": False,
            "vram_total_gb": float(torch_info.get("vram_total_gb", 0.0)),
            "vram_free_gb": float(torch_info.get("vram_free_gb", 0.0)),
            "driver": None,
            "nvidia_detected": True,
        }

    sys_name = platform.system()
    mps_info = _probe_torch_mps()
    cpu_brand = _run_sysctl("machdep.cpu.brand_string") if sys_name == "Darwin" else None
    is_apple_silicon = sys_name == "Darwin" and (
        platform.machine().lower() in ("arm64", "aarch64")
        or (cpu_brand is not None and "apple" in cpu_brand.lower())
    )

    if mps_info.get("mps"):
        chip_name = cpu_brand or "Apple Silicon"
        mem_info = _probe_memory()
        um_gb = float(mem_info.get("total_gb", 0.0))
        return {
            "name": f"{chip_name} (MPS ユニファイドメモリ)",
            "cuda": False,
            "mps": True,
            "backend": "mps",
            "unified_memory": True,
            "unified_memory_gb": um_gb,
            "non_nvidia_gpu": False,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "driver": "Metal (MPS)",
            "nvidia_detected": False,
        }

    if is_apple_silicon:
        chip_name = cpu_brand or "Apple Silicon"
        mem_info = _probe_memory()
        um_gb = float(mem_info.get("total_gb", 0.0))
        return {
            "name": f"{chip_name} (MPS未有効・CPUモード)",
            "cuda": False,
            "mps": False,
            "backend": "cpu",
            "unified_memory": True,
            "unified_memory_gb": um_gb,
            "non_nvidia_gpu": True,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "driver": None,
            "nvidia_detected": False,
        }

    other_gpu = _probe_non_nvidia_gpu()
    if other_gpu:
        return {
            "name": f"{other_gpu} (非NVIDIA・CPU動作)",
            "cuda": False,
            "mps": False,
            "backend": "cpu",
            "unified_memory": False,
            "non_nvidia_gpu": True,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "driver": None,
            "nvidia_detected": False,
        }

    return {
        "name": "未検出（CPUモード）",
        "cuda": False,
        "mps": False,
        "backend": "cpu",
        "unified_memory": False,
        "non_nvidia_gpu": False,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "driver": None,
        "nvidia_detected": False,
    }


def _probe_memory_macos_sysctl() -> dict[str, float] | None:
    if platform.system() != "Darwin":
        return None
    raw_mem = _run_sysctl("hw.memsize")
    if not raw_mem:
        return None
    try:
        total_bytes = int(raw_mem.strip())
    except ValueError:
        return None
    if total_bytes <= 0:
        return None
    total_gb = round(total_bytes / (1024**3), 2)
    avail_gb = round(total_gb * 0.5, 2)
    try:
        proc = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3, check=False)
        if proc.returncode == 0 and proc.stdout:
            page_size = 4096
            m_page = re.search(r"page size of (\d+) bytes", proc.stdout)
            if m_page:
                page_size = int(m_page.group(1))
            free_pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                m_cnt = re.search(rf"{re.escape(key)}:\s+(\d+)", proc.stdout)
                if m_cnt:
                    free_pages += int(m_cnt.group(1))
            if free_pages > 0:
                avail_gb = round(min(total_bytes, free_pages * page_size) / (1024**3), 2)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {"total_gb": total_gb, "available_gb": avail_gb}


def _probe_memory_windows_ctypes() -> dict[str, float] | None:
    if os.name != "nt" or platform.system() != "Windows":
        return None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return {
            "total_gb": round(status.ullTotalPhys / (1024**3), 2),
            "available_gb": round(status.ullAvailPhys / (1024**3), 2),
        }
    except Exception:
        return None


def _probe_memory() -> dict[str, float]:
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
        }
    except ImportError:
        pass
    mac_mem = _probe_memory_macos_sysctl()
    if mac_mem is not None:
        return mac_mem
    win_mem = _probe_memory_windows_ctypes()
    if win_mem is not None:
        return win_mem
    return {"total_gb": 0.0, "available_gb": 0.0}


def _probe_ffmpeg() -> dict[str, Any]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return {"available": False, "version": None}
    try:
        proc = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        match = re.search(r"ffmpeg version\s+([^\s]+)", first, re.IGNORECASE)
        version = match.group(1) if match else (first[:48] or "検出済み")
        return {"available": True, "version": version}
    except (OSError, subprocess.SubprocessError):
        return {"available": True, "version": "検出済み"}


def _probe_disk() -> dict[str, float]:
    target = OUTPUT_DIR if OUTPUT_DIR.exists() else APP_ROOT
    try:
        usage = shutil.disk_usage(target)
        return {"free_gb": round(usage.free / (1024**3), 2)}
    except OSError:
        return {"free_gb": 0.0}


def _probe_soundseparator() -> dict[str, Any]:
    ss_dir = find_soundseparator_dir()
    if ss_dir is None:
        return {"detected": False, "path": None, "shared_models": [], "shared_model_ids": []}
    models = list_models(soundsep_dir=ss_dir)
    shared = [m for m in models if m["status"] == "shared_soundsep"]
    return {
        "detected": True,
        "path": str(ss_dir),
        "shared_models": [m["label"] for m in shared],
        "shared_model_ids": [m["id"] for m in shared],
    }


def probe_hardware() -> dict[str, Any]:
    """実PCのハードウェア・ツール生情報をまとめて計測する。"""
    return {
        "cpu": _probe_cpu(),
        "gpu": _probe_gpu(),
        "memory": _probe_memory(),
        "ffmpeg": _probe_ffmpeg(),
        "disk": _probe_disk(),
        "soundseparator": _probe_soundseparator(),
    }


# ---------------------------------------------------------------------------
# 純粋判定ロジック（スペック → 診断カード + 推奨プラン）
# ---------------------------------------------------------------------------


def _eval_cpu(raw: dict[str, Any]) -> dict[str, Any]:
    cores = int(raw.get("cores", 0))
    threads = int(raw.get("threads", 0))
    name = str(raw.get("name", "不明なCPU"))
    if cores >= 8 or threads >= 12:
        status = "ok"
        note = f"{cores}コア / {threads}スレッド。映像解析と空間HRTFバイノーラル処理を快適に実行できます。"
    elif cores >= 4 or threads >= 6:
        status = "warn"
        note = f"{cores}コア / {threads}スレッド。標準動作しますが、長尺動画の空間レンダリングはやや時間がかかります。"
    else:
        status = "error"
        note = f"{cores}コア / {threads}スレッド。コア数が少ないため、1〜3分程度の短尺クリップでの利用を推奨します。"
    return {"name": name, "cores": cores, "threads": threads, "status": status, "note": note}


def _eval_gpu(raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name", "未検出"))
    cuda = bool(raw.get("cuda", False))
    mps = bool(raw.get("mps", False))
    unified_memory = bool(raw.get("unified_memory", mps))
    unified_memory_gb = round(float(raw.get("unified_memory_gb", 0.0) or 0.0), 2)
    non_nvidia_gpu = bool(raw.get("non_nvidia_gpu", False))
    backend = str(raw.get("backend") or ("cuda" if cuda else ("mps" if mps else "cpu")))
    total = round(float(raw.get("vram_total_gb", 0.0)), 2)
    free = round(float(raw.get("vram_free_gb", 0.0)), 2)
    driver = raw.get("driver")
    nvidia_detected = bool(raw.get("nvidia_detected", False))

    if cuda and total >= 15.5:
        if free > 0 and free < 4.0:
            status = "warn"
            note = (
                f"高性能GPU（総VRAM {total:.1f} GB）ですが現在の空きVRAMが {free:.1f} GB です。"
                "他のGPUアプリを閉じると最高品質で安定動作します。"
            )
        else:
            status = "ok"
            note = (
                f"CUDA対応GPUを検出（空き {free:.1f} GB / 総VRAM {total:.1f} GB）。"
                "有音の高精度分離・話者分離・無音Foley生成まで快適です。"
            )
    elif cuda and total >= 11.5:
        status = "ok"
        note = (
            f"CUDA対応GPUを検出（空き {free:.1f} GB / 総VRAM {total:.1f} GB）。"
            "有音リミックスは高品質対応、無音Foleyはオフロード併用で動作します。"
        )
    elif cuda and total >= 6.0:
        status = "warn"
        note = (
            f"ミドル帯CUDA GPU（空き {free:.1f} GB / 総VRAM {total:.1f} GB）。"
            "有音リミックス（標準品質）は動作しますが、無音Foleyはオフロード必須かつ短尺推奨です。"
        )
    elif mps:
        um_label = f"{unified_memory_gb:.1f} GB" if unified_memory_gb > 0 else "計測中"
        if unified_memory_gb >= 23.0:
            status = "ok"
            note = (
                f"Apple Silicon MPS（ユニファイドメモリ {um_label}）を検出。"
                "有音AI分離・話者分離をMPSで高速実行できます（PYTORCH_ENABLE_MPS_FALLBACK=1 推奨、"
                "無音Foley生成時は省メモリ設定・短尺推奨）。"
            )
        elif unified_memory_gb >= 15.0:
            status = "warn"
            note = (
                f"Apple Silicon MPS（ユニファイドメモリ {um_label}）を検出。"
                "有音リミックスはMPSで動作しますが、OSとメモリ共有のため無音Foley等は省メモリ・短尺推奨です"
                "（PYTORCH_ENABLE_MPS_FALLBACK=1 推奨）。"
            )
        else:
            status = "warn"
            note = (
                f"Apple Silicon MPS（ユニファイドメモリ {um_label}）を検出しましたが容量が小さめです。"
                "PYTORCH_ENABLE_MPS_FALLBACK=1 を有効にし、1〜3分程度の短尺動画でご利用ください。"
            )
    elif nvidia_detected and not cuda:
        status = "error"
        note = (
            f"NVIDIA GPU（{name}）は検出されましたが、PyTorchのCUDAが無効です。"
            "setup.bat / setup.sh で CUDA版 PyTorch を導入してください（audio-separatorは導入禁止）。"
        )
    elif non_nvidia_gpu:
        status = "warn"
        note = (
            f"非NVIDIA GPU（{name}）を検出しました。PyTorch CUDA / MPS 非対応のため CPU モード"
            "（軽量DSP分離＋物理HRTF空間Mix）で動作します。無音AI Foley本番生成は非対応です。"
        )
    else:
        status = "error"
        note = (
            "CUDA / MPS 対応GPUが検出されませんでした（CPU動作）。"
            "軽量DSP分離＋空間Mixは動作しますが、高精度AI分離や無音Foley生成には対応GPUが必要です。"
        )

    return {
        "name": name,
        "cuda": cuda,
        "mps": mps,
        "backend": backend,
        "unified_memory": unified_memory,
        "unified_memory_gb": unified_memory_gb,
        "non_nvidia_gpu": non_nvidia_gpu,
        "vram_total_gb": total,
        "vram_free_gb": free,
        "driver": driver,
        "status": status,
        "note": note,
    }


def _eval_memory(raw: dict[str, Any], *, unified_memory: bool = False) -> dict[str, Any]:
    total = round(float(raw.get("total_gb", 0.0)), 2)
    avail = round(float(raw.get("available_gb", 0.0)), 2)
    um_suffix = "（ユニファイドメモリとしてGPUとも共有）" if unified_memory else ""
    if total >= 30.0:
        status = "ok"
        note = f"総容量 {total:.1f} GB（空き {avail:.1f} GB）{um_suffix}。長尺映像・複数AIモデル切替・空間多ch出力を余裕で処理できます。"
    elif total >= 15.0:
        status = "warn"
        note = f"総容量 {total:.1f} GB（空き {avail:.1f} GB）{um_suffix}。標準的な中尺動画まで動作しますが、Foley生成時は他アプリ終了を推奨します。"
    else:
        status = "error"
        note = f"総容量 {total:.1f} GB（空き {avail:.1f} GB）{um_suffix}。メモリ不足のリスクが高いため、短尺（1〜3分以内）やプレビュー区間処理を使ってください。"
    return {"total_gb": total, "available_gb": avail, "status": status, "note": note}


def _eval_ffmpeg(raw: dict[str, Any]) -> dict[str, Any]:
    avail = bool(raw.get("available", False))
    version = raw.get("version")
    if avail:
        status = "ok"
        note = f"FFmpeg 利用可能（{version or '検出済み'}）。映像解析・48kHz音声抽出・バイノーラルMP4出力に対応します。"
    else:
        status = "error"
        note = "FFmpeg が見つかりません。動画の読み込みと書き出しに必須のため、FFmpeg を導入して PATH に追加してください。"
    return {"available": avail, "version": version, "status": status, "note": note}


def _eval_disk(raw: dict[str, Any]) -> dict[str, Any]:
    free_gb = round(float(raw.get("free_gb", 0.0)), 2)
    if free_gb >= 20.0:
        status = "ok"
        note = f"空き容量 {free_gb:.1f} GB。サラウンドWAV・ADM BWF・AIモデルの保存に十分です。"
    elif free_gb >= 5.0:
        status = "warn"
        note = f"空き容量 {free_gb:.1f} GB。動画変換は可能ですが、大型Foleyモデルの追加前に容量確保を推奨します。"
    else:
        status = "error"
        note = f"空き容量 {free_gb:.1f} GB。容量が不足しているため、出力前に最低5 GB以上（推奨20 GB以上）を確保してください。"
    return {"free_gb": free_gb, "status": status, "note": note}


def _build_recommendations(
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    memory: dict[str, Any],
    ffmpeg: dict[str, Any],
    soundsep: dict[str, Any],
) -> dict[str, Any]:
    cuda = gpu["cuda"]
    mps = bool(gpu.get("mps", False))
    non_nvidia_gpu = bool(gpu.get("non_nvidia_gpu", False))
    vram = gpu["vram_total_gb"]
    ram = memory["total_gb"]
    cores = cpu["cores"]
    has_ss = bool(soundsep.get("detected"))

    if cuda and vram >= 15.5 and ram >= 30.0 and cores >= 8:
        tier = "S"
        tier_label = "Tier S：スタジオ最上位（有音・無音フル機能快適）"
        voiced = {
            "supported": True,
            "status": "ok",
            "comfortable_minutes": 30,
            "max_recommended_minutes": 90,
            "quality_grade": "S",
            "quality_label": "Sランク：映画スタジオ品質（ステレオ3分離＋残響除去＋最大8話者分離＋深度3D配置）",
            "models": {
                "separator": "BandIt Plus (ステレオDME) + MelBand Dereverb",
                "diarization": "Nemotron-3-Diarization (最大8話者)",
                "overlap": "SpeechBrain SepFormer / 定位・帯域ソフト分離",
                "vision": "OpenCV顔・口元・動き追跡 + Depth Anything V2 Small",
            },
            "reason": (
                f"VRAM {vram:.1f} GB・RAM {ram:.0f} GB の余裕があり、高精度なDME分離・残響除去・話者別声分け・3D空間Mixを快適に通貫処理できます。"
                + (" SoundSeparator の共有モデルもそのまま活用可能です。" if has_ss else "")
            ),
            "estimated_speed_factor": 0.45,
        }
        silent = {
            "supported": True,
            "status": "ok",
            "comfortable_seconds": 60,
            "max_recommended_seconds": 180,
            "quality_grade": "S",
            "quality_label": "Sランク：HunyuanVideo-Foley XXL（48kHz高精細Foley＋空間配置）",
            "models": {
                "foley": "HunyuanVideo-Foley XXL",
                "offload": False,
                "vision": "OpenCV動き重心・カット解析 + Depth Anything V2 Small",
            },
            "reason": (
                f"VRAM {vram:.1f} GB により HunyuanVideo-Foley XXL のローカル生成に対応します。"
                "60秒前後まで快適、最大3分程度までチャンク分割クロスフェード結合で処理できます。"
            ),
            "estimated_speed_factor": 1.6,
        }
    elif cuda and vram >= 11.5 and ram >= 15.0:
        tier = "A"
        tier_label = "Tier A：ハイスペック（高品質有音＆短〜中尺Foley対応）"
        voiced = {
            "supported": True,
            "status": "ok",
            "comfortable_minutes": 15,
            "max_recommended_minutes": 45,
            "quality_grade": "A",
            "quality_label": "Aランク：高品質（BandIt Plus分離＋話者分離＋空間Mix）",
            "models": {
                "separator": "BandIt Plus (ステレオDME)",
                "diarization": "Nemotron-3-Diarization",
                "overlap": "定位・帯域ソフト分離（任意でSepFormer）",
                "vision": "OpenCV顔・口元・動き追跡",
            },
            "reason": (
                f"VRAM {vram:.1f} GB・RAM {ram:.0f} GB のため、有音動画の分離と話者別空間配置を高品質に実行できます。"
                "15分前後まで快適、最長45分目安です。"
            ),
            "estimated_speed_factor": 0.75,
        }
        silent = {
            "supported": True,
            "status": "warn",
            "comfortable_seconds": 30,
            "max_recommended_seconds": 90,
            "quality_grade": "A",
            "quality_label": "Aランク：HunyuanVideo-Foley XL（オフロード併用＋空間配置）",
            "models": {
                "foley": "HunyuanVideo-Foley XL (CPUオフロード併用)",
                "offload": True,
                "vision": "OpenCV動き重心・カット解析",
            },
            "reason": (
                f"VRAM {vram:.1f} GB 環境では HunyuanVideo-Foley XL + オフロード設定が安定します。"
                "30秒前後を目安に、長尺はプレビュー区間指定でお試しください。"
            ),
            "estimated_speed_factor": 3.2,
        }
    elif cuda and vram >= 6.0 and ram >= 12.0:
        tier = "B"
        tier_label = "Tier B：ミドルスペック（標準品質・短〜中尺向け）"
        voiced = {
            "supported": True,
            "status": "warn",
            "comfortable_minutes": 5,
            "max_recommended_minutes": 15,
            "quality_grade": "B",
            "quality_label": "Bランク：標準品質（省VRAM分離＋音響クラスタ話者分け＋空間Mix）",
            "models": {
                "separator": "BandIt Plus (標準) または DBS-TIGER-DnR (省VRAM)",
                "diarization": "VAD＋MFCC/ピッチ音響クラスタリング",
                "overlap": "定位・帯域ソフト分離",
                "vision": "OpenCV顔・口元・動き追跡",
            },
            "reason": (
                f"VRAM {vram:.1f} GB・RAM {ram:.0f} GB のため、5分以内の有音動画であれば標準品質で安定処理できます。"
                "15分を超える動画はプレビュー区間指定を推奨します。"
            ),
            "estimated_speed_factor": 1.35,
        }
        silent = {
            "supported": True,
            "status": "warn",
            "comfortable_seconds": 15,
            "max_recommended_seconds": 30,
            "quality_grade": "B",
            "quality_label": "Bランク：HunyuanVideo-Foley XL（オフロード必須・短尺限定）",
            "models": {
                "foley": "HunyuanVideo-Foley XL (--enable_offload必須)",
                "offload": True,
                "vision": "OpenCV動き重心・カット解析",
            },
            "reason": (
                f"VRAM {vram:.1f} GB では無音Foley生成の余裕が少ないため、オフロード有効かつ15〜30秒以内の短尺クリップに限定することを推奨します。"
                "VRAM不足時は他アプリを閉じて再実行してください。"
            ),
            "estimated_speed_factor": 5.5,
        }
    elif mps and ram >= 23.0 and cores >= 8:
        tier = "A"
        tier_label = "Tier A (Apple Silicon MPS)：ハイスペック（大容量ユニファイドメモリ）"
        voiced = {
            "supported": True,
            "status": "ok",
            "comfortable_minutes": 15,
            "max_recommended_minutes": 45,
            "quality_grade": "A",
            "quality_label": "Aランク (MPS)：高品質（BandIt Plus分離＋話者分離＋空間Mix）",
            "models": {
                "separator": "BandIt Plus (ステレオDME)",
                "diarization": "Nemotron-3-Diarization",
                "overlap": "定位・帯域ソフト分離（任意でSepFormer）",
                "vision": "OpenCV顔・口元・動き追跡 + Depth Anything V2 Small",
            },
            "reason": (
                f"Apple Silicon MPS（ユニファイドメモリ {ram:.0f} GB）により、有音動画の分離と話者別空間配置を高品質に実行できます。"
                "未対応演算フォールバックのため PYTORCH_ENABLE_MPS_FALLBACK=1（run.sh 既定有効）を維持してください。"
            ),
            "estimated_speed_factor": 0.95,
        }
        max_foley_sec = 45 if ram >= 30.0 else 30
        voiced_note_extra = "他アプリを閉じた状態で" if ram < 30.0 else ""
        silent = {
            "supported": True,
            "status": "warn",
            "comfortable_seconds": 15,
            "max_recommended_seconds": max_foley_sec,
            "quality_grade": "A",
            "quality_label": "Aランク (MPS)：HunyuanVideo-Foley XL（省メモリ・オフロード推奨）",
            "models": {
                "foley": "HunyuanVideo-Foley XL (MPS省メモリ・オフロード併用)",
                "offload": True,
                "vision": "OpenCV動き重心・カット解析",
            },
            "reason": (
                f"ユニファイドメモリ {ram:.0f} GB の MPS 環境ではメモリを OS と共有するため、"
                f"Foley 生成は{voiced_note_extra}省メモリ設定（オフロード有効）・PYTORCH_ENABLE_MPS_FALLBACK=1 で 15〜{max_foley_sec} 秒以内の短尺を推奨します。"
            ),
            "estimated_speed_factor": 3.8,
        }
    elif mps and ram >= 15.0:
        tier = "B"
        tier_label = "Tier B (Apple Silicon MPS)：標準スペック（MPS省メモリ運用）"
        voiced = {
            "supported": True,
            "status": "warn",
            "comfortable_minutes": 5,
            "max_recommended_minutes": 15,
            "quality_grade": "B",
            "quality_label": "Bランク (MPS)：標準品質（省メモリ分離＋音響クラスタ話者分け＋空間Mix）",
            "models": {
                "separator": "BandIt Plus (標準) または DBS-TIGER-DnR (省メモリ)",
                "diarization": "VAD＋MFCC/ピッチ音響クラスタリング",
                "overlap": "定位・帯域ソフト分離",
                "vision": "OpenCV顔・口元・動き追跡",
            },
            "reason": (
                f"Apple Silicon MPS（ユニファイドメモリ {ram:.0f} GB）により、5〜15分までの有音動画を安定処理できます。"
                "一部演算のCPUフォールバック用に PYTORCH_ENABLE_MPS_FALLBACK=1（run.sh で自動設定）が必要です。"
            ),
            "estimated_speed_factor": 1.45,
        }
        silent = {
            "supported": True,
            "status": "warn",
            "comfortable_seconds": 10,
            "max_recommended_seconds": 20,
            "quality_grade": "B",
            "quality_label": "Bランク (MPS)：HunyuanVideo-Foley XL（省メモリ注意・極短尺限定）",
            "models": {
                "foley": "HunyuanVideo-Foley XL (MPS省メモリ・オフロード必須)",
                "offload": True,
                "vision": "OpenCV動き重心・カット解析",
            },
            "reason": (
                f"ユニファイドメモリ {ram:.0f} GB では OS とメモリを共有するため、Foley 生成は省メモリ注意です。"
                "他アプリを終了し、PYTORCH_ENABLE_MPS_FALLBACK=1・オフロード有効の 10〜20 秒クリップでお試しください。"
            ),
            "estimated_speed_factor": 6.0,
        }
    else:
        tier = "C"
        if mps:
            tier_label = "Tier C (MPS 省メモリ)：短尺リミックス向け"
            voiced_reason = (
                f"Apple Silicon MPS 環境ですがユニファイドメモリが {ram:.0f} GB のため、安全のため短尺（1〜3分）または軽量分離を推奨します"
                "（PYTORCH_ENABLE_MPS_FALLBACK=1 有効推奨）。"
            )
            silent_reason = (
                f"無音動画からの AI Foley 生成（HunyuanVideo-Foley）はユニファイドメモリ 16 GB 以上（推奨 24 GB 以上）が必要です。"
                f"現在の {ram:.0f} GB 環境ではテスト用のプレビュー合成フォールバックのみ利用できます。"
            )
        elif non_nvidia_gpu:
            tier_label = "Tier C (非NVIDIA GPU)：CPU軽量モード（短尺リミックス向け）"
            voiced_reason = (
                "非NVIDIA GPU（AMD / Intel 等）は PyTorch CUDA / MPS 非対応のため、CPU 完結のローカル DSP 3分離と物理 HRTF 空間エンジンで動作します。"
                "1〜3分程度の短尺動画で立体音響化を体験できます。"
            )
            silent_reason = (
                "無音動画からの AI Foley 生成（HunyuanVideo-Foley）は NVIDIA CUDA GPU（または 16 GB 以上の Apple Silicon MPS）が必要です。"
                "非NVIDIA GPU 環境ではテスト用のプレビュー合成フォールバックのみ動作します。"
            )
        else:
            tier_label = "Tier C：CPU軽量モード（短尺リミックス・プレビュー向け）"
            voiced_reason = (
                "CUDA対応GPUが利用できない（またはVRAM/RAMが不足している）ため、CPU完結のDSP分離と物理HRTF空間エンジンで動作します。"
                "AI分離より分離精度は控えめですが、1〜3分の短尺動画で立体音響化を体験できます。"
            )
            silent_reason = (
                "無音動画からのAI Foley生成（HunyuanVideo-Foley）はCUDA対応NVIDIA GPU（推奨12GB以上・最低目安8GB＋オフロード）が必要です。"
                "現在のPCではテスト用のプレビュー合成フォールバックのみ動作します。"
            )

        voiced = {
            "supported": True,
            "status": "warn",
            "comfortable_minutes": 1,
            "max_recommended_minutes": 3,
            "quality_grade": "C",
            "quality_label": "Cランク：CPU軽量品質（ローカルDSP 3分離＋音響話者分け＋物理HRTF空間Mix）",
            "models": {
                "separator": "ローカルDSP 3分離（Mid/Side＋帯域・HPSSフォールバック）",
                "diarization": "VAD＋MFCC/ピッチ音響クラスタリング",
                "overlap": "定位・帯域ソフト分離",
                "vision": "OpenCV顔・動き追跡（軽量サンプリング）",
            },
            "reason": voiced_reason,
            "estimated_speed_factor": 3.5,
        }
        silent = {
            "supported": False,
            "status": "error",
            "comfortable_seconds": 0,
            "max_recommended_seconds": 0,
            "quality_grade": "C",
            "quality_label": "非対応：CUDA GPU または 16GB+ Apple Silicon MPS 必須",
            "models": {
                "foley": "本番生成は非対応（テスト用モーション合成フォールバックのみ利用可）",
                "offload": True,
                "vision": "OpenCV動き重心解析",
            },
            "reason": silent_reason,
            "estimated_speed_factor": 0.0,
        }

    # Ambisonics 推奨上限次数（1|3|5|7）を RAM・CPU・Tier から判定
    # RAM 16GB以下や低スペックでは3次以下推奨（5次/7次は警告）、高スペック（32GB以上＆8コア以上）では7次快適
    if ram >= 30.0 and cores >= 8 and tier in ("S", "A"):
        max_ambi_order = 7
    elif ram >= 24.0 and cores >= 6 and tier in ("S", "A"):
        max_ambi_order = 5
    elif ram >= 15.0 and cores >= 4 and tier in ("S", "A", "B"):
        max_ambi_order = 3
    else:
        max_ambi_order = 1

    ambi_orders: dict[str, dict[str, Any]] = {}
    for order in (1, 3, 5, 7):
        ch = (order + 1) ** 2
        mb_per_min = round((SAMPLE_RATE * 4 * ch * 60.0) / (1024**2), 1)
        is_ok = order <= max_ambi_order
        if is_ok:
            order_status = "ok"
            order_note = f"{ch}ch（約 {mb_per_min:.0f} MB/分）。このPCで快適に出力できます。"
        elif order == 3 and max_ambi_order == 1:
            order_status = "warn"
            order_note = f"{ch}ch（約 {mb_per_min:.0f} MB/分）。短尺動画なら動作しますが、1次 FOA の方が軽快です。"
        else:
            order_status = "warn"
            order_note = (
                f"{ch}ch（約 {mb_per_min:.0f} MB/分）。RAM {ram:.0f} GB 環境ではメモリ・ディスク負荷が高いため"
                f" {max_ambi_order}次以下または短尺クリップを推奨します。"
            )
        ambi_orders[str(order)] = {
            "order": order,
            "channels": ch,
            "mb_per_min": mb_per_min,
            "status": order_status,
            "label": f"{order}次 {'FOA' if order == 1 else 'HOA'} ({ch}ch AmbiX)",
            "note": order_note,
        }

    # FFmpeg未導入時は両モードとも実行不可の警告を反映
    if not ffmpeg["available"]:
        voiced["supported"] = False
        voiced["status"] = "error"
        voiced["reason"] = "FFmpeg が見つからないため動画の読み込み・書き出しができません。先に FFmpeg を導入してください。"
        silent["supported"] = False
        silent["status"] = "error"
        silent["reason"] = "FFmpeg が見つからないため動画処理ができません。先に FFmpeg を導入してください。"

    return {
        "tier": tier,
        "tier_label": tier_label,
        "max_recommended_ambisonics_order": max_ambi_order,
        "ambisonics": {
            "max_recommended_order": max_ambi_order,
            "default_order": 3 if max_ambi_order >= 3 else 1,
            "orders": ambi_orders,
        },
        "atmos_note": (
            "Dolby Atmos 互換出力は完全フリー＆オープン国際規格（ITU-R BS.2076 ADM BWF + 7.1.4 Bed + Dynamic Objects + 内蔵HRTF直接バイノーラル）です。"
            "DaVinci Resolve / EAR / Reaper 等で扱えます（民生TV向け .ec3 E-AC-3 JOC 配信用のみ別途対応ツールが必要）。"
        ),
        "voiced": voiced,
        "silent": silent,
    }


def assess_system(raw_hw: dict[str, Any]) -> dict[str, Any]:
    """計測済みハードウェア生情報から診断結果辞書を構築する（純粋関数）。"""
    cpu = _eval_cpu(raw_hw.get("cpu") or {})
    mem_raw = dict(raw_hw.get("memory") or {})
    gpu_raw = dict(raw_hw.get("gpu") or {})
    if gpu_raw.get("mps") and "unified_memory" not in gpu_raw:
        gpu_raw["unified_memory"] = True
    if gpu_raw.get("unified_memory") and not gpu_raw.get("unified_memory_gb"):
        gpu_raw["unified_memory_gb"] = float(mem_raw.get("total_gb") or 0.0)
    gpu = _eval_gpu(gpu_raw)
    memory = _eval_memory(mem_raw, unified_memory=bool(gpu.get("unified_memory")))
    ffmpeg = _eval_ffmpeg(raw_hw.get("ffmpeg") or {})
    disk = _eval_disk(raw_hw.get("disk") or {})
    ss_raw = raw_hw.get("soundseparator") or {}
    soundseparator = {
        "detected": bool(ss_raw.get("detected", False)),
        "path": ss_raw.get("path"),
        "shared_models": list(ss_raw.get("shared_models") or []),
    }
    recommendations = _build_recommendations(cpu, gpu, memory, ffmpeg, soundseparator)

    statuses = [cpu["status"], gpu["status"], memory["status"], ffmpeg["status"], disk["status"]]
    if "error" in statuses:
        overall_status = "error"
    elif "warn" in statuses:
        overall_status = "warn"
    else:
        overall_status = "ok"

    return {
        "overall_status": overall_status,
        "cpu": cpu,
        "gpu": gpu,
        "memory": memory,
        "ffmpeg": ffmpeg,
        "disk": disk,
        "soundseparator": soundseparator,
        "recommendations": recommendations,
    }


def check_system() -> dict[str, Any]:
    """実PCの動作環境を診断し、UI・API・エージェント共通の診断レポートを返す。"""
    return assess_system(probe_hardware())


def _format_duration_ja(seconds: float) -> str:
    secs = max(1, int(round(seconds)))
    if secs < 60:
        return f"約 {secs}秒"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        return f"約 {mins}分{rem}秒" if rem else f"約 {mins}分"
    hours, rem_mins = divmod(mins, 60)
    return f"約 {hours}時間{rem_mins}分"


def evaluate_video_plan(
    duration_sec: float,
    has_audio: bool,
    mode: str = "auto",
    quality_preset: str = "auto",
    spatial_format: str = "atmos",
    ambisonics_order: int | None = None,
    system_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """対象動画の条件とPCスペックを照合し、適合判定・警告・選定モデル・推定処理時間・WAV容量を返す。"""
    sys_info = system_info if system_info is not None else check_system()
    recs = sys_info["recommendations"]
    tier = recs["tier"]
    dur = max(0.0, float(duration_sec))
    max_ambi_order = int(recs.get("max_recommended_ambisonics_order", 3))

    # モード正規化
    norm_mode = (mode or "auto").strip().lower()
    if norm_mode == "auto":
        effective_mode = "voiced" if has_audio else "silent_foley"
    elif norm_mode in ("voiced", "voiced_remix"):
        effective_mode = "voiced"
    elif norm_mode in ("silent", "silent_foley", "foley"):
        effective_mode = "silent_foley"
    else:
        effective_mode = "voiced" if has_audio else "silent_foley"

    # サラウンド規格・次数・WAV容量見積り
    wav_est = estimate_spatial_wav_size(dur, spatial_format=spatial_format, ambisonics_order=ambisonics_order)
    fmt = wav_est["spatial_format"]
    resolved_order = wav_est["ambisonics_order"]

    # サラウンド規格の負荷係数
    if fmt == "atmos":
        format_mult = 1.15
    else:
        order_mults = {1: 0.95, 2: 1.05, 3: 1.2, 4: 1.4, 5: 1.65, 6: 1.95, 7: 2.3}
        format_mult = order_mults.get(int(resolved_order or 3), 1.2)

    # 品質プリセットの負荷係数
    preset = (quality_preset or "auto").strip().lower()
    preset_mult = {"auto": 1.0, "high": 1.45, "standard": 1.0, "fast": 0.65}.get(preset, 1.0)

    warnings: list[str] = []
    status = "ok"
    recommended_preview_seconds: int | None = None
    can_run = True

    if not sys_info["ffmpeg"]["available"]:
        status = "error"
        can_run = False
        warnings.append("FFmpeg が見つかりません。動画の解析・書き出しを行うには FFmpeg の導入が必要です。")

    free_disk_gb = float(sys_info["disk"]["free_gb"])
    if sys_info["disk"]["status"] == "error":
        status = "error"
        warnings.append(f"ディスク空き容量が不足しています（{free_disk_gb:.1f} GB）。出力に失敗する恐れがあります。")
    elif sys_info["disk"]["status"] == "warn":
        if status == "ok":
            status = "warn"
        warnings.append(f"ディスク空き容量が少なめです（{free_disk_gb:.1f} GB）。")

    # Ambisonics 高次（特に5次/7次）選択時の推奨次数・メモリ・ディスク容量チェック
    if resolved_order is not None:
        est_mb = wav_est["estimated_mb"]
        est_gb = wav_est["estimated_gb"]
        ram_total = float(sys_info["memory"]["total_gb"])
        if resolved_order > max_ambi_order:
            if status == "ok":
                status = "warn"
            warnings.append(
                f"Ambisonics {resolved_order}次（{wav_est['channels']}ch・推定WAV約 {est_mb:.0f} MB）は"
                f"このPCの推奨上限次数（{max_ambi_order}次）を超えています。"
                f"RAM {ram_total:.0f} GB 環境ではメモリ負荷や書き出し時間が増大するため、{max_ambi_order}次以下または短尺クリップを推奨します。"
            )
        # 5次/7次の大容量ディスク残量チェック（作業一時領域含め1.8倍の余裕を確認）
        needed_gb = max(est_gb * 1.8, est_gb + 0.5)
        if est_gb > 0 and free_disk_gb < needed_gb:
            status = "error"
            warnings.append(
                f"Ambisonics {resolved_order}次 WAV の推定容量（約 {est_gb:.2f} GB）に対し、"
                f"ディスク空き容量（{free_disk_gb:.1f} GB）が不足する恐れがあります。次数を下げるか空き容量を確保してください。"
            )
        elif wav_est["requires_rf64"]:
            if status == "ok":
                status = "warn"
            warnings.append(
                f"Ambisonics {resolved_order}次 WAV の推定容量が 3.5 GB を超える（約 {est_gb:.2f} GB）ため、"
                "自動的に RF64（64bit拡張WAV）形式で書き出されます。"
            )

    if effective_mode == "voiced":
        profile = recs["voiced"]
        selected_models = dict(profile["models"])
        quality_grade = profile["quality_grade"]
        quality_label = profile["quality_label"]
        quality_explanation = profile["reason"]

        if not has_audio:
            if status == "ok":
                status = "warn"
            warnings.append("この動画は無音（または極小音量）と判定されました。有音リミックスより「無音Foley生成」モードを推奨します。")

        comfort_sec = float(profile["comfortable_minutes"]) * 60.0
        max_sec = float(profile["max_recommended_minutes"]) * 60.0

        if max_sec > 0 and dur > max_sec:
            status = "error" if tier in ("B", "C") else "warn"
            recommended_preview_seconds = int(min(comfort_sec, 60.0 if tier == "C" else 120.0))
            warnings.append(
                f"動画長（{dur / 60.0:.1f}分）がこのPCの推奨上限（{profile['max_recommended_minutes']}分）を超えています。"
                f"まずは「お試し区間（{recommended_preview_seconds}秒）」での確認をおすすめします。"
            )
        elif comfort_sec > 0 and dur > comfort_sec:
            if status == "ok":
                status = "warn"
            warnings.append(
                f"動画長（{dur / 60.0:.1f}分）は快適目安（{profile['comfortable_minutes']}分）を超えています。処理時間が長くなります。"
            )

        if preset == "high" and tier in ("B", "C"):
            if status == "ok":
                status = "warn"
            warnings.append(
                f"現在のPC（Tier {tier}）で「高品質」を指定するとVRAM/RAM不足や長時間の待機が発生する恐れがあります。「PC自動最適化」または「標準/軽量」を推奨します。"
            )
        if profile["status"] == "warn" and status == "ok":
            status = "warn"

        speed_factor = float(profile["estimated_speed_factor"]) or 1.5
        estimated_seconds = round(max(4.0, dur * speed_factor * format_mult * preset_mult + 6.0), 1)
    else:
        profile = recs["silent"]
        selected_models = dict(profile["models"])
        quality_grade = profile["quality_grade"]
        quality_label = profile["quality_label"]
        quality_explanation = profile["reason"]

        if not profile["supported"]:
            status = "error"
            can_run = False
            recommended_preview_seconds = 10
            warnings.append(
                "このPCにはCUDA対応GPU（8GB以上目安）または大容量Apple Silicon MPS（16GB以上目安）がないため、HunyuanVideo-Foley の本番生成は非対応です。"
                "動作確認のみ行う場合は「テスト用フォールバック許可」を有効にし、10秒以内の短尺でお試しください。"
            )
            estimated_seconds = round(max(5.0, min(dur, 30.0) * 1.2 + 4.0), 1)
        else:
            comfort_sec = float(profile["comfortable_seconds"])
            max_sec = float(profile["max_recommended_seconds"])
            if max_sec > 0 and dur > max_sec:
                status = "error" if tier == "B" else "warn"
                recommended_preview_seconds = int(max(10, comfort_sec))
                warnings.append(
                    f"動画長（{dur:.0f}秒）が無音Foley生成の推奨上限（{int(max_sec)}秒）を超えています。"
                    f"お試し区間（{recommended_preview_seconds}秒）を指定して生成することを強く推奨します。"
                )
            elif comfort_sec > 0 and dur > comfort_sec:
                if status == "ok":
                    status = "warn"
                warnings.append(
                    f"動画長（{dur:.0f}秒）は快適目安（{int(comfort_sec)}秒）を超えています。チャンク分割生成のため時間がかかります。"
                )

            if profile["status"] == "warn" and status == "ok":
                status = "warn"

            speed_factor = float(profile["estimated_speed_factor"]) or 3.0
            estimated_seconds = round(max(8.0, dur * speed_factor * format_mult * preset_mult + 10.0), 1)

    return {
        "status": status,
        "can_run": can_run,
        "tier": tier,
        "effective_mode": effective_mode,
        "spatial_format": fmt,
        "spatial_format_label": wav_est["format_label"],
        "ambisonics_order": resolved_order,
        "max_recommended_ambisonics_order": max_ambi_order,
        "spatial_channels": wav_est["channels"],
        "estimated_master_wav_mb": wav_est["estimated_mb"],
        "estimated_master_wav_gb": wav_est["estimated_gb"],
        "requires_rf64": wav_est["requires_rf64"],
        "quality_preset": preset,
        "warnings": warnings,
        "selected_models": selected_models,
        "quality_grade": quality_grade,
        "quality_label": quality_label,
        "quality_explanation": quality_explanation,
        "recommended_preview_seconds": recommended_preview_seconds,
        "estimated_seconds": estimated_seconds,
        "estimated_time_label": _format_duration_ja(estimated_seconds),
    }

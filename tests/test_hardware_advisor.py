# coding: utf-8
"""hardware_advisor / config / model_manager / MCP プロトコルのユニットテスト。

高スペック（RTX 4090 24GB / 64GB+）・中スペック（VRAM 8GB / 16GB RAM）・
低スペック（GPUなし / 8GB RAM）の3ケースをモック検証し、
✅(ok) / ⚠️(warn) / ✕(error) 判定・推奨動画長・モデル選定が妥当に変化することを確認する。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from app import config, hardware_advisor, model_manager
from tools import ciness_agent, ciness_mcp


def _high_spec_hw() -> dict:
    return {
        "cpu": {"name": "AMD Ryzen 9 9950X 16-Core Processor", "cores": 16, "threads": 32},
        "gpu": {
            "name": "NVIDIA GeForce RTX 4090",
            "cuda": True,
            "vram_total_gb": 24.0,
            "vram_free_gb": 20.5,
            "driver": "566.03",
            "nvidia_detected": True,
        },
        "memory": {"total_gb": 64.0, "available_gb": 48.0},
        "ffmpeg": {"available": True, "version": "7.1"},
        "disk": {"free_gb": 350.0},
        "soundseparator": {
            "detected": True,
            "path": r"D:\dummy\SoundtSeparator",
            "shared_models": ["BandIt Plus", "Nemotron-3-Diarization"],
        },
    }


def _mid_spec_hw() -> dict:
    return {
        "cpu": {"name": "Intel Core i5-12400", "cores": 6, "threads": 12},
        "gpu": {
            "name": "NVIDIA GeForce RTX 3060 Ti",
            "cuda": True,
            "vram_total_gb": 8.0,
            "vram_free_gb": 6.8,
            "driver": "560.94",
            "nvidia_detected": True,
        },
        "memory": {"total_gb": 16.0, "available_gb": 9.5},
        "ffmpeg": {"available": True, "version": "6.1"},
        "disk": {"free_gb": 45.0},
        "soundseparator": {"detected": False, "path": None, "shared_models": []},
    }


def _low_spec_hw() -> dict:
    return {
        "cpu": {"name": "Intel Core i3-10100U", "cores": 2, "threads": 4},
        "gpu": {
            "name": "未検出（CPUモード）",
            "cuda": False,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "driver": None,
            "nvidia_detected": False,
        },
        "memory": {"total_gb": 8.0, "available_gb": 3.2},
        "ffmpeg": {"available": True, "version": "6.0"},
        "disk": {"free_gb": 12.0},
        "soundseparator": {"detected": False, "path": None, "shared_models": []},
    }


class TestHardwareAdvisorProfiles(unittest.TestCase):
    """高・中・低スペック3ケースの診断結果と動画プラン判定を検証する。"""

    def test_high_spec_rtx4090_64gb(self) -> None:
        with mock.patch.object(hardware_advisor, "probe_hardware", return_value=_high_spec_hw()):
            report = hardware_advisor.check_system()

        self.assertEqual(report["overall_status"], "ok")
        self.assertEqual(report["cpu"]["status"], "ok")
        self.assertEqual(report["gpu"]["status"], "ok")
        self.assertEqual(report["memory"]["status"], "ok")
        self.assertEqual(report["ffmpeg"]["status"], "ok")
        self.assertEqual(report["disk"]["status"], "ok")
        self.assertTrue(report["soundseparator"]["detected"])
        self.assertIn("BandIt Plus", report["soundseparator"]["shared_models"])

        recs = report["recommendations"]
        self.assertEqual(recs["tier"], "S")
        self.assertTrue(recs["voiced"]["supported"])
        self.assertEqual(recs["voiced"]["status"], "ok")
        self.assertEqual(recs["voiced"]["quality_grade"], "S")
        self.assertGreaterEqual(recs["voiced"]["comfortable_minutes"], 20)
        self.assertGreaterEqual(recs["voiced"]["max_recommended_minutes"], 60)
        self.assertIn("BandIt Plus", recs["voiced"]["models"]["separator"])
        self.assertIn("Nemotron-3-Diarization", recs["voiced"]["models"]["diarization"])

        self.assertTrue(recs["silent"]["supported"])
        self.assertEqual(recs["silent"]["status"], "ok")
        self.assertEqual(recs["silent"]["quality_grade"], "S")
        self.assertIn("XXL", recs["silent"]["models"]["foley"])
        self.assertFalse(recs["silent"]["models"]["offload"])
        self.assertGreaterEqual(recs["silent"]["comfortable_seconds"], 45)

        # 5分有音動画 → ok
        plan_voiced = hardware_advisor.evaluate_video_plan(
            duration_sec=300.0,
            has_audio=True,
            mode="auto",
            quality_preset="auto",
            spatial_format="atmos",
            system_info=report,
        )
        self.assertEqual(plan_voiced["status"], "ok")
        self.assertTrue(plan_voiced["can_run"])
        self.assertEqual(plan_voiced["effective_mode"], "voiced")
        self.assertEqual(plan_voiced["warnings"], [])
        self.assertGreater(plan_voiced["estimated_seconds"], 0)

        # 30秒無音動画 → ok
        plan_silent = hardware_advisor.evaluate_video_plan(
            duration_sec=30.0,
            has_audio=False,
            mode="auto",
            quality_preset="auto",
            spatial_format="ambisonics_hoa",
            system_info=report,
        )
        self.assertEqual(plan_silent["status"], "ok")
        self.assertTrue(plan_silent["can_run"])
        self.assertEqual(plan_silent["effective_mode"], "silent_foley")

    def test_mid_spec_vram8gb_ram16gb(self) -> None:
        with mock.patch.object(hardware_advisor, "probe_hardware", return_value=_mid_spec_hw()):
            report = hardware_advisor.check_system()

        self.assertEqual(report["overall_status"], "warn")
        self.assertEqual(report["gpu"]["status"], "warn")
        self.assertEqual(report["memory"]["status"], "warn")
        self.assertEqual(report["ffmpeg"]["status"], "ok")

        recs = report["recommendations"]
        self.assertEqual(recs["tier"], "B")
        self.assertTrue(recs["voiced"]["supported"])
        self.assertEqual(recs["voiced"]["status"], "warn")
        self.assertEqual(recs["voiced"]["quality_grade"], "B")
        self.assertLessEqual(recs["voiced"]["max_recommended_minutes"], 20)

        self.assertTrue(recs["silent"]["supported"])
        self.assertEqual(recs["silent"]["status"], "warn")
        self.assertTrue(recs["silent"]["models"]["offload"])
        self.assertIn("XL", recs["silent"]["models"]["foley"])
        self.assertLessEqual(recs["silent"]["max_recommended_seconds"], 30)

        # 推奨上限(15分)を超える25分有音動画 → error + お試し区間提案
        plan_long = hardware_advisor.evaluate_video_plan(
            duration_sec=25 * 60.0,
            has_audio=True,
            mode="voiced",
            quality_preset="high",
            spatial_format="atmos",
            system_info=report,
        )
        self.assertEqual(plan_long["status"], "error")
        self.assertTrue(any("推奨上限" in w for w in plan_long["warnings"]))
        self.assertTrue(any("高品質" in w for w in plan_long["warnings"]))
        self.assertIsNotNone(plan_long["recommended_preview_seconds"])

        # 無音60秒（上限30秒超）→ error + プレビュー提案
        plan_silent_long = hardware_advisor.evaluate_video_plan(
            duration_sec=60.0,
            has_audio=False,
            mode="silent_foley",
            quality_preset="auto",
            spatial_format="ambisonics_foa",
            system_info=report,
        )
        self.assertEqual(plan_silent_long["status"], "error")
        self.assertIsNotNone(plan_silent_long["recommended_preview_seconds"])

    def test_low_spec_no_gpu_ram8gb(self) -> None:
        with mock.patch.object(hardware_advisor, "probe_hardware", return_value=_low_spec_hw()):
            report = hardware_advisor.check_system()

        self.assertEqual(report["overall_status"], "error")
        self.assertEqual(report["cpu"]["status"], "error")
        self.assertEqual(report["gpu"]["status"], "error")
        self.assertEqual(report["memory"]["status"], "error")
        self.assertEqual(report["disk"]["status"], "warn")

        recs = report["recommendations"]
        self.assertEqual(recs["tier"], "C")
        self.assertTrue(recs["voiced"]["supported"])
        self.assertEqual(recs["voiced"]["quality_grade"], "C")
        self.assertLessEqual(recs["voiced"]["max_recommended_minutes"], 3)
        self.assertIn("DSP", recs["voiced"]["models"]["separator"])

        # 無音AI Foleyは非対応(error)
        self.assertFalse(recs["silent"]["supported"])
        self.assertEqual(recs["silent"]["status"], "error")
        self.assertEqual(recs["silent"]["max_recommended_seconds"], 0)

        # 無音プラン評価 → error / can_run=False
        plan_silent = hardware_advisor.evaluate_video_plan(
            duration_sec=15.0,
            has_audio=False,
            mode="auto",
            quality_preset="auto",
            spatial_format="atmos",
            system_info=report,
        )
        self.assertEqual(plan_silent["status"], "error")
        self.assertFalse(plan_silent["can_run"])
        self.assertTrue(len(plan_silent["warnings"]) >= 1)

        # 短い1分有音動画 → 実行可能（warn）
        plan_short_voiced = hardware_advisor.evaluate_video_plan(
            duration_sec=45.0,
            has_audio=True,
            mode="voiced",
            quality_preset="fast",
            spatial_format="ambisonics_foa",
            system_info=report,
        )
        self.assertTrue(plan_short_voiced["can_run"])
        self.assertEqual(plan_short_voiced["status"], "warn")

    def test_profile_monotonic_scaling(self) -> None:
        """高→中→低スペックの順に推奨動画長と速度目安が単調に厳しくなることを確認する。"""
        high = hardware_advisor.assess_system(_high_spec_hw())["recommendations"]
        mid = hardware_advisor.assess_system(_mid_spec_hw())["recommendations"]
        low = hardware_advisor.assess_system(_low_spec_hw())["recommendations"]

        self.assertGreater(high["voiced"]["max_recommended_minutes"], mid["voiced"]["max_recommended_minutes"])
        self.assertGreater(mid["voiced"]["max_recommended_minutes"], low["voiced"]["max_recommended_minutes"])
        self.assertGreater(high["silent"]["max_recommended_seconds"], mid["silent"]["max_recommended_seconds"])
        self.assertGreater(mid["silent"]["max_recommended_seconds"], low["silent"]["max_recommended_seconds"])
        self.assertLess(high["voiced"]["estimated_speed_factor"], mid["voiced"]["estimated_speed_factor"])
        self.assertLess(mid["voiced"]["estimated_speed_factor"], low["voiced"]["estimated_speed_factor"])

    def test_missing_ffmpeg_blocks_both_modes(self) -> None:
        hw = _high_spec_hw()
        hw["ffmpeg"] = {"available": False, "version": None}
        report = hardware_advisor.assess_system(hw)
        self.assertEqual(report["overall_status"], "error")
        self.assertEqual(report["ffmpeg"]["status"], "error")
        self.assertFalse(report["recommendations"]["voiced"]["supported"])
        self.assertFalse(report["recommendations"]["silent"]["supported"])

        plan = hardware_advisor.evaluate_video_plan(30.0, True, system_info=report)
        self.assertEqual(plan["status"], "error")
        self.assertFalse(plan["can_run"])

    def test_nvidia_present_but_cpu_torch_detected(self) -> None:
        hw = _high_spec_hw()
        hw["gpu"]["cuda"] = False
        report = hardware_advisor.assess_system(hw)
        self.assertEqual(report["gpu"]["status"], "error")
        self.assertIn("CUDA", report["gpu"]["note"])

    def test_ambisonics_order_recommendations_and_wav_estimates(self) -> None:
        """Ambisonics 1〜7次推奨上限・WAV容量見積り・RAM/ディスク警告を検証する。"""
        high = hardware_advisor.assess_system(_high_spec_hw())
        mid = hardware_advisor.assess_system(_mid_spec_hw())
        low = hardware_advisor.assess_system(_low_spec_hw())

        # 高スペック: 7次(64ch)推奨上限 & 7次評価 ok
        self.assertEqual(high["recommendations"]["max_recommended_ambisonics_order"], 7)
        self.assertEqual(high["recommendations"]["ambisonics"]["orders"]["7"]["status"], "ok")
        plan_high_7 = hardware_advisor.evaluate_video_plan(
            duration_sec=60.0,
            has_audio=True,
            mode="voiced",
            quality_preset="auto",
            spatial_format="ambisonics_hoa7",
            system_info=high,
        )
        self.assertEqual(plan_high_7["status"], "ok")
        self.assertEqual(plan_high_7["ambisonics_order"], 7)
        self.assertEqual(plan_high_7["spatial_channels"], 64)
        self.assertAlmostEqual(plan_high_7["estimated_master_wav_mb"], 703.12, delta=5.0)
        self.assertFalse(plan_high_7["requires_rf64"])

        # 高スペックでも 6分 7次(64ch) は 3.5GB超 → RF64 自動切替通知
        plan_high_rf64 = hardware_advisor.evaluate_video_plan(
            duration_sec=360.0,
            has_audio=True,
            mode="voiced",
            quality_preset="auto",
            spatial_format="ambisonics_hoa7",
            system_info=high,
        )
        self.assertTrue(plan_high_rf64["requires_rf64"])
        self.assertTrue(any("RF64" in w for w in plan_high_rf64["warnings"]))

        # 中スペック (RAM 16GB): 推奨上限は 3次、5次/7次は warn 警告
        self.assertEqual(mid["recommendations"]["max_recommended_ambisonics_order"], 3)
        self.assertEqual(mid["recommendations"]["ambisonics"]["orders"]["3"]["status"], "ok")
        self.assertEqual(mid["recommendations"]["ambisonics"]["orders"]["5"]["status"], "warn")
        self.assertEqual(mid["recommendations"]["ambisonics"]["orders"]["7"]["status"], "warn")
        plan_mid_7 = hardware_advisor.evaluate_video_plan(
            duration_sec=60.0,
            has_audio=True,
            mode="voiced",
            quality_preset="standard",
            spatial_format="ambisonics_hoa7",
            system_info=mid,
        )
        self.assertEqual(plan_mid_7["status"], "warn")
        self.assertTrue(any("推奨上限次数" in w for w in plan_mid_7["warnings"]))

        # 低スペック (RAM 8GB / GPUなし): 推奨上限は 1次、7次は warn
        self.assertEqual(low["recommendations"]["max_recommended_ambisonics_order"], 1)
        plan_low_5 = hardware_advisor.evaluate_video_plan(
            duration_sec=30.0,
            has_audio=True,
            mode="voiced",
            quality_preset="fast",
            spatial_format="ambisonics_hoa",
            ambisonics_order=5,
            system_info=low,
        )
        self.assertEqual(plan_low_5["ambisonics_order"], 5)
        self.assertEqual(plan_low_5["spatial_channels"], 36)
        self.assertEqual(plan_low_5["status"], "warn")
        self.assertTrue(any("推奨上限次数" in w for w in plan_low_5["warnings"]))

        # ディスク残量不足時の 7次 WAV 容量エラー検知
        tight_disk_hw = _high_spec_hw()
        tight_disk_hw["disk"]["free_gb"] = 6.0
        tight_disk_report = hardware_advisor.assess_system(tight_disk_hw)
        plan_disk_tight = hardware_advisor.evaluate_video_plan(
            duration_sec=600.0,  # 7次 64ch 10分 ≈ 6.87 GB > 空き 6.0 GB
            has_audio=True,
            mode="voiced",
            quality_preset="auto",
            spatial_format="ambisonics_hoa7",
            system_info=tight_disk_report,
        )
        self.assertEqual(plan_disk_tight["status"], "error")
        self.assertTrue(any("ディスク空き容量" in w for w in plan_disk_tight["warnings"]))


class TestRealSystemAndInfrastructure(unittest.TestCase):
    """実PCの check_system() 契約・モデル管理・MCPプロトコルの基本検証。"""

    def test_real_check_system_schema(self) -> None:
        report = hardware_advisor.check_system()
        for key in ("overall_status", "cpu", "gpu", "memory", "ffmpeg", "disk", "soundseparator", "recommendations"):
            self.assertIn(key, report)
        self.assertIn(report["overall_status"], ("ok", "warn", "error"))
        for key in ("name", "cores", "threads", "status", "note"):
            self.assertIn(key, report["cpu"])
        for key in (
            "name",
            "cuda",
            "mps",
            "backend",
            "unified_memory",
            "non_nvidia_gpu",
            "vram_total_gb",
            "vram_free_gb",
            "driver",
            "status",
            "note",
        ):
            self.assertIn(key, report["gpu"])
        for key in ("total_gb", "available_gb", "status", "note"):
            self.assertIn(key, report["memory"])
        for key in ("available", "version", "status", "note"):
            self.assertIn(key, report["ffmpeg"])
        for key in ("free_gb", "status", "note"):
            self.assertIn(key, report["disk"])
        for key in ("detected", "path", "shared_models"):
            self.assertIn(key, report["soundseparator"])

        recs = report["recommendations"]
        self.assertIn(recs["tier"], ("S", "A", "B", "C"))
        for key in ("supported", "status", "max_recommended_minutes", "comfortable_minutes", "quality_grade", "quality_label", "models", "reason", "estimated_speed_factor"):
            self.assertIn(key, recs["voiced"])
        for key in ("supported", "status", "max_recommended_seconds", "comfortable_seconds", "quality_grade", "quality_label", "models", "reason", "estimated_speed_factor"):
            self.assertIn(key, recs["silent"])

    def test_model_manager_status_resolution(self) -> None:
        config.ensure_directories()
        with tempfile.TemporaryDirectory(dir=str(config.OUTPUT_DIR)) as tmp:
            root = Path(tmp)
            local_models = root / "local_models"
            ss_dir = root / "SoundtSeparator"
            (local_models / "hrtf").mkdir(parents=True)
            (local_models / "hrtf" / "mit_kemar_normal_pinna.sofa").write_bytes(b"SOFA_TEST_DATA")
            (ss_dir / "models").mkdir(parents=True)
            (ss_dir / "models" / "model_bandit_plus_dnr_sdr_11.47.chpt").write_bytes(b"BANDIT_WEIGHTS")

            entries = {m["id"]: m for m in model_manager.list_models(models_root=local_models, soundsep_dir=ss_dir)}
            self.assertEqual(entries["mit_kemar_sofa"]["status"], "installed")
            self.assertIsNotNone(entries["mit_kemar_sofa"]["path"])
            self.assertEqual(entries["bandit"]["status"], "shared_soundsep")
            self.assertIsNotNone(entries["bandit"]["path"])
            self.assertEqual(entries["hunyuan_foley_xxl"]["status"], "missing")
            self.assertIsNone(entries["hunyuan_foley_xxl"]["path"])

    def test_config_find_soundseparator_env_override(self) -> None:
        config.ensure_directories()
        with tempfile.TemporaryDirectory(dir=str(config.OUTPUT_DIR)) as tmp:
            with mock.patch.dict(os.environ, {"SOUNDSEPARATOR_DIR": tmp}):
                found = config.find_soundseparator_dir()
                self.assertEqual(found, Path(tmp).resolve())

    def test_mcp_initialize_and_tools_list(self) -> None:
        server = ciness_mcp.CineSSMcpServer()
        init = ciness_mcp.dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
            server,
        )
        self.assertEqual(init["serverInfo"]["name"], "ciness")

        listed = ciness_mcp.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, server)
        names = {t["name"] for t in listed["tools"]}
        expected = {
            "ciness_status",
            "ciness_start_server",
            "ciness_wait_job",
            "ciness_claim_and_run_job",
            "ciness_job_status",
            "ciness_cancel_job",
            "ciness_get_director_context",
            "ciness_apply_director_plan",
        }
        self.assertEqual(names, expected)
        self.assertEqual(len(listed["tools"]), 8)

    def test_foley_env_detection_and_shutdown_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from app.audio.foley_engine import find_hunyuan_foley
        from app.server import app

        config.ensure_directories()
        # 1. 一時的な擬似 HunyuanVideo-Foley ディレクトリでの検出テスト
        with tempfile.TemporaryDirectory(dir=str(config.OUTPUT_DIR)) as tmp:
            root = Path(tmp)
            (root / "app").mkdir(parents=True)
            (root / "app" / "infer.py").write_text("# dummy", encoding="utf-8")
            (root / "hf-cache").mkdir()
            mdir = root / "models" / "HunyuanVideo-Foley"
            mdir.mkdir(parents=True)
            for fname in (
                "hunyuanvideo_foley.pth",
                "hunyuanvideo_foley_xl.pth",
                "synchformer_state_dict.pth",
                "vae_128d_48k.pth",
            ):
                (mdir / fname).write_bytes(b"WEIGHT")

            with mock.patch.dict(os.environ, {"HUNYUAN_FOLEY_DIR": str(root)}):
                info = find_hunyuan_foley()
                self.assertTrue(info["available"])
                self.assertEqual(Path(info["infer_script"]).name, "infer.py")
                self.assertEqual(info["hf_home"], str(root.resolve() / "hf-cache"))
                self.assertIsNotNone(info["variants"]["XXL"])
                self.assertIsNotNone(info["variants"]["XL"])

        # 2. 実機に C:/AI/HunyuanVideo-Foley が存在する環境ならそれも自動検出されること
        if Path("C:/AI/HunyuanVideo-Foley/app/infer.py").is_file():
            real_info = find_hunyuan_foley()
            self.assertTrue(real_info["available"])
            self.assertTrue(real_info["variants"]["XXL"] or real_info["variants"]["XL"])

        # 3. POST /api/system/shutdown と停止フラグ・エージェント通知確認（exit 無効化下）
        flag = config.SHUTDOWN_FLAG_PATH
        had_flag = flag.is_file()
        try:
            with mock.patch.dict(os.environ, {"CINESS_DISABLE_EXIT": "1"}):
                client = TestClient(app)
                resp = client.post("/api/system/shutdown", json={"force": False})
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(body["status"], "shutting_down")
                self.assertTrue(flag.is_file())

                nxt = client.get("/api/agent/next")
                self.assertEqual(nxt.status_code, 200)
                self.assertTrue(nxt.json().get("shutdown"))
        finally:
            if not had_flag:
                flag.unlink(missing_ok=True)


class TestCrossPlatformAndDirectorCLI(unittest.TestCase):
    """クロスプラットフォーム判定（Mac MPS / 非NVIDIA GPU / venv / device）と監督CLI・MCPを検証する。"""

    def test_find_venv_python_windows_and_posix(self) -> None:
        config.ensure_directories()
        with tempfile.TemporaryDirectory(dir=str(config.OUTPUT_DIR)) as tmp:
            root = Path(tmp)
            # 1. 空の場合は存在しないパスが返る
            empty_py = config.find_venv_python(root)
            self.assertFalse(empty_py.is_file())

            # 2. POSIX .venv/bin/python3 -> .venv/bin/python 優先順
            bin_dir = root / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            py3 = bin_dir / "python3"
            py3.write_text("", encoding="utf-8")
            self.assertEqual(config.find_venv_python(root), py3)

            py_bin = bin_dir / "python"
            py_bin.write_text("", encoding="utf-8")
            self.assertEqual(config.find_venv_python(root), py_bin)

            # 3. Windows .venv/Scripts/python.exe
            win_dir = root / ".venv" / "Scripts"
            win_dir.mkdir(parents=True)
            win_py = win_dir / "python.exe"
            win_py.write_text("", encoding="utf-8")
            if os.name == "nt":
                self.assertEqual(config.find_venv_python(root), win_py)
            else:
                self.assertEqual(config.find_venv_python(root), py_bin)

    def test_get_torch_device_priority_and_fallback(self) -> None:
        def _fake_torch(cuda_ok: bool, mps_ok: bool) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: cuda_ok),
                backends=types.SimpleNamespace(
                    mps=types.SimpleNamespace(is_available=lambda: mps_ok)
                ),
            )

        with mock.patch.dict(sys.modules, {"torch": _fake_torch(True, True)}):
            self.assertEqual(config.get_torch_device(), ("cuda", "NVIDIA CUDA"))

        with mock.patch.dict(sys.modules, {"torch": _fake_torch(False, True)}):
            self.assertEqual(config.get_torch_device(), ("mps", "Apple Silicon MPS"))

        with mock.patch.dict(sys.modules, {"torch": _fake_torch(False, False)}):
            self.assertEqual(config.get_torch_device(), ("cpu", "CPU"))

        # torch 未導入時でも例外を出さず ("cpu", "CPU") を返す
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(config.get_torch_device(), ("cpu", "CPU"))

    def test_apple_silicon_mps_unified_memory_tiers(self) -> None:
        # 1. 32GB Apple Silicon MPS -> Tier A, silent=True(offload+省メモリ助言), HOA7
        hw_32gb = {
            "cpu": {"name": "Apple M2 Max", "cores": 12, "threads": 12},
            "gpu": {
                "name": "Apple M2 Max (MPS ユニファイドメモリ)",
                "cuda": False,
                "mps": True,
                "backend": "mps",
                "unified_memory": True,
                "non_nvidia_gpu": False,
                "vram_total_gb": 0.0,
                "vram_free_gb": 0.0,
                "driver": "Metal (MPS)",
                "nvidia_detected": False,
            },
            "memory": {"total_gb": 32.0, "available_gb": 22.0},
            "ffmpeg": {"available": True, "version": "7.1"},
            "disk": {"free_gb": 120.0},
            "soundseparator": {"detected": False, "path": None, "shared_models": []},
        }
        rep32 = hardware_advisor.assess_system(hw_32gb)
        self.assertEqual(rep32["gpu"]["backend"], "mps")
        self.assertTrue(rep32["gpu"]["mps"])
        self.assertTrue(rep32["gpu"]["unified_memory"])
        self.assertEqual(rep32["gpu"]["status"], "ok")
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK=1", rep32["gpu"]["note"])
        self.assertEqual(rep32["recommendations"]["tier"], "A")
        self.assertTrue(rep32["recommendations"]["voiced"]["supported"])
        self.assertTrue(rep32["recommendations"]["silent"]["supported"])
        self.assertTrue(rep32["recommendations"]["silent"]["models"]["offload"])
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK=1", rep32["recommendations"]["silent"]["reason"])
        self.assertEqual(rep32["recommendations"]["max_recommended_ambisonics_order"], 7)

        # 2. 16GB Apple Silicon MPS -> Tier B, warn, silent=True(短尺・省メモリ注意), HOA3
        hw_16gb = dict(hw_32gb)
        hw_16gb["cpu"] = {"name": "Apple M2", "cores": 8, "threads": 8}
        hw_16gb["memory"] = {"total_gb": 16.0, "available_gb": 10.0}
        rep16 = hardware_advisor.assess_system(hw_16gb)
        self.assertEqual(rep16["recommendations"]["tier"], "B")
        self.assertEqual(rep16["gpu"]["status"], "warn")
        self.assertTrue(rep16["recommendations"]["silent"]["supported"])
        self.assertIn("省メモリ", rep16["recommendations"]["silent"]["reason"])
        self.assertEqual(rep16["recommendations"]["max_recommended_ambisonics_order"], 3)

        # 3. 8GB Apple Silicon MPS -> Tier C, silent=False
        hw_8gb = dict(hw_32gb)
        hw_8gb["memory"] = {"total_gb": 8.0, "available_gb": 4.0}
        rep8 = hardware_advisor.assess_system(hw_8gb)
        self.assertEqual(rep8["recommendations"]["tier"], "C")
        self.assertFalse(rep8["recommendations"]["silent"]["supported"])

    def test_macos_sysctl_probes(self) -> None:
        def _fake_run(cmd, **_kwargs):
            if cmd[:2] == ["sysctl", "-n"]:
                mapping = {
                    "machdep.cpu.brand_string": "Apple M3 Pro",
                    "hw.physicalcpu": "12",
                    "hw.logicalcpu": "12",
                    "hw.memsize": str(24 * 1024**3),
                }
                val = mapping.get(cmd[2], "")
                return mock.Mock(returncode=0 if val else 1, stdout=val + "\n")
            if cmd[:1] == ["vm_stat"]:
                vm_out = (
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                    "Pages free:                             300000.\n"
                    "Pages inactive:                         200000.\n"
                )
                return mock.Mock(returncode=0, stdout=vm_out)
            return mock.Mock(returncode=1, stdout="")

        with (
            mock.patch.object(hardware_advisor.platform, "system", return_value="Darwin"),
            mock.patch.object(hardware_advisor.subprocess, "run", side_effect=_fake_run),
            mock.patch.dict(sys.modules, {"psutil": None}),
        ):
            cpu = hardware_advisor._probe_cpu()
            mem = hardware_advisor._probe_memory()

        self.assertEqual(cpu["name"], "Apple M3 Pro")
        self.assertEqual(cpu["cores"], 12)
        self.assertEqual(cpu["threads"], 12)
        self.assertAlmostEqual(mem["total_gb"], 24.0, delta=0.1)
        self.assertGreater(mem["available_gb"], 5.0)

    def test_non_nvidia_gpu_fallback(self) -> None:
        hw_amd = {
            "cpu": {"name": "AMD Ryzen 7 7840U", "cores": 8, "threads": 16},
            "gpu": {
                "name": "AMD Radeon 780M Graphics (非NVIDIA・CPU動作)",
                "cuda": False,
                "mps": False,
                "backend": "cpu",
                "unified_memory": False,
                "non_nvidia_gpu": True,
                "vram_total_gb": 0.0,
                "vram_free_gb": 0.0,
                "driver": None,
                "nvidia_detected": False,
            },
            "memory": {"total_gb": 32.0, "available_gb": 24.0},
            "ffmpeg": {"available": True, "version": "7.0"},
            "disk": {"free_gb": 100.0},
            "soundseparator": {"detected": False, "path": None, "shared_models": []},
        }
        rep = hardware_advisor.assess_system(hw_amd)
        self.assertTrue(rep["gpu"]["non_nvidia_gpu"])
        self.assertEqual(rep["gpu"]["backend"], "cpu")
        self.assertEqual(rep["gpu"]["status"], "warn")
        self.assertIn("非NVIDIA", rep["gpu"]["note"])
        self.assertEqual(rep["recommendations"]["tier"], "C")
        self.assertTrue(rep["recommendations"]["voiced"]["supported"])
        self.assertFalse(rep["recommendations"]["silent"]["supported"])
        self.assertEqual(rep["recommendations"]["max_recommended_ambisonics_order"], 1)

    def test_model_manager_torch_index_selection(self) -> None:
        self.assertIsNone(model_manager._select_foley_torch_index_url(system_name="Darwin", has_nvidia=False))
        self.assertEqual(
            model_manager._select_foley_torch_index_url(system_name="Windows", has_nvidia=True),
            "https://download.pytorch.org/whl/cu124",
        )
        self.assertEqual(
            model_manager._select_foley_torch_index_url(system_name="Windows", has_nvidia=False),
            "https://download.pytorch.org/whl/cpu",
        )
        self.assertEqual(
            model_manager._select_foley_torch_index_url(system_name="Linux", has_nvidia=False),
            "https://download.pytorch.org/whl/cpu",
        )

    def test_director_cli_and_mcp_tools(self) -> None:
        sample_ctx = {
            "job_id": "job_test_01",
            "duration_sec": 15.0,
            "intensity_track": [{"t_sec": 0.0, "temperature": 0.8}],
            "scene_segments": [{"start_sec": 0.0, "end_sec": 15.0, "label": "クライマックス"}],
            "sfx_layers": [{"name": "sfx_sub", "role": "sfx_sub"}],
            "sfx_events": [{"t_sec": 3.2, "peak_db": -2.5}],
            "director_plan": {"notes": "自動生成", "keyframes": []},
            "director_sheet_url": "/api/jobs/job_test_01/media/director_sheet",
        }
        sample_apply_resp = {
            "job_id": "job_test_01",
            "status": "done",
            "director_plan": {"notes": "テスト演出", "keyframes": [{"t_sec": 2.0, "gain_db": 3.0}]},
            "result": {"lufs": -14.2},
        }

        with mock.patch.object(ciness_agent, "_http_json", return_value=sample_ctx) as m_get:
            out_ctx = ciness_agent.cmd_director_context("job_test_01", server_url="http://127.0.0.1:8780")
        m_get.assert_called_once_with(
            "GET", "http://127.0.0.1:8780/api/jobs/job_test_01/director-context", timeout=10.0
        )
        self.assertEqual(out_ctx["job_id"], "job_test_01")
        self.assertEqual(out_ctx["director_sheet_url"], "/api/jobs/job_test_01/media/director_sheet")

        # MCP tools/call: ciness_get_director_context & ciness_apply_director_plan
        server = ciness_mcp.CineSSMcpServer()
        with mock.patch.object(ciness_agent, "_http_json", return_value=sample_ctx):
            res1 = ciness_mcp.dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {"name": "ciness_get_director_context", "arguments": {"job_id": "job_test_01"}},
                },
                server,
            )
        self.assertFalse(res1["isError"])
        self.assertEqual(json.loads(res1["content"][0]["text"])["job_id"], "job_test_01")

        # --plan ファイル指定＆二重ラップ解除のテスト
        config.ensure_directories()
        with tempfile.TemporaryDirectory(dir=str(config.OUTPUT_DIR)) as tmp:
            plan_file = Path(tmp) / "plan.json"
            plan_file.write_text(
                json.dumps({"director_plan": {"notes": "テスト演出", "keyframes": [{"t_sec": 2.0, "gain_db": 3.0}]}}),
                encoding="utf-8",
            )
            with mock.patch.object(ciness_agent, "_http_json", return_value=sample_apply_resp) as m_post:
                res_apply = ciness_agent.cmd_apply_director(
                    "job_test_01", str(plan_file), server_url="http://127.0.0.1:8780"
                )
        m_post.assert_called_once_with(
            "POST",
            "http://127.0.0.1:8780/api/jobs/job_test_01/director-plan",
            {"director_plan": {"notes": "テスト演出", "keyframes": [{"t_sec": 2.0, "gain_db": 3.0}]}},
            timeout=180.0,
        )
        self.assertEqual(res_apply["result"]["lufs"], -14.2)

        with mock.patch.object(ciness_agent, "_http_json", return_value=sample_apply_resp):
            res2 = ciness_mcp.dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {
                        "name": "ciness_apply_director_plan",
                        "arguments": {
                            "job_id": "job_test_01",
                            "director_plan": {"notes": "テスト演出", "keyframes": []},
                        },
                    },
                },
                server,
            )
        self.assertFalse(res2["isError"])
        self.assertEqual(json.loads(res2["content"][0]["text"])["status"], "done")


if __name__ == "__main__":
    unittest.main()

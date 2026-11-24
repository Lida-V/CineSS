# coding: utf-8
"""hardware_advisor / config / model_manager / MCP プロトコルのユニットテスト。

高スペック（RTX 4090 24GB / 64GB+）・中スペック（VRAM 8GB / 16GB RAM）・
低スペック（GPUなし / 8GB RAM）の3ケースをモック検証し、
✅(ok) / ⚠️(warn) / ✕(error) 判定・推奨動画長・モデル選定が妥当に変化することを確認する。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import config, hardware_advisor, model_manager
from tools import ciness_mcp


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
        for key in ("name", "cuda", "vram_total_gb", "vram_free_gb", "driver", "status", "note"):
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
        }
        self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()

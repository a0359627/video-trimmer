"""Unit tests for edl_auditor.py (pre-render micro-window repair, timing sanitizer, and 8-dimension audit)."""

import tempfile
import unittest
from pathlib import Path

from scripts.edl_auditor import (
    audit_edl_quality,
    detect_micro_window_anomalies,
    generate_edl_audit_markdown,
    repair_edl_micro_windows,
    sanitize_refined_edl,
)


def _sample_whisper_units():
    return [
        {"id": 1, "start": 5.0, "end": 10.0, "text": "大家好歡迎來到科學頻道", "is_target_speaker": True},
        {"id": 2, "start": 12.0, "end": 16.0, "text": "今天我們要聊量子力學的", "is_target_speaker": True},
        {"id": 3, "start": 17.0, "end": 24.0, "text": "今天我們要聊量子力學的核心概念與雙縫實驗", "is_target_speaker": True},
        {"id": 4, "start": 30.0, "end": 40.0, "text": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性", "is_target_speaker": True},
        {"id": 5, "start": 45.0, "end": 55.0, "text": "最後請記得訂閱我們的頻道並開啟小鈴鐺", "is_target_speaker": True},
    ]


class TestMicroWindowAnomalyDetectionAndRepair(unittest.TestCase):
    def test_detects_potential_residual_retake_and_repairs_window(self):
        whisper_units = _sample_whisper_units()
        model_edl = {
            "project_title": "Test",
            "final_edl": [
                {"clip_id": 1, "topic": "開場", "sentence_ids": [1], "source_in": 5.0, "source_out": 10.0, "transcript": "大家好歡迎來到科學頻道"},
                {"clip_id": 2, "topic": "NG句", "sentence_ids": [2], "source_in": 12.0, "source_out": 16.0, "transcript": "今天我們要聊量子力學的"},
                {"clip_id": 3, "topic": "正式句", "sentence_ids": [3], "source_in": 17.0, "source_out": 24.0, "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗"},
            ],
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=60.0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["type"], "POTENTIAL_RESIDUAL_RETAKE")
        self.assertEqual(anomalies[0]["replace_range"], (1, 3))

        captured_calls = []

        def _fake_window_inference(win_units, win_start, win_end, start_offset, end_offset, anomaly):
            captured_calls.append((start_offset, end_offset, [u["id"] for u in win_units]))
            return {
                "final_edl": [
                    {
                        "clip_id": 1,
                        "topic": "正式句修復",
                        "sentence_ids": [3],
                        "source_in": 17.0,
                        "source_out": 24.0,
                        "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗",
                    }
                ]
            }

        repaired_edl, repairs = repair_edl_micro_windows(
            model_edl=model_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            run_window_inference_fn=_fake_window_inference,
        )
        self.assertEqual(len(repairs), 1)
        self.assertEqual(len(captured_calls), 1)
        self.assertEqual(len(repaired_edl["final_edl"]), 2)
        self.assertEqual(repaired_edl["final_edl"][0]["sentence_ids"], [1])
        self.assertEqual(repaired_edl["final_edl"][1]["sentence_ids"], [3])

    def test_detects_and_rescues_missing_script_block_in_mode_a(self):
        whisper_units = _sample_whisper_units()
        script_text = (
            "大家好歡迎來到科學頻道\n"
            "今天我們要聊量子力學的核心概念與雙縫實驗\n"
            "雙縫實驗證明了微觀粒子同時具備波動性與粒子性\n"
            "最後請記得訂閱我們的頻道並開啟小鈴鐺\n"
        )
        # Omit sentence 4 (Script Block 03) from initial model_edl
        model_edl = {
            "project_title": "Mode A Test",
            "final_edl": [
                {"clip_id": 1, "topic": "開場", "sentence_ids": [1], "source_in": 5.0, "source_out": 10.0, "transcript": "大家好歡迎來到科學頻道"},
                {"clip_id": 2, "topic": "主旨", "sentence_ids": [3], "source_in": 17.0, "source_out": 24.0, "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗"},
                {"clip_id": 3, "topic": "結尾", "sentence_ids": [5], "source_in": 45.0, "source_out": 55.0, "transcript": "最後請記得訂閱我們的頻道並開啟小鈴鐺"},
            ],
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=60.0, script_text=script_text)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["type"], "MISSING_SCRIPT_BLOCK")
        self.assertEqual(anomalies[0]["script_block"]["label"], "[Script Block 03]")

        def _fake_rescue_inference(win_units, win_start, win_end, start_offset, end_offset, anomaly):
            return {
                "final_edl": [
                    {
                        "clip_id": 1,
                        "topic": "雙縫實驗",
                        "sentence_ids": [4],
                        "source_in": 30.0,
                        "source_out": 40.0,
                        "transcript": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性",
                    }
                ]
            }

        repaired_edl, repairs = repair_edl_micro_windows(
            model_edl=model_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            run_window_inference_fn=_fake_rescue_inference,
            script_text=script_text,
        )
        self.assertEqual(len(repairs), 1)
        self.assertEqual([c["sentence_ids"] for c in repaired_edl["final_edl"]], [[1], [3], [4], [5]])


class TestSanitizeRefinedEdl(unittest.TestCase):
    def test_resolves_overlaps_restores_plosive_tail_and_merges_micro_clips(self):
        raw_refined = [
            {
                "clip_id": 1,
                "topic": "A",
                "sentence_ids": [1],
                "t_first": 5.0,
                "t_last": 10.10,
                "source_in": 4.92,
                "source_out": 10.00,  # Truncates t_last (10.10) -> should restore to >= 10.10
                "duration": 5.08,
                "cps": 4.0,
                "in_margin": 0.08,
                "out_margin": 0.0,
                "transcript": "第一段完整的台詞內容",
            },
            {
                "clip_id": 2,
                "topic": "B",
                "sentence_ids": [2],
                "t_first": 10.12,
                "t_last": 10.35,
                "source_in": 10.05,  # Overlaps Clip 1 source_out AND is a <0.45s micro-clip
                "source_out": 10.38,
                "duration": 0.33,
                "cps": 3.5,
                "in_margin": 0.07,
                "out_margin": 0.03,
                "transcript": "補充詞",
            },
        ]
        sanitized, stats = sanitize_refined_edl(raw_refined, total_dur=30.0)
        self.assertEqual(stats["plosive_tails_restored"], 1)
        self.assertEqual(stats["overlaps_resolved"], 1)
        self.assertEqual(stats["micro_clips_merged"], 1)
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized[0]["sentence_ids"], [1, 2])
        self.assertGreaterEqual(sanitized[0]["source_out"], 10.35)


class TestAuditEdlQuality(unittest.TestCase):
    def test_clean_edl_passes_quality_gate_and_generates_markdown(self):
        whisper_units = _sample_whisper_units()
        refined_edl = [
            {
                "clip_id": 1,
                "topic": "開場",
                "sentence_ids": [1],
                "t_first": 5.0,
                "t_last": 10.0,
                "source_in": 4.92,
                "source_out": 10.12,
                "duration": 5.20,
                "cps": 4.2,
                "in_margin": 0.08,
                "out_margin": 0.12,
                "transcript": "大家好歡迎來到科學頻道",
            },
            {
                "clip_id": 2,
                "topic": "主題",
                "sentence_ids": [3],
                "t_first": 17.0,
                "t_last": 24.0,
                "source_in": 16.92,
                "source_out": 24.12,
                "duration": 7.20,
                "cps": 4.5,
                "in_margin": 0.08,
                "out_margin": 0.12,
                "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗",
            },
        ]
        report = audit_edl_quality(
            refined_edl=refined_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        verdict = report["agent_verdict"]
        self.assertTrue(verdict["pass_quality_gate"])
        self.assertEqual(verdict["suggested_action"], "DELIVER")
        self.assertEqual(verdict["fatal_violations"], [])

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "report.md"
            generate_edl_audit_markdown(report, md_path)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("## 1. Executive Summary & Agent Verdict", md_text)
            self.assertIn("`True`", md_text)

    def test_residual_retake_fails_quality_gate_with_one_shot_remediate(self):
        whisper_units = _sample_whisper_units()
        refined_edl = [
            {
                "clip_id": 1,
                "topic": "NG",
                "sentence_ids": [2],
                "t_first": 12.0,
                "t_last": 16.0,
                "source_in": 11.92,
                "source_out": 16.10,
                "duration": 4.18,
                "cps": 3.8,
                "in_margin": 0.08,
                "out_margin": 0.10,
                "transcript": "今天我們要聊量子力學的",
            },
            {
                "clip_id": 2,
                "topic": "OK",
                "sentence_ids": [3],
                "t_first": 17.0,
                "t_last": 24.0,
                "source_in": 16.92,
                "source_out": 24.10,
                "duration": 7.18,
                "cps": 4.2,
                "in_margin": 0.08,
                "out_margin": 0.10,
                "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗",
            },
        ]
        report = audit_edl_quality(
            refined_edl=refined_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        verdict = report["agent_verdict"]
        self.assertFalse(verdict["pass_quality_gate"])
        self.assertEqual(verdict["suggested_action"], "ONE_SHOT_REMEDIATE")
        self.assertTrue(any("residual retake" in v for v in verdict["fatal_violations"]))

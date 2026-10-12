"""Show Mode 離線單元測試：麥克風角色歸屬、倒數口令、角色拆句、重排容許、全片組裝正規化、Mode C 提示詞。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.assembly import format_candidates_for_assembly, normalize_assembled_edl, parse_assembly_json
from scripts.edl_auditor import audit_edl_quality, deduplicate_and_sort_clips, sanitize_refined_edl
from scripts.gemini_client import build_prompt
from scripts.roles import (
    ROLE_CUE,
    ROLE_HOST,
    ROLE_OFFSCREEN,
    attribute_word_roles,
    detect_offscreen_channel,
    has_role_tags,
    split_segments_by_role,
)
from scripts.transcribe import format_whisper_transcript_for_prompt, merge_whisper_segments_to_sentences

SR = 16000


def _tone(start, end, left_db, right_db, sr=SR):
    """Return (n, 2) float32 audio with the given per-channel RMS levels over [start, end)."""
    n = int((end - start) * sr)
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    out = np.zeros((n, 2), dtype=np.float32)
    out[:, 0] = sig * (10 ** (left_db / 20.0)) / rms
    out[:, 1] = sig * (10 ** (right_db / 20.0)) / rms
    return out


def _synthetic_two_mic_audio():
    """0-1s host on right (L -40 / R -25), 1-2s director on left (L -28 / R -45), 2-3s host again."""
    return np.concatenate([
        _tone(0.0, 1.0, -40.0, -25.0),
        _tone(1.0, 2.0, -28.0, -45.0),
        _tone(2.0, 3.0, -40.0, -25.0),
    ])


def _words(spec):
    return [{"word": w, "start": s, "end": e} for (w, s, e) in spec]


class TestMicRoles(unittest.TestCase):
    def test_detects_quieter_channel_as_offscreen(self):
        audio = _synthetic_two_mic_audio()
        assert detect_offscreen_channel(audio) == "left"

    def test_words_are_attributed_by_channel_difference(self):
        audio = _synthetic_two_mic_audio()
        seg = {"id": 1, "start": 0.0, "end": 3.0, "text": "主持人說 導演問 主持人答",
               "words": _words([("主持人說", 0.1, 0.9), ("導演問", 1.1, 1.9), ("主持人答", 2.1, 2.9)])}
        out = attribute_word_roles([seg], audio, SR, offscreen_channel="left")
        roles = [w["mic_role"] for w in out[0]["words"]]
        assert roles == [ROLE_HOST, ROLE_OFFSCREEN, ROLE_HOST]
        assert all(w["is_target_speaker"] for w in out[0]["words"])

    def test_split_by_role_creates_single_speaker_segments_and_keeps_offscreen_usable(self):
        audio = _synthetic_two_mic_audio()
        seg = {"id": 7, "start": 0.0, "end": 3.0, "text": "x",
               "words": _words([("主持人說", 0.1, 0.9), ("導演問", 1.1, 1.9), ("主持人答", 2.1, 2.9)])}
        split = split_segments_by_role(attribute_word_roles([seg], audio, SR, offscreen_channel="left"))
        assert [s["speaker_id"] for s in split] == [ROLE_HOST, ROLE_OFFSCREEN, ROLE_HOST]
        assert [s["id"] for s in split] == [1, 2, 3]
        assert split[1]["text"] == "導演問"
        assert split[1]["is_target_speaker"] is True  # OFFSCREEN stays usable; the prompt decides
        assert all(s["source_segment_id"] == 7 for s in split)

    def test_leading_countdown_becomes_cue_and_is_not_target_speaker(self):
        audio = _tone(0.0, 4.0, -40.0, -25.0)
        seg = {"id": 1, "start": 0.0, "end": 4.0, "text": "5,4,3,2 如果我們有業配的話",
               "words": _words([("5,", 0.0, 0.3), ("4,", 0.3, 0.6), ("3,", 0.6, 0.9), ("2", 0.9, 1.2),
                                ("如果", 1.5, 1.8), ("我們", 1.8, 2.1), ("有", 2.1, 2.3), ("業配", 2.3, 2.6), ("的話", 2.6, 2.9)])}
        split = split_segments_by_role(attribute_word_roles([seg], audio, SR, offscreen_channel="left"))
        assert split[0]["speaker_id"] == ROLE_CUE
        assert split[0]["is_target_speaker"] is False
        assert split[0]["text"].replace(",", "") == "5432"
        assert split[1]["speaker_id"] == ROLE_HOST
        assert split[1]["text"] == "如果我們有業配的話"

    def test_tiny_edge_word_on_wrong_mic_is_smoothed_into_host_run(self):
        # "如果" (first word) measured on the director mic, glued (gap 0.05s) to the host's sentence
        audio = np.concatenate([_tone(0.0, 0.4, -28.0, -45.0), _tone(0.4, 3.0, -40.0, -25.0)])
        seg = {"id": 1, "start": 0.0, "end": 3.0, "text": "如果我跟你說",
               "words": _words([("如果", 0.05, 0.35), ("我", 0.4, 0.8), ("跟你說", 0.8, 1.6)])}
        split = split_segments_by_role(attribute_word_roles([seg], audio, SR, offscreen_channel="left"))
        assert len(split) == 1 and split[0]["speaker_id"] == ROLE_HOST
        assert split[0]["text"] == "如果我跟你說"

    def test_standalone_short_offscreen_word_with_gap_is_kept(self):
        # "沒錯" said by the director 0.6s after the host finished: not glued, stays OFFSCREEN
        audio = np.concatenate([_tone(0.0, 2.0, -40.0, -25.0), _tone(2.0, 3.0, -28.0, -45.0)])
        seg = {"id": 1, "start": 0.0, "end": 3.0, "text": "保護它 沒錯",
               "words": _words([("保護", 0.1, 0.8), ("它", 0.8, 1.2), ("沒錯", 2.1, 2.6)])}
        split = split_segments_by_role(attribute_word_roles([seg], audio, SR, offscreen_channel="left"))
        assert [s["speaker_id"] for s in split] == [ROLE_HOST, ROLE_OFFSCREEN]

    def test_single_digit_inside_content_is_not_a_cue(self):
        audio = _tone(0.0, 3.0, -40.0, -25.0)
        seg = {"id": 1, "start": 0.0, "end": 3.0, "text": "一瓶大概一百台幣",
               "words": _words([("一", 0.0, 0.2), ("瓶", 0.2, 0.4), ("大概", 0.4, 0.8), ("一", 0.8, 1.0), ("百", 1.0, 1.2), ("台幣", 1.2, 1.6)])}
        split = split_segments_by_role(attribute_word_roles([seg], audio, SR, offscreen_channel="left"))
        assert len(split) == 1 and split[0]["speaker_id"] == ROLE_HOST

    def test_merge_splits_sentences_at_role_change_and_prompt_shows_tags(self):
        audio = _synthetic_two_mic_audio()
        seg = {"id": 1, "start": 0.0, "end": 3.0, "text": "x",
               "words": _words([("主持人說", 0.1, 0.9), ("導演問", 1.0, 1.9), ("主持人答", 2.0, 2.9)])}
        split = split_segments_by_role(attribute_word_roles([seg], audio, SR, offscreen_channel="left"))
        sentences = merge_whisper_segments_to_sentences(split)
        assert [s["speaker_id"] for s in sentences] == [ROLE_HOST, ROLE_OFFSCREEN, ROLE_HOST]
        assert has_role_tags(sentences)
        text = format_whisper_transcript_for_prompt(sentences)
        assert "[OFFSCREEN] 導演問" in text and "[HOST] 主持人說" in text and "Show Mode" in text

    def test_prompt_without_roles_is_unchanged(self):
        sentences = merge_whisper_segments_to_sentences([
            {"id": 1, "start": 0.0, "end": 1.0, "text": "第一句", "words": []},
        ])
        assert not has_role_tags(sentences)
        text = format_whisper_transcript_for_prompt(sentences)
        assert "[HOST]" not in text and "Last Take Wins" in text


class TestReorderSupport(unittest.TestCase):
    def _clips(self):
        return [
            {"clip_id": 1, "topic": "B", "source_in": 20.0, "source_out": 25.0, "duration": 5.0, "t_last": 24.8,
             "sentence_ids": [5], "transcript": "職人醬油的組合包拿來做套餐搭配", "cps": 3.0},
            {"clip_id": 2, "topic": "A", "source_in": 10.0, "source_out": 15.0, "duration": 5.0, "t_last": 14.8,
             "sentence_ids": [2], "transcript": "丸八的甘口醬油拿來當桌上醬油", "cps": 3.0},
        ]

    def test_sanitize_restores_timeline_order_when_reorder_allowed(self):
        sanitized, _ = sanitize_refined_edl(self._clips(), total_dur=60.0, allow_reorder=True)
        assert [c["topic"] for c in sanitized] == ["B", "A"]
        assert [c["clip_id"] for c in sanitized] == [1, 2]
        assert all("_seq" not in c and "_seq_last" not in c for c in sanitized)

    def test_sanitize_sorts_by_source_when_reorder_not_allowed(self):
        sanitized, _ = sanitize_refined_edl(self._clips(), total_dur=60.0)
        assert [c["topic"] for c in sanitized] == ["A", "B"]

    def test_sanitize_reorder_still_resolves_source_overlaps(self):
        clips = [
            {"clip_id": 1, "topic": "later-first", "source_in": 12.0, "source_out": 18.0, "duration": 6.0,
             "t_last": 17.8, "t_first": 12.2, "sentence_ids": [3], "transcript": "後段", "cps": 3.0},
            {"clip_id": 2, "topic": "earlier-second", "source_in": 10.0, "source_out": 13.0, "duration": 3.0,
             "t_last": 12.0, "t_first": 10.1, "sentence_ids": [2], "transcript": "前段", "cps": 3.0},
        ]
        sanitized, stats = sanitize_refined_edl(clips, total_dur=60.0, allow_reorder=True)
        assert stats["overlaps_resolved"] == 1
        assert [c["topic"] for c in sanitized] == ["later-first", "earlier-second"]
        a = next(c for c in sanitized if c["topic"] == "earlier-second")
        b = next(c for c in sanitized if c["topic"] == "later-first")
        assert a["source_out"] <= b["source_in"] + 1e-6

    def test_audit_treats_reorder_as_warning_only_when_allowed(self):
        units = [
            {"id": 2, "start": 10.0, "end": 15.0, "text": "丸八的甘口醬油拿來當桌上醬油", "words": [], "is_target_speaker": True},
            {"id": 5, "start": 20.0, "end": 25.0, "text": "職人醬油的組合包拿來做套餐搭配", "words": [], "is_target_speaker": True},
        ]
        clips = [dict(c, in_margin=0.1, out_margin=0.1) for c in self._clips()]
        strict = audit_edl_quality(clips, units, 60.0, Path("x.mp4"))
        assert strict["agent_verdict"]["pass_quality_gate"] is False
        assert any("out-of-order" in v for v in strict["agent_verdict"]["fatal_violations"])

        lenient = audit_edl_quality(clips, units, 60.0, Path("x.mp4"), allow_reorder=True,
                                    mode_label="Mode C (Script Reference, Show Mode)", extra_cli_flags="--assemble --allow-reorder")
        assert lenient["agent_verdict"]["pass_quality_gate"] is True
        assert any("reordered" in w for w in lenient["agent_verdict"]["warnings"])
        assert lenient["summary"]["mode"].startswith("Mode C")
        assert lenient["dimensions"]["2_timeline_integrity"]["reorder_allowed"] is True

    def test_audit_remediation_cmd_carries_show_mode_flags(self):
        units = [{"id": 2, "start": 10.0, "end": 15.0, "text": "第一段內容", "words": [], "is_target_speaker": True}]
        clips = [{"clip_id": 1, "topic": "A", "source_in": 10.0, "source_out": 11.0, "duration": 1.0, "t_last": 14.8,
                  "sentence_ids": [2], "transcript": "第一段內容", "cps": 3.0, "in_margin": 0.1, "out_margin": 0.1}]
        report = audit_edl_quality(clips, units, 60.0, Path("x.mp4"), extra_cli_flags="--assemble --xml-only")
        assert report["agent_verdict"]["pass_quality_gate"] is False  # t_last truncated -> fatal
        assert report["agent_verdict"]["remediation_cmd"].endswith("--assemble --xml-only")

    def test_dedupe_keep_order_drops_shared_sentences_but_keeps_timeline(self):
        units = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "a", "words": []},
            {"id": 2, "start": 3.0, "end": 5.0, "text": "b", "words": []},
            {"id": 3, "start": 6.0, "end": 8.0, "text": "c", "words": []},
        ]
        clips = [
            {"clip_id": 1, "sentence_ids": [3], "topic": "c"},
            {"clip_id": 2, "sentence_ids": [1], "topic": "a"},
            {"clip_id": 3, "sentence_ids": [1, 2], "topic": "dup"},
        ]
        out = deduplicate_and_sort_clips(clips, units, keep_order=True)
        assert [c["topic"] for c in out] == ["c", "a"]
        assert [c["clip_id"] for c in out] == [1, 2]


class TestAssembly(unittest.TestCase):
    def _units(self):
        return [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "開場", "words": [], "speaker_id": "HOST", "is_target_speaker": True},
            {"id": 2, "start": 2.5, "end": 3.0, "text": "5432", "words": [], "speaker_id": "CUE", "is_target_speaker": False},
            {"id": 3, "start": 3.5, "end": 6.0, "text": "正片", "words": [], "speaker_id": "HOST", "is_target_speaker": True},
            {"id": 4, "start": 6.2, "end": 7.0, "text": "好扯", "words": [], "speaker_id": "OFFSCREEN", "is_target_speaker": True},
        ]

    def test_normalize_orders_by_timeline_position_and_drops_invalid(self):
        chunks = [{"final_edl": [{"clip_id": 1, "sentence_ids": [3], "visual_check": "眼神直視"}]}]
        assembled = {
            "final_edl": [
                {"clip_id": 9, "sentence_ids": [4], "timeline_position": 3, "role": "weird"},
                {"clip_id": 8, "sentence_ids": [3], "timeline_position": 2, "source_in": 99.0, "source_out": 100.0},
                {"clip_id": 7, "sentence_ids": [1], "timeline_position": 1, "role": "host"},
                {"clip_id": 6, "sentence_ids": [2], "timeline_position": 4},
                {"clip_id": 5, "sentence_ids": [42], "timeline_position": 5},
            ],
            "notes": "ok",
        }
        out = normalize_assembled_edl(assembled, self._units(), chunks)
        edl = out["final_edl"]
        assert [c["sentence_ids"] for c in edl] == [[1], [3], [4]]
        assert [c["clip_id"] for c in edl] == [1, 2, 3]
        assert [c["timeline_position"] for c in edl] == [1, 2, 3]
        assert edl[1]["source_in"] == 3.5 and edl[1]["source_out"] == 6.0  # recomputed from sentences
        assert edl[1]["visual_check"] == "眼神直視"  # inherited from chunk candidate
        assert edl[2]["role"] == "offscreen"  # inferred from OFFSCREEN sentence
        assert edl[0]["visual_check"] == "未經視覺確認"
        reasons = [d["reason"] for d in out["assembly_dropped"]]
        assert "cue-only sentences" in reasons and "no valid sentence_ids" in reasons

    def test_parse_assembly_json_tolerates_fences(self):
        raw = "```json\n{\"final_edl\": [], \"notes\": \"n\"}\n```"
        assert parse_assembly_json(raw)["notes"] == "n"

    def test_candidates_are_rendered_per_chunk(self):
        text = format_candidates_for_assembly([
            {"final_edl": [{"clip_id": 1, "role": "host", "segment": "Hook", "sentence_ids": [1],
                            "source_in": 0.0, "source_out": 2.0, "transcript": "開場", "take_selection_reason": "r", "visual_check": "v"}]},
            {"final_edl": []},
        ])
        assert "[區段 1 / 候選 1]" in text and "[區段 2] （沒有候選）" in text


class TestModeCPrompt(unittest.TestCase):
    def test_reference_script_activates_mode_c_without_mode_a_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prompt_file = tmp_path / "video_cut_prompt.md"
            prompt_file.write_text("# base prompt", encoding="utf-8")
            ref = tmp_path / "script.txt"
            ref.write_text("總開場\n\n主持人：第一句\n聲音角色：吐槽\n", encoding="utf-8")
            text = build_prompt([prompt_file], reference_script_path=ref, whisper_units=[])
            assert "Mode C" in text and "Mode A" not in text and "[Script Block" not in text
            assert "聲音角色：吐槽" in text
            # Mode A still works when no reference script is given
            text_a = build_prompt([prompt_file], script_path=ref, whisper_units=[])
            assert "Mode A" in text_a and "[Script Block 01]" in text_a


if __name__ == "__main__":
    unittest.main()

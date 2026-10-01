import unittest
import tests  # noqa: F401
import pytest

from pathlib import Path
import tempfile
from scripts.gemini_client import build_prompt, format_script_blocks_for_prompt
from scripts.transcribe import (
    align_clip_with_whisper,
    coalesce_adjacent_sub_units,
    resolve_clip_sub_units,
)


def _sentence(id_, start, end, text, words):
    return {"id": id_, "start": start, "end": end, "text": text, "words": words}


def _words_for(text, start, dur):
    """均勻切分字級時間戳，供合成測試資料使用。"""
    n = len(text)
    w = dur / n
    return [
        {"word": ch, "start": round(start + i * w, 3), "end": round(start + (i + 1) * w, 3)}
        for i, ch in enumerate(text)
    ]


def _build_units():
    return [
        _sentence(1, 0.0, 1.0, "早安大家好", _words_for("早安大家好", 0.0, 1.0)),
        _sentence(2, 1.2, 2.4, "今天天氣晴朗溫暖", _words_for("今天天氣晴朗溫暖", 1.2, 1.2)),
        _sentence(3, 2.6, 3.6, "謝謝收看再見", _words_for("謝謝收看再見", 2.6, 1.0)),
    ]


class TestAlignClipWithWhisper(unittest.TestCase):
    def test_empty_whisper_units_returns_clip_bounds_and_no_neighbors(self):
        clip = {"source_in": 5.0, "source_out": 8.0}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper([], clip, total_dur=10.0)
        assert (t_first, t_last, prev_end, next_start) == (5.0, 8.0, None, None)

    def test_id_match_without_transcript_uses_word_bounds(self):
        units = _build_units()
        clip = {"start_sentence_id": 2, "end_sentence_id": 2}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert t_first == pytest.approx(1.2)
        assert t_last == pytest.approx(2.4)
        assert prev_end == pytest.approx(1.0)    # 上一句 (id=1) 的結尾
        assert next_start == pytest.approx(2.6)  # 下一句 (id=3) 的起點

    def test_id_match_with_transcript_uses_char_level_diff_alignment(self):
        units = _build_units()
        clip = {"start_sentence_id": 2, "end_sentence_id": 2, "transcript": "今天天氣晴朗溫暖"}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert t_first == pytest.approx(1.2, abs=0.05)
        assert t_last == pytest.approx(2.4, abs=0.05)
        assert prev_end == pytest.approx(1.0)
        assert next_start == pytest.approx(2.6)

    def test_first_sentence_has_no_prev_neighbor(self):
        units = _build_units()
        clip = {"start_sentence_id": 1, "end_sentence_id": 1}
        _, _, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert prev_end is None
        assert next_start == pytest.approx(1.2)

    def test_last_sentence_has_no_next_neighbor(self):
        units = _build_units()
        clip = {"start_sentence_id": 3, "end_sentence_id": 3}
        _, _, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert prev_end == pytest.approx(2.4)
        assert next_start is None

    def test_time_window_fallback_when_no_ids_provided(self):
        units = _build_units()
        clip = {"source_in": 1.1, "source_out": 2.5, "transcript": "今天天氣晴朗溫暖"}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert t_first == pytest.approx(1.2, abs=0.05)
        assert t_last == pytest.approx(2.4, abs=0.05)

    def test_no_candidate_segments_returns_raw_window(self):
        units = _build_units()
        clip = {"source_in": 100.0, "source_out": 105.0, "transcript": "無關內容"}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=200.0)
        assert (t_first, t_last, prev_end, next_start) == (100.0, 105.0, None, None)

    def test_target_speaker_in_point_lock_skips_non_target_words(self):
        """若首個句子的前置單詞為場外非目標人員 (is_target_speaker=False)，t_first 跳至首個主講人單詞"""
        words = [
            {"word": "Action", "start": 1.0, "end": 1.5, "is_target_speaker": False},
            {"word": "嗨", "start": 2.0, "end": 2.3, "is_target_speaker": True},
            {"word": "大家好", "start": 2.3, "end": 3.0, "is_target_speaker": True},
        ]
        sentence = _sentence(1, 1.0, 3.0, "Action 嗨 大家好", words)
        clip = {"start_sentence_id": 1, "end_sentence_id": 1, "transcript": "嗨 大家好"}
        t_first, t_last, _, _ = align_clip_with_whisper([sentence], clip, total_dur=10.0)
        assert t_first == pytest.approx(2.0, abs=0.05)
        assert t_last == pytest.approx(3.0, abs=0.05)

    def test_resolve_clip_sub_units_preserves_selected_sentences_and_splits_dead_air_pauses(self):
        """驗證 resolve_clip_sub_units 尊重 Gemini 選取的 sentence_ids 不做破壞性刪除，並在 >=0.40s 看稿停頓處拆開收緊"""
        units = [
            _sentence(1, 10.0, 12.0, "打開了潘朵拉的盒子對就是WiFi你大概也聽過", _words_for("打開了潘朵拉的盒子對就是WiFi你大概也聽過", 10.0, 2.0)),
            _sentence(2, 12.5, 14.5, "你大概也聽過WiFi訊號撞到人體會出現細微摔", _words_for("你大概也聽過WiFi訊號撞到人體會出現細微摔", 12.5, 2.0)),
            _sentence(3, 16.0, 19.0, "你大概也聽過WiFi訊號撞到人體會出現細微衰減與相位變化", _words_for("你大概也聽過WiFi訊號撞到人體會出現細微衰減與相位變化", 16.0, 3.0)),
        ]
        # Gemini 選取了 Sentence 1 與 Sentence 3 (跳過 NG Sentence 2)，且 Sentence 1 結尾與 Sentence 3 開頭為頂真修辭銜接
        clip = {"sentence_ids": [1, 3], "start_sentence_id": 1, "end_sentence_id": 3, "topic": "HOOK"}
        sub_units = resolve_clip_sub_units(units, clip, total_dur=30.0)

        # Sentence 1 絕不會被誤刪，且因與 Sentence 3 間隔長空白而被拆為 2 個緊湊子片段！
        assert len(sub_units) == 2
        assert sub_units[0]["t_first"] == pytest.approx(10.0, abs=0.05)
        assert sub_units[0]["t_last"] == pytest.approx(12.0, abs=0.05)
        assert sub_units[1]["t_first"] == pytest.approx(16.0, abs=0.05)
        assert sub_units[1]["t_last"] == pytest.approx(19.0, abs=0.05)

    def test_resolve_clip_sub_units_trims_trailing_stumble_via_llm_transcript(self):
        """Verify that when LLM selects Sentence 6 but omits a trailing false start in transcript, t_last snaps to the clean word boundary."""
        words = [
            {"word": "完美的監控工具", "start": 39.16, "end": 42.50},
            {"word": "其實已經在你家裡了", "start": 42.50, "end": 45.20},
            {"word": "那就是WiFi", "start": 45.20, "end": 47.10},
            {"word": "你大概也聽過細微摔", "start": 47.30, "end": 51.60},
        ]
        s6 = _sentence(
            6,
            39.16,
            51.60,
            "完美的監控工具其實已經在你家裡了那就是WiFi你大概也聽過細微摔",
            words,
        )
        clip = {
            "sentence_ids": [6],
            "start_sentence_id": 6,
            "end_sentence_id": 6,
            "transcript": "完美的監控工具其實已經在你家裡了，那就是WiFi。",
            "topic": "Opening",
        }
        sub_units = resolve_clip_sub_units([s6], clip, total_dur=60.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(39.16, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(47.10, abs=0.02)

    def test_coalesce_adjacent_sub_units_merges_contiguous_clips_and_preserves_skipped_id_cuts(self):
        """Verify global cross-clip coalescing merges contiguous Sentence IDs (<0.40s) across clips without merging across skipped NG IDs."""
        expanded = [
            {
                "topic": "Part A",
                "sentence_ids": [6],
                "t_first": 39.16,
                "t_last": 43.20,
                "prev_sentence_end": 34.0,
                "next_sentence_start": 43.35,
                "transcript": "完美的監控工具其實已經在你家裡了",
            },
            {
                "topic": "Part B",
                "sentence_ids": [7],
                "t_first": 43.35,
                "t_last": 47.10,
                "prev_sentence_end": 43.20,
                "next_sentence_start": 47.30,
                "transcript": "那就是WiFi",
            },
            {
                "topic": "Part C (Skipped NG ID 8)",
                "sentence_ids": [9],
                "t_first": 51.60,
                "t_last": 58.00,
                "prev_sentence_end": 51.40,
                "next_sentence_start": None,
                "transcript": "你大概也聽過WiFi訊號撞到人體會出現細微衰減",
            },
        ]
        coalesced = coalesce_adjacent_sub_units(expanded, max_internal_gap=0.40)
        assert len(coalesced) == 2
        assert coalesced[0]["sentence_ids"] == [6, 7]
        assert coalesced[0]["t_first"] == pytest.approx(39.16)
        assert coalesced[0]["t_last"] == pytest.approx(47.10)
        assert coalesced[1]["sentence_ids"] == [9]
        assert coalesced[1]["t_first"] == pytest.approx(51.60)

    def test_dual_mode_prompt_formatting_mode_a_and_mode_b(self):
        """Verify build_prompt generates Mode A numbered script blocks with script_path and Mode B without script_path."""
        prompt_candidates = [
            Path(__file__).resolve().parent.parent
            / "skills"
            / "video-trimmer"
            / "prompts"
            / "video_cut_prompt.md"
        ]
        units = _build_units()

        # Mode B (no script)
        prompt_b = build_prompt(prompt_candidates, script_path=None, whisper_units=units)
        assert "Mode B: Unscripted Intent-Window Take Arbitration (Active)" in prompt_b
        assert "請訂閱，請訂閱，請訂閱，重要的事情要說三遍" in prompt_b

        # Mode A (with script)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write("# Section 1\n第一段講稿內容\n第二段講稿內容\n")
            tmp_path = Path(tmp.name)
        try:
            prompt_a = build_prompt(prompt_candidates, script_path=tmp_path, whisper_units=units)
            assert "Mode A: Monotonic Script-Anchored Take Arbitration (Active)" in prompt_a
            assert "[Script Block 01] 第一段講稿內容" in prompt_a
            assert "[Script Block 02] 第二段講稿內容" in prompt_a
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_tc04_short_transcript_trimming_locks_exact_rightmost_word_bounds(self):
        """TC-04: Verify 1-to-2 char transcripts like '4%' are not rejected by a 3-char floor and lock to the final token."""
        words = [
            {"word": "4%", "start": 250.0, "end": 250.5},
            {"word": "呃", "start": 251.0, "end": 252.0},
            {"word": "對不起重來", "start": 252.2, "end": 255.0},
            {"word": "4%", "start": 257.4, "end": 258.0},
        ]
        s_short = _sentence(38, 250.0, 258.0, "4% 呃 對不起重來 4%", words)
        clip = {
            "sentence_ids": [38],
            "start_sentence_id": 38,
            "end_sentence_id": 38,
            "transcript": "4%",
            "topic": "Short Figure",
        }
        sub_units = resolve_clip_sub_units([s_short], clip, total_dur=300.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(257.4, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(258.0, abs=0.02)
        assert sub_units[0]["trimmed_head"] is True

    def test_tc05_trimmed_boundary_prevents_coalesce_from_resurrecting_stumble(self):
        """TC-05: Verify coalesce_adjacent_sub_units never merges across a boundary where the LLM trimmed a tail stumble."""
        words_s6 = [
            {"word": "完美的監控工具那就是WiFi", "start": 39.16, "end": 47.10},
            {"word": "你大概也聽過細微摔", "start": 47.20, "end": 51.60},
        ]
        words_s7 = [
            {"word": "對就是你家路由器發出的訊號", "start": 47.25, "end": 55.00},
        ]
        s6 = _sentence(6, 39.16, 51.60, "完美的監控工具那就是WiFi你大概也聽過細微摔", words_s6)
        s7 = _sentence(7, 47.25, 55.00, "對就是你家路由器發出的訊號", words_s7)

        clip1 = {"sentence_ids": [6], "transcript": "完美的監控工具那就是WiFi", "topic": "Part 1"}
        clip2 = {"sentence_ids": [7], "transcript": "對就是你家路由器發出的訊號", "topic": "Part 2"}

        su1 = resolve_clip_sub_units([s6, s7], clip1, total_dur=60.0)
        su2 = resolve_clip_sub_units([s6, s7], clip2, total_dur=60.0)
        assert su1[0]["trimmed_tail"] is True
        assert su1[0]["t_last"] == pytest.approx(47.10, abs=0.02)

        # Even though Sentence 6 and 7 are contiguous IDs and gap (47.25 - 47.10 = 0.15s) < 0.40s,
        # coalesce_adjacent_sub_units must NOT merge them because su1.trimmed_tail is True.
        coalesced = coalesce_adjacent_sub_units(su1 + su2, max_internal_gap=0.40)
        assert len(coalesced) == 2
        assert coalesced[0]["t_last"] == pytest.approx(47.10, abs=0.02)
        assert coalesced[1]["t_first"] == pytest.approx(47.25, abs=0.02)

    def test_intra_sentence_repeat_trims_to_rightmost_repetition_and_keeps_gemini_transcript(self):
        """Verify A + A + B intra-sentence repeat trims to the second A + B and preserves Gemini's corrected transcript."""
        words = [
            {"word": "研究者推測", "start": 100.0, "end": 101.5},
            {"word": "BFI本來就是壓縮", "start": 101.5, "end": 103.8},
            {"word": "研究者推測", "start": 104.0, "end": 105.5},
            {"word": "BFI本來就是壓縮", "start": 105.5, "end": 107.8},
            {"word": "後的資訊", "start": 107.8, "end": 109.5},
        ]
        s15 = _sentence(15, 100.0, 109.5, "研究者推測BFI本來就是壓縮研究者推測BFI本來就是壓縮後的資訊", words)
        clip = {
            "sentence_ids": [15],
            "transcript": "研究者推測，BFI 本來就是壓縮後的資訊。",
            "topic": "BFI Explanation",
        }
        sub_units = resolve_clip_sub_units([s15], clip, total_dur=120.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(104.0, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(109.5, abs=0.02)
        assert sub_units[0]["transcript"] == "研究者推測，BFI 本來就是壓縮後的資訊。"
        assert sub_units[0]["whisper_transcript"] == "研究者推測BFI本來就是壓縮後的資訊"





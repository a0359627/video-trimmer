"""acoustic.py 的離線單元測試：以合成音訊驗證方向性與邊界正確性，不追求絕對數值精準。"""

import unittest
import numpy as np
import tests  # noqa: F401
from scripts.acoustic import calculate_clip_cps, refine_speech_bounds_locked

SR = 16000


def _sine(duration, sr=SR, freq=200.0, amplitude=0.3):
    n = int(duration * sr)
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _silence(duration, sr=SR, amplitude=0.0005):
    n = int(duration * sr)
    rng = np.random.default_rng(0)
    return (amplitude * rng.standard_normal(n)).astype(np.float64)


class TestCalculateClipCps(unittest.TestCase):
    def test_empty_transcript_returns_default(self):
        cps, syllables = calculate_clip_cps("", 5.0)
        assert cps == 3.0
        assert syllables == 0

    def test_chinese_transcript_counts_characters(self):
        cps, syllables = calculate_clip_cps("你好世界", 2.0)
        assert syllables == 4
        assert cps == 2.0


class TestRefineSpeechBoundsLocked(unittest.TestCase):
    def test_out_point_extends_past_early_whisper_end_into_trailing_decay(self):
        # Whisper 標記 t_last=0.95 提前於實際語音結束 (1.0s)，且尾端有充分靜音供偵測。
        speech = _sine(1.0)
        silence = _silence(1.5)
        audio = np.concatenate([speech, silence])
        total_dur = len(audio) / SR

        final_in, final_out, cps, in_m, out_m = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=0.95, total_dur=total_dur, transcript="測試片段"
        )
        assert final_out > 0.95
        assert out_m >= 0

    def test_in_margin_is_non_negative_and_before_t_first(self):
        speech = _sine(1.0)
        silence_before = _silence(0.5)
        audio = np.concatenate([silence_before, speech, _silence(0.5)])
        total_dur = len(audio) / SR

        final_in, final_out, cps, in_m, out_m = refine_speech_bounds_locked(
            audio, SR, t_first=0.5, t_last=1.4, total_dur=total_dur, transcript="測試片段"
        )
        assert in_m >= 0
        assert final_in <= 0.5

    def test_forward_search_does_not_cross_next_sentence_start(self):
        """句尾追蹤範圍不可跨越下一句起點：兩句之間僅有短暫（小於 hangover 70ms）停頓時，
        沒有邊界防護會被下一句語音『拉』過去；加上 next_sentence_start 後應被鎖住。"""
        sentence_a = _sine(1.0)
        short_gap = _silence(0.04)  # 40ms < HANGOVER_STEPS(7) * 10ms = 70ms
        sentence_b = _sine(1.0)
        audio = np.concatenate([sentence_a, short_gap, sentence_b, _silence(0.3)])
        total_dur = len(audio) / SR
        next_sentence_start = 1.0 + 0.04

        _, final_out_uncapped, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=1.0, total_dur=total_dur, transcript="句子A",
            next_sentence_start=None,
        )
        assert final_out_uncapped > next_sentence_start, "前提失敗：短暫停頓應足以被下一句語音拉過去"

        _, final_out_capped, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=1.0, total_dur=total_dur, transcript="句子A",
            next_sentence_start=next_sentence_start,
        )
        assert final_out_capped <= next_sentence_start + 1e-9

    def test_backward_search_does_not_cross_prev_sentence_end(self):
        """句首追蹤範圍不可跨越上一句結尾：兩句之間僅有短暫停頓時，
        沒有邊界防護會回溯進上一句語音範圍；加上 prev_sentence_end 後應被鎖住。"""
        sentence_a = _sine(1.0)
        short_gap = _silence(0.04)
        sentence_b = _sine(1.0)
        audio = np.concatenate([sentence_a, short_gap, sentence_b, _silence(0.3)])
        total_dur = len(audio) / SR
        prev_sentence_end = 1.0
        t_first_b = 1.04

        final_in_uncapped, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=t_first_b, t_last=2.0, total_dur=total_dur, transcript="句子B",
            prev_sentence_end=None,
        )
        assert final_in_uncapped < prev_sentence_end, "前提失敗：短暫停頓應足以讓回溯搜尋進入上一句"

        final_in_capped, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=t_first_b, t_last=2.0, total_dur=total_dur, transcript="句子B",
            prev_sentence_end=prev_sentence_end,
        )
        assert final_in_capped >= prev_sentence_end - 1e-9

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

    def test_leading_speech_is_preserved_never_jumped(self):
        """聲學層完整保護句首主語，絕不因後續存在自然換氣停頓而跨越跳切。"""
        subject_speech = _sine(1.0)  # 1.0s 句首主語 (如「說到這裡」)
        pause = _silence(0.50)  # 500ms 自然呼吸停頓
        main_speech = _sine(1.50)  # 後續語句
        audio = np.concatenate([subject_speech, pause, main_speech, _silence(0.5)])
        total_dur = len(audio) / SR

        final_in, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=3.0, total_dur=total_dur, transcript="說到這裡的老故事"
        )
        # 必須完整保留句首主語，起點不得跳躍至 1.0s 之後
        self.assertLessEqual(final_in, 0.15)



def _noise(duration, amplitude, seed=1, sr=SR):
    n = int(duration * sr)
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float64)


def _soft(duration, rms_dbfs, sr=SR, freq=180.0):
    """A sine with the given RMS level (dBFS)."""
    return _sine(duration, sr=sr, freq=freq, amplitude=(10 ** (rms_dbfs / 20)) * np.sqrt(2))


BG = 0.001  # background noise, about -60 dBFS


class TestFirstWordProtection(unittest.TestCase):
    """Cut-in rules around the Whisper first-word start (t_first). Synthetic audio, 16 kHz."""

    def test_soft_first_syllable_connected_to_onset_falls_back_to_word_start(self):
        # Soft first syllable (-38 dBFS, below the speech threshold) runs straight into loud speech.
        audio = np.concatenate([
            _noise(1.0, BG), _soft(0.5, -38.0), _sine(1.5), _noise(1.5, BG, seed=2),
        ])
        final_in, _, _, in_m, _ = refine_speech_bounds_locked(
            audio, SR, t_first=1.0, t_last=3.0, total_dur=len(audio) / SR, transcript="測試片段",
            prev_sentence_end=0.5,
        )
        self.assertLessEqual(final_in, 1.0, "the soft first syllable must stay in the clip")
        self.assertGreaterEqual(final_in, 0.5)

    def test_soft_first_word_with_high_noise_floor_estimate_at_clip_end(self):
        # Loud noise after the clip makes the noise floor (estimated around t_last) high. The soft first
        # word (-30 dBFS) is then below both thresholds, but well above the background near t_first.
        audio = np.concatenate([
            _noise(1.0, BG), _soft(0.6, -30.0), _sine(1.4), _noise(1.6, 0.056, seed=3),
        ])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=1.0, t_last=3.0, total_dur=len(audio) / SR, transcript="測試片段",
            prev_sentence_end=0.5,
        )
        self.assertLessEqual(final_in, 1.0, "the soft first word must stay in the clip")

    def test_early_whisper_first_word_over_silence_keeps_acoustic_onset(self):
        # Whisper starts the first word 0.6 s early, inside silence: keep the acoustic onset.
        audio = np.concatenate([_noise(1.6, BG), _sine(1.4), _noise(1.0, BG, seed=2)])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=1.0, t_last=2.9, total_dur=len(audio) / SR, transcript="測試片段",
            prev_sentence_end=0.5,
        )
        self.assertGreater(final_in, 1.4, "the cut-in must not go back over the silence")
        self.assertLess(final_in, 1.6, "the cut-in must stay before the vocal onset")

    def test_previous_take_residual_at_t_first_then_silence_keeps_acoustic_onset(self):
        # The previous take ends at 1.05 s; Whisper puts the sentence boundary (and t_first) at 1.0 s,
        # inside that residual. Silence follows until the real onset at 1.7 s.
        audio = np.concatenate([
            _noise(0.3, BG), _sine(0.75), _noise(0.65, BG, seed=2), _sine(1.5), _noise(0.8, BG, seed=3),
        ])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=1.0, t_last=3.1, total_dur=len(audio) / SR, transcript="測試片段",
            prev_sentence_end=1.0,
        )
        self.assertGreater(final_in, 1.05, "the residual of the previous take must be cut out")
        self.assertLess(final_in, 1.7, "the cut-in must stay before the vocal onset")

    def test_quiet_residual_separated_by_silence_keeps_acoustic_onset(self):
        # t_first in silence, a short quiet residual (-38 dBFS), clear silence, then the onset.
        audio = np.concatenate([
            _noise(1.1, BG), _soft(0.2, -38.0), _noise(0.6, BG, seed=2), _sine(1.3), _noise(0.8, BG, seed=3),
        ])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=1.0, t_last=3.1, total_dur=len(audio) / SR, transcript="測試片段",
            prev_sentence_end=0.5,
        )
        self.assertGreater(final_in, 1.3, "the residual before the silence gap must be cut out")
        self.assertLess(final_in, 1.9)

    def test_real_first_word_followed_by_pause_is_kept(self):
        # A loud first word starts at t_first (after the previous sentence ended in silence), then a pause.
        # This is not a residual: the sound did not start before the previous sentence end.
        audio = np.concatenate([
            _noise(1.0, BG), _sine(0.2), _noise(0.4, BG, seed=2), _sine(1.4), _noise(0.8, BG, seed=3),
        ])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=1.0, t_last=2.9, total_dur=len(audio) / SR, transcript="測試片段",
            prev_sentence_end=0.5,
        )
        self.assertLessEqual(final_in, 1.0, "the first word must stay in the clip")


class TestOutPointNextSoundLimit(unittest.TestCase):
    """The cut-out limit is the acoustic onset of the next sound, not the (early) Whisper next-sentence start."""

    def test_tail_past_early_whisper_next_start_is_kept(self):
        # The last word really ends at 1.17 s; Whisper ends it at 1.0 s and starts the next sentence there.
        audio = np.concatenate([_sine(1.17), _noise(0.53, BG), _sine(0.8), _noise(0.5, BG, seed=2)])
        _, final_out, _, _, out_m = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=1.0, total_dur=len(audio) / SR, transcript="測試片段",
            next_sentence_start=1.0,
        )
        self.assertGreaterEqual(final_out, 1.17, "the tail after the early Whisper time must be kept")
        self.assertLessEqual(final_out, 1.7 - 0.03 + 1e-9, "the cut-out must stop before the next sound")
        self.assertGreater(out_m, 0.0)

    def test_cut_out_stops_before_next_sound_of_a_dropped_sentence(self):
        # The next sentence (not in the program) starts 0.15 s after the speech end; Whisper marks it early.
        audio = np.concatenate([_sine(1.0), _noise(0.15, BG), _sine(0.8), _noise(0.5, BG, seed=2)])
        _, final_out, _, _, _ = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=0.95, total_dur=len(audio) / SR, transcript="測試片段",
            next_sentence_start=0.95, tail_pad=0.25,
        )
        self.assertGreaterEqual(final_out, 1.0)
        self.assertLessEqual(final_out, 1.15 - 0.03 + 0.011, "neither settle time nor tail pad may enter the next sound")

    def test_tail_pad_uses_next_onset_not_whisper_start(self):
        # Whisper puts the next sentence at the last word end; the next sound is 1.0 s later.
        audio = np.concatenate([_sine(1.0), _noise(1.0, BG), _sine(0.8), _noise(0.5, BG, seed=2)])
        _, out_no_pad, _, _, out_m = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=1.0, total_dur=len(audio) / SR, transcript="測試片段",
            next_sentence_start=1.0,
        )
        _, out_pad, _, _, out_m_pad = refine_speech_bounds_locked(
            audio, SR, t_first=0.0, t_last=1.0, total_dur=len(audio) / SR, transcript="測試片段",
            next_sentence_start=1.0, tail_pad=0.25,
        )
        self.assertAlmostEqual(out_pad - out_no_pad, 0.25, delta=0.011)
        self.assertLess(out_pad, 2.0 - 0.03)
        self.assertEqual(out_m, out_m_pad, "the reported out-margin does not include the tail pad")

    def test_other_microphone_next_line_limits_the_cut_out(self):
        # Show Mode: the own microphone is silent after 1.0 s, the other speaker starts at 1.1 s.
        own = np.concatenate([_sine(1.0), _noise(1.5, BG)])
        other = np.concatenate([_noise(1.1, BG, seed=4), _sine(0.8, freq=260.0), _noise(0.6, BG, seed=5)])
        kwargs = dict(t_first=0.0, t_last=1.0, total_dur=len(own) / SR, transcript="測試片段",
                      next_sentence_start=1.0, tail_pad=0.25)
        _, out_alone, _, _, _ = refine_speech_bounds_locked(own, SR, **kwargs)
        _, out_show, _, _, _ = refine_speech_bounds_locked(own, SR, other_audio=[other], **kwargs)
        self.assertGreater(out_alone, 1.1, "precondition: without the other microphone the pad reaches 1.1 s")
        self.assertLessEqual(out_show, 1.1 - 0.03 + 0.011, "the cut-out must not enter the other speaker's line")
        self.assertGreaterEqual(out_show, 1.0)


class TestShowModeFirstWords(unittest.TestCase):
    """Show Mode: the in-point check also looks at the other microphone."""

    def test_first_words_spoken_on_other_microphone_are_kept(self):
        # The sentence starts with words on the other microphone (0.8 s - 1.5 s), the own voice starts at 1.6 s.
        own = np.concatenate([_noise(1.6, BG), _sine(1.4), _noise(1.0, BG, seed=2)])
        other = np.concatenate([_noise(0.8, BG, seed=3), _sine(0.7, freq=260.0), _noise(2.5, BG, seed=4)])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            own, SR, t_first=1.0, t_last=2.9, total_dur=len(own) / SR, transcript="測試片段",
            prev_sentence_end=0.5, other_audio=[other],
        )
        self.assertLessEqual(final_in, 1.0, "words of this sentence on the other microphone must stay")

    def test_other_speaker_tail_at_t_first_is_not_kept(self):
        # The other speaker's previous line ends 0.1 s after t_first: a tail, not words of this sentence.
        own = np.concatenate([_noise(1.6, BG), _sine(1.4), _noise(1.0, BG, seed=2)])
        other = np.concatenate([_noise(0.4, BG, seed=3), _sine(0.7, freq=260.0), _noise(2.9, BG, seed=4)])
        final_in, _, _, _, _ = refine_speech_bounds_locked(
            own, SR, t_first=1.0, t_last=2.9, total_dur=len(own) / SR, transcript="測試片段",
            prev_sentence_end=1.0, other_audio=[other],
        )
        self.assertGreater(final_in, 1.4, "the other speaker's tail and the silence must be cut out")
        self.assertLess(final_in, 1.6)

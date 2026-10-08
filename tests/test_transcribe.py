import unittest

from scripts.transcribe import (
    merge_whisper_segments_to_sentences,
    normalize_text,
)


def _seg(id_, start, end, text, words=None):
    return {"id": id_, "start": start, "end": end, "text": text, "words": words or []}


class TestNormalizeText(unittest.TestCase):
    def test_strips_punctuation_and_lowercases(self):
        assert normalize_text("Hello, World!") == "helloworld"

    def test_keeps_chinese_characters(self):
        assert normalize_text("你好，世界！") == "你好世界"


class TestMergeWhisperSegmentsToSentences(unittest.TestCase):
    def test_empty_input_returns_empty_list(self):
        assert merge_whisper_segments_to_sentences([]) == []

    def test_large_gap_splits_into_separate_sentences(self):
        segs = [
            _seg(1, 0.0, 1.0, "第一句"),
            _seg(2, 1.8, 2.5, "第二句"),  # gap = 0.8s >= max_gap(0.55s)
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "第一句"
        assert sentences[1]["text"] == "第二句"

    def test_small_gap_without_punctuation_merges_into_one_sentence(self):
        segs = [
            _seg(1, 0.0, 1.0, "第一句"),
            _seg(2, 1.1, 1.8, "接續內容"),  # gap = 0.1s, 無句尾標點閉合
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1
        assert "第一句" in sentences[0]["text"]
        assert "接續內容" in sentences[0]["text"]

    def test_closure_punctuation_with_micro_gap_splits(self):
        segs = [
            _seg(1, 0.0, 1.0, "這是第一句。"),
            _seg(2, 1.25, 2.0, "這是第二句"),  # gap=0.25s >= 0.20s 且前句已標點閉合
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_conjunction_at_next_start_prevents_split(self):
        segs = [
            _seg(1, 0.0, 1.0, "他說要來。"),
            _seg(2, 1.6, 2.0, "但是後來沒有"),  # gap=0.6s >= 0.55s，但下句為連詞開頭且 gap<0.90s
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1

    def test_japanese_chinese_switch_forces_split(self):
        segs = [
            _seg(1, 0.0, 1.0, "這是中文"),
            _seg(2, 1.05, 1.5, "ありがとうございます"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_word_level_timestamps_are_preserved_and_extended(self):
        words_a = [{"word": "你好", "start": 0.0, "end": 0.5}]
        words_b = [{"word": "嗎", "start": 1.1, "end": 1.4}]
        segs = [
            _seg(1, 0.0, 1.0, "你好", words=words_a),
            _seg(2, 1.1, 1.4, "嗎", words=words_b),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1
        assert sentences[0]["words"] == words_a + words_b

    def test_target_speaker_change_forces_split(self):
        """場外人員雜音 (is_target_speaker=False) 與主講人 (is_target_speaker=True) 強制分句"""
        segs = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Action", "is_target_speaker": False, "speaker_id": "SPEAKER_CREW"},
            {"id": 2, "start": 1.1, "end": 2.0, "text": "各位好", "is_target_speaker": True, "speaker_id": "SPEAKER_HOST"},
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["is_target_speaker"] is False
        assert sentences[1]["is_target_speaker"] is True

    def test_speaker_turn_taking_forces_split(self):
        """訪談中說話者輪替 (SPEAKER_00 vs SPEAKER_01) 強制分句"""
        segs = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "請問你怎麼看", "is_target_speaker": True, "speaker_id": "SPEAKER_00"},
            {"id": 2, "start": 1.1, "end": 2.0, "text": "我覺得非常好", "is_target_speaker": True, "speaker_id": "SPEAKER_01"},
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["speaker_id"] == "SPEAKER_00"
        assert sentences[1]["speaker_id"] == "SPEAKER_01"

    def test_long_pause_one_second_forces_split(self):
        """超過 1.0 秒長停頓強制物理切句，即使是同一說話者且帶有連詞"""
        segs = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "前面講得不太順。", "is_target_speaker": True, "speaker_id": "SPEAKER_00"},
            {"id": 2, "start": 2.1, "end": 3.0, "text": "但是重新來過", "is_target_speaker": True, "speaker_id": "SPEAKER_00"},
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_retake_segments_are_never_merged_into_same_sentence(self):
        """講錯重講的相鄰 Whisper segments（即使停頓僅 0.25s）強制保持為獨立 Sentence ID"""
        segs = [
            _seg(1, 0.0, 2.0, "缺點是這套系統運作的前提"),
            _seg(2, 2.25, 4.5, "缺點是這套系統運作的前提是連接"),
            _seg(3, 4.70, 7.2, "缺點是這套系統運作的前提是連接比對"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 3
        assert sentences[0]["text"] == "缺點是這套系統運作的前提"
        assert sentences[1]["text"] == "缺點是這套系統運作的前提是連接"
        assert sentences[2]["text"] == "缺點是這套系統運作的前提是連接比對"

    def test_paused_in_sentence_restart_splits_at_physical_silence(self):
        """單一片段內若存在物理停頓 (>=0.18s) 且停頓後重複開頭前綴，於物理停頓處拆分為獨立 Sentence"""
        words = [
            {"word": "例如", "start": 0.0, "end": 0.2},
            {"word": "你", "start": 0.2, "end": 0.35},
            {"word": "自己", "start": 0.35, "end": 0.60},
            # 物理停頓 0.25s (>= 0.18s) 後重講「例如你...」
            {"word": "例如", "start": 0.85, "end": 1.05},
            {"word": "你", "start": 1.05, "end": 1.20},
            {"word": "帶著", "start": 1.20, "end": 1.45},
            {"word": "自己", "start": 1.45, "end": 1.70},
            {"word": "手機", "start": 1.70, "end": 2.00},
        ]
        segs = [_seg(1, 0.0, 2.0, "例如你自己 例如你帶著自己手機", words=words)]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "例如你自己"
        assert sentences[1]["text"] == "例如你帶著自己手機"




class TestExtractAudio16kMono(unittest.TestCase):
    def _make_source(self, path, seconds=2):
        import subprocess
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=30:d={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path),
            ],
            check=True, capture_output=True,
        )

    def test_extracts_once_then_reuses(self):
        import shutil
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.transcribe import audio_16k_path, extract_audio_16k_mono

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw_footage.mp4"
            self._make_source(src)
            wav = audio_16k_path(tmp, src)
            self.assertEqual(wav.name, "raw_footage_audio_16k.wav")
            extract_audio_16k_mono(src, wav, source_duration=2.0)
            self.assertTrue(wav.exists())
            with patch("scripts.transcribe.subprocess.run") as mock_run:
                extract_audio_16k_mono(src, wav, source_duration=2.0)
                mock_run.assert_not_called()

    def test_short_audio_raises(self):
        import shutil
        import tempfile
        from pathlib import Path
        from scripts.exceptions import FFmpegError
        from scripts.transcribe import audio_16k_path, extract_audio_16k_mono

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw_footage.mp4"
            self._make_source(src)
            with self.assertRaises(FFmpegError):
                extract_audio_16k_mono(src, audio_16k_path(tmp, src), source_duration=10.0)

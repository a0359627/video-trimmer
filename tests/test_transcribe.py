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

    def test_modal_particles_split_sentence(self):
        """句尾帶有語氣詞（啊、嗎、呢等）且長度足夠時，強制與下一子句拆開"""
        segs = [
            _seg(1, 0.0, 2.08, "這算不算是某種超能力啊"),
            _seg(2, 2.08, 3.84, "許多人鼓吹原始人飲食法"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["text"], "這算不算是某種超能力啊")
        self.assertEqual(sentences[1]["text"], "許多人鼓吹原始人飲食法")

    def test_rapid_intra_sentence_prefix_restarts_split(self):
        """單句內若出現多次極短促重講（如連續 3 次以上開頭重講），即使微停頓小於 0.22s 也能切分"""
        words = [
            {"word": "為了", "start": 0.0, "end": 0.2},
            {"word": "追查", "start": 0.2, "end": 0.4},
            {"word": "為了", "start": 0.42, "end": 0.6},
            {"word": "追查", "start": 0.6, "end": 0.8},
            {"word": "這誇張", "start": 0.8, "end": 1.1},
            {"word": "的高考", "start": 1.1, "end": 1.4},
        ]
        segs = [_seg(1, 0.0, 1.4, "為了追查 為了追查 這誇張的高考", words=words)]
        sentences = merge_whisper_segments_to_sentences(segs)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["text"], "為了追查")
        self.assertEqual(sentences[1]["text"], "為了追查這誇張的高考")


    def test_incomplete_number_prevents_split(self):
        segs = [
            _seg(1, 0.0, 1.0, "從原本的0."),
            _seg(2, 1.44, 2.5, "7%到2.7%的比例"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        self.assertEqual(len(sentences), 1)
        self.assertIn("0. 7%", sentences[0]["text"])

    def test_homophone_retake_detection(self):
        from scripts.transcribe import is_fuzzy_prefix_restart
        # "它也成" vs "他也曾"
        self.assertTrue(is_fuzzy_prefix_restart("它也成和口腔健康相關", "他也曾和口腔菌相改變叫差的口腔健康相關"))


class TestTrimCrossClipSeamOverlaps(unittest.TestCase):
    def test_trim_overlapping_tail_text(self):
        from scripts.transcribe import trim_cross_clip_seam_overlaps

        whisper_units = [
            {
                "id": 1,
                "words": [
                    {"word": "這算不算是某種超能力啊", "start": 0.0, "end": 2.0},
                    {"word": "許多人鼓吹原始人飲食法", "start": 2.0, "end": 3.8},
                ],
            },
            {
                "id": 2,
                "words": [
                    {"word": "許多鼓吹原始人飲食法的人", "start": 5.0, "end": 7.5},
                    {"word": "總宣稱我們的基因還停留在石器時代", "start": 7.5, "end": 11.0},
                ],
            },
        ]
        units = [
            {"t_first": 0.0, "t_last": 3.8, "transcript": "這算不算是某種超能力啊 許多人鼓吹原始人飲食法"},
            {"t_first": 5.0, "t_last": 11.0, "transcript": "許多鼓吹原始人飲食法的人 總宣稱我們的基因還停留在石器時代"},
        ]
        trimmed = trim_cross_clip_seam_overlaps(units, whisper_units, min_overlap_chars=4)
        self.assertAlmostEqual(trimmed[0]["t_last"], 2.0, places=1)
        self.assertIn("這算不算是某種超能力啊", trimmed[0]["transcript"])
        self.assertNotIn("原始人飲食法", trimmed[0]["transcript"])

    def test_trim_seam_anchor_short_overlap(self):
        from scripts.transcribe import trim_cross_clip_seam_overlaps

        whisper_units = [
            {
                "id": 18,
                "words": [
                    {"word": "讓冷門外掛變成主流配備", "start": 1695.0, "end": 1699.5},
                    {"word": "這就叫", "start": 1699.5, "end": 1700.5},
                ],
            },
            {
                "id": 19,
                "words": [
                    {"word": "這就叫", "start": 1704.0, "end": 1705.0},
                    {"word": "軟性選擇掃蕩", "start": 1705.0, "end": 1708.0},
                ],
            },
        ]
        units = [
            {"t_first": 1695.0, "t_last": 1700.5, "transcript": "讓冷門外掛變成主流配備這就叫"},
            {"t_first": 1704.0, "t_last": 1708.0, "transcript": "這就叫軟性選擇掃蕩"},
        ]
        trimmed = trim_cross_clip_seam_overlaps(units, whisper_units, min_overlap_chars=4)
        self.assertAlmostEqual(trimmed[0]["t_last"], 1699.5, places=1)
        self.assertEqual(trimmed[0]["transcript"], "讓冷門外掛變成主流配備")

    def test_trim_prefix_duplicate_dropped(self):
        from scripts.transcribe import trim_cross_clip_seam_overlaps

        whisper_units = [
            {
                "id": 28,
                "words": [
                    {"word": "所以", "start": 1888.77, "end": 1889.20},
                    {"word": "說", "start": 1889.20, "end": 1889.45},
                    {"word": "人類", "start": 1889.45, "end": 1889.76},
                ],
            },
            {
                "id": 29,
                "words": [
                    {"word": "所以", "start": 1908.40, "end": 1908.70},
                    {"word": "說", "start": 1908.70, "end": 1908.90},
                    {"word": "人類", "start": 1908.90, "end": 1909.20},
                    {"word": "不是", "start": 1909.20, "end": 1909.50},
                    {"word": "從舊石器時代", "start": 1909.50, "end": 1910.80},
                ],
            },
        ]
        units = [
            {"t_first": 1888.77, "t_last": 1889.76, "transcript": "所以說人類"},
            {"t_first": 1908.40, "t_last": 1910.80, "transcript": "所以說人類不是從舊石器時代"},
        ]
        trimmed = trim_cross_clip_seam_overlaps(units, whisper_units)
        # Unit 28 should be dropped because it is an abandoned prefix of Unit 29
        self.assertEqual(len(trimmed), 1)
        self.assertIn("不是從舊石器時代", trimmed[0]["transcript"])

    def test_trim_intra_clip_opening_stutter(self):
        from scripts.transcribe import trim_cross_clip_seam_overlaps

        whisper_units = [
            {
                "id": 16,
                "words": [
                    {"word": "這些高好被變異啊", "start": 1675.0, "end": 1677.5},
                    {"word": "但這不是有人被馬鈴薯咬到", "start": 1678.0, "end": 1681.0},
                    {"word": "基因突然叮一聲升級", "start": 1681.0, "end": 1683.0},
                    {"word": "這些高好被變異啊", "start": 1683.0, "end": 1685.0},
                    {"word": "早在農業出現以前就存在了", "start": 1685.0, "end": 1688.0},
                ],
            }
        ]
        units = [
            {"t_first": 1660.0, "t_last": 1670.0, "transcript": "前一個鏡頭的句子"},
            {
                "t_first": 1675.0,
                "t_last": 1688.0,
                "transcript": "這些高好被變異啊但這不是有人被馬鈴薯咬到基因突然叮一聲升級這些高好被變異啊早在農業出現以前就存在了",
            },
        ]
        trimmed = trim_cross_clip_seam_overlaps(units, whisper_units)
        self.assertEqual(len(trimmed), 2)
        # The opening stutter "這些高好被變異啊" before "但" should be trimmed away
        self.assertAlmostEqual(trimmed[1]["t_first"], 1678.0, places=1)
        self.assertTrue(trimmed[1]["transcript"].startswith("但這不是"))

    def test_trim_intra_clip_opening_immediate_stutter(self):
        from scripts.transcribe import trim_cross_clip_seam_overlaps

        whisper_units = [
            {
                "id": 40,
                "words": [
                    {"word": "它", "start": 1065.82, "end": 1066.28},
                    {"word": "它", "start": 1066.28, "end": 1066.98},
                    {"word": "規", "start": 1066.98, "end": 1067.00},
                    {"word": "劃", "start": 1067.00, "end": 1070.96},
                ],
            }
        ]
        units = [
            {
                "t_first": 1065.82,
                "t_last": 1070.96,
                "transcript": "它它規劃",
            }
        ]
        trimmed = trim_cross_clip_seam_overlaps(units, whisper_units)
        self.assertEqual(len(trimmed), 1)
        # Should trim past the first stutter word into the repeated word
        self.assertAlmostEqual(trimmed[0]["t_first"], 1066.28, places=2)
        self.assertNotIn("它它", trimmed[0]["transcript"])

    def test_trim_intra_clip_immediate_phrase_restart(self):
        from scripts.transcribe import trim_cross_clip_seam_overlaps

        whisper_units = [
            {
                "id": 2,
                "words": [
                    {"word": "另一家是來自", "start": 21.0, "end": 22.5},
                    {"word": "另一家是來自", "start": 23.0, "end": 24.5},
                    {"word": "新竹的團隊", "start": 24.5, "end": 26.0},
                ],
            }
        ]
        units = [
            {
                "t_first": 21.0,
                "t_last": 26.0,
                "transcript": "另一家是來自另一家是來自新竹的團隊",
            }
        ]
        trimmed = trim_cross_clip_seam_overlaps(units, whisper_units)
        self.assertEqual(len(trimmed), 1)
        # Snaps to the second repeated take at 23.0s
        self.assertAlmostEqual(trimmed[0]["t_first"], 23.0, places=1)
        self.assertEqual(trimmed[0]["transcript"], "另一家是來自新竹的團隊")

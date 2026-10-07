"""Whisper 轉錄、語意句子合併、時間戳工具與文字對齊。"""

import difflib
import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlx_whisper
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

HAS_WHISPER = HAS_MLX_WHISPER or HAS_FASTER_WHISPER


def normalize_text(text):
    return re.sub(r'[^\w一-鿿]', '', text).lower()


def _compute_sentence_speaker(sentence):
    """計算句子內多數說話者與合法主講人狀態"""
    words = sentence.get('words', [])
    if words:
        target_cnt = sum(1 for w in words if w.get('is_target_speaker', True))
        sentence['is_target_speaker'] = (target_cnt >= len(words) / 2.0)
        spk_counts = {}
        for w in words:
            spk = w.get('speaker_id')
            if spk:
                spk_counts[spk] = spk_counts.get(spk, 0) + 1
        if spk_counts:
            sentence['speaker_id'] = max(spk_counts.items(), key=lambda x: x[1])[0]


FILLER_PREFIXES = (
    "那麼", "那", "其實", "然後", "而且",
    "好", "來", "對", "就是", "所以",
    "OK", "ok", "現在", "另外",
)


def normalize_phonetic_homophones(text: str) -> str:
    """Canonicalize frequent Chinese homophone drift in Whisper ASR (他/她/它, 得/地/的, etc.)."""
    t = re.sub(r"[他她]", "它", text)
    t = re.sub(r"[得地]", "的", t)
    t = re.sub(r"作", "做", t)
    t = re.sub(r"再", "在", t)
    t = re.sub(r"它也成", "它也曾", t)
    return t


def strip_filler_words(text: str) -> str:
    """
    Remove common spoken filler prefixes and non-alphanumeric characters.
    ASD-STE100: Use plain normalized string output.
    """
    cleaned = re.sub(r"^[^\w一-鿿]+", "", text or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in FILLER_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip("，,、 　")
                changed = True
                break
    cleaned = normalize_phonetic_homophones(cleaned)
    return re.sub(r"[^\w一-鿿]", "", cleaned).lower()


def is_fuzzy_prefix_restart(
    curr_text: str,
    next_text: str,
    similarity_threshold: float = 0.70,
    min_common_chars: int = 4,
) -> bool:
    """
    Detect if next_text restarts the opening clause of curr_text.
    Tolerate ASR homophone drift, filler particles, and truncated false starts.
    ASD-STE100: Return True when restart detected, else False.
    """
    c_clean = strip_filler_words(curr_text)
    n_clean = strip_filler_words(next_text)

    if len(c_clean) < 3 or len(n_clean) < 3:
        return False

    # Extract opening anchors (window of 6 to 14 characters)
    window_len = min(14, max(6, len(n_clean)))
    c_head = c_clean[:window_len]
    n_head = n_clean[:window_len]

    # 1. Exact common prefix of length >= min_common_chars
    anchor_len = min(min_common_chars, len(c_head), len(n_head))
    if anchor_len >= 3 and c_head[:anchor_len] == n_head[:anchor_len]:
        return True

    # 2. Opening clause containment (strictly compare c_head and n_head)
    # A restart occurs only when the start of n_clean matches the start of c_clean.
    # DO NOT search n_head across the entire c_clean string.
    if len(c_head) >= min_common_chars and c_head[:min_common_chars] in n_head[: min_common_chars + 3]:
        return True
    if len(n_head) >= min_common_chars and n_head[:min_common_chars] in c_head[: min_common_chars + 3]:
        return True

    # 3. Fuzzy SequenceMatcher comparison (handles homophones)
    compare_len = min(len(c_head), len(n_head))
    if compare_len >= 4:
        ratio = difflib.SequenceMatcher(None, c_head[:compare_len], n_head[:compare_len]).ratio()
        if ratio >= similarity_threshold:
            return True

    return False


def is_earlier_sentence_ng_retake(prev_text: str, next_text: str) -> bool:
    """
    判定前一個語意單元 (prev_text) 是否為後一個語意單元 (next_text) 的 NG 重講前綴或半截廢話。
    同時容忍句尾最後 1~2 個字因吃螺絲產生的同音錯字（例如「細微摔」vs「細微衰減」）。
    """
    if is_fuzzy_prefix_restart(prev_text, next_text):
        return True

    p_norm = strip_filler_words(prev_text)
    n_norm = strip_filler_words(next_text)
    if len(p_norm) < 3 or len(n_norm) < 3:
        p_norm = normalize_text(prev_text)
        n_norm = normalize_text(next_text)
    if len(p_norm) < 3 or len(n_norm) < 3:
        return False

    # 1. 開頭前綴完全相同（>= 3 字），且後句長度不短於前句的 75%（代表重新完整重講）
    if p_norm[:3] == n_norm[:3] and len(n_norm) >= int(len(p_norm) * 0.75):
        return True

    # 2. 前句扣除最後 1 個可能吃螺絲的錯字後（長度 >= 4），完整出現在後句中
    if len(p_norm) >= 4 and p_norm[:-1] in n_norm:
        return True

    # 3. 前句與後句開頭區段具有高序列相似度（>= 0.72）
    if len(p_norm) >= 5:
        head_window = n_norm[: len(p_norm) + 4]
        if difflib.SequenceMatcher(None, p_norm, head_window).ratio() >= 0.72:
            return True

    return False


CONJUNCTIONS = ("但是", "而且", "所以", "然而", "如果", "因為", "不過", "雖然", "或是", "或者")


def _split_sentence_on_paused_restarts(
    sentence_data: dict,
    min_pause_sec: float = 0.22,
    min_clause_chars: int = 4,
) -> list[dict]:
    """
    Split a merged sentence at physical intra-sentence pauses, stretched boundary words,
    or rapid prefix restarts (stutters/retakes where the speaker restarts the clause opening).

    ASD-STE100:
    Evaluate word-level acoustic gaps and durations.
    When a physical pause (gap >= min_pause_sec) or stretched onset separates two clauses,
    or when the speaker immediately restarts the opening clause prefix (Prefix Anchor Restart),
    split into independent sentence units so the LLM can arbitrate each take individually.
    Keep conjunction prefixes attached when the gap is below 0.85s unless an exact restart occurs.
    """
    words = sentence_data.get("words", [])
    if len(words) < 6:
        return [sentence_data]

    split_word_indices = []
    seg_start_idx = 0
    n = len(words)

    for j in range(1, n):
        gap = float(words[j].get("start", 0.0)) - float(words[j - 1].get("end", 0.0))
        word_dur = float(words[j].get("end", 0.0)) - float(words[j].get("start", 0.0))
        w_chars = max(1, len(normalize_text(words[j].get("word", ""))))
        is_stretched_onset = (word_dur >= 1.20) and ((word_dur / w_chars) >= 0.45)

        prev_text = "".join(w.get("word", "") for w in words[seg_start_idx:j]).strip()
        rem_text = "".join(w.get("word", "") for w in words[j:]).strip()
        prev_norm = normalize_text(prev_text)
        rem_norm = normalize_text(rem_text)

        if len(prev_norm) < 1 or len(rem_norm) < 2:
            continue

        # Immediate stutter repetition at clause start (e.g. "它 它", "這 這")
        is_opening_stutter = (
            (j - seg_start_idx == 2)
            and len(prev_norm) <= 4
            and (normalize_text(words[seg_start_idx].get("word", "")) == normalize_text(words[seg_start_idx + 1].get("word", "")))
            and len(rem_norm) >= 3
        )

        # Prefix Anchor Restart Detection (rapid retakes where speaker repeats the opening phrase)
        curr_opening = normalize_text("".join(w.get("word", "") for w in words[seg_start_idx : seg_start_idx + 4]))
        cand_opening = normalize_text("".join(w.get("word", "") for w in words[j : j + 4]))
        cand_stripped = strip_filler_words(cand_opening)

        has_prefix_restart = False
        if len(curr_opening) >= 3:
            if len(cand_opening) >= 3 and cand_opening.startswith(curr_opening[:3]):
                has_prefix_restart = True
            elif len(cand_stripped) >= 3 and cand_stripped.startswith(curr_opening[:3]):
                has_prefix_restart = True

        has_physical_pause = (gap >= min_pause_sec) or is_stretched_onset

        # Incomplete syntax protection (decimals, digits, prepositions)
        is_prev_incomplete_num = bool(re.search(r'\d+\.$', prev_text.strip())) or bool(re.search(r'\d+$', prev_text.strip()) and re.match(r'^[0-9.%]', rem_text.strip()))
        is_prev_incomplete_prep = any(prev_text.strip().endswith(p) for p in ("與", "和", "或", "在", "的", "把", "將", "向", "對", "從", "自"))

        should_split = False
        if is_opening_stutter:
            should_split = True
        elif has_prefix_restart and (j - seg_start_idx >= 2 and len(prev_norm) >= 3):
            should_split = True
        elif has_physical_pause and (
            (len(prev_norm) >= min_clause_chars and len(rem_norm) >= min_clause_chars)
            or (gap >= 0.80 and len(prev_norm) >= 3 and len(rem_norm) >= 3)
        ):
            should_split = True

        if should_split and not has_prefix_restart and not is_opening_stutter and gap < 1.20 and word_dur < 1.50:
            if is_prev_incomplete_num or is_prev_incomplete_prep:
                should_split = False

        if should_split and not has_prefix_restart and not is_opening_stutter and gap < 0.85 and any(rem_text.startswith(c) for c in CONJUNCTIONS):
            should_split = False

        if should_split:
            split_word_indices.append(j)
            seg_start_idx = j

    if not split_word_indices:
        return [sentence_data]

    sub_sentences = []
    boundaries = [0] + split_word_indices + [n]
    for idx in range(len(boundaries) - 1):
        w_slice = words[boundaries[idx] : boundaries[idx + 1]]
        if not w_slice:
            continue
        sub_s = {
            "id": 0,
            "start": w_slice[0]["start"],
            "end": w_slice[-1]["end"],
            "text": "".join(w.get("word", "") for w in w_slice).strip(),
            "words": list(w_slice),
            "orig_segment_ids": list(sentence_data.get("orig_segment_ids", [])),
            "speaker_id": sentence_data.get("speaker_id", "SPEAKER_00"),
            "is_target_speaker": sentence_data.get("is_target_speaker", True),
        }
        _compute_sentence_speaker(sub_s)
        sub_sentences.append(sub_s)

    return sub_sentences


def merge_whisper_segments_to_sentences(segments, max_gap=0.20, max_sentence_dur=8.0):
    """
    Group Whisper segments into clause-level sentences using pure acoustic pauses and punctuation.

    ASD-STE100:
    Segment speech by physical breath pauses, punctuation closure, and speaker turns.
    Do not compare text strings to guess retakes.
    Preserve conjunction attachment to prevent isolated transition tokens.
    """
    if not segments:
        return []

    CLOSURE_PUNCT = ("。", "！", "？", "!", "?", "……", "...")
    MODAL_PARTICLES = ("啊", "嗎", "吧", "呢", "呀", "啦", "囉", "喔", "哦", "耶")

    raw_sentences = []
    curr = None
    prev_s = None

    for s_idx, s in enumerate(segments):
        s_text = s["text"].strip()
        s_start = s["start"]
        s_end = s["end"]
        s_words = s.get("words", [])
        s_is_target = s.get("is_target_speaker", True)
        s_spk = s.get("speaker_id", "SPEAKER_00")
        s_next = segments[s_idx + 1] if s_idx + 1 < len(segments) else None

        if curr is None:
            curr = {
                "id": 1,
                "start": s_start,
                "end": s_end,
                "text": s_text,
                "words": list(s_words),
                "orig_segment_ids": [s["id"]],
                "speaker_id": s_spk,
                "is_target_speaker": s_is_target,
            }
            _compute_sentence_speaker(curr)
            prev_s = s
            continue

        gap = s_start - curr["end"]
        curr_text = curr["text"].strip()
        curr_norm_len = len(normalize_text(curr_text))
        s_norm_len = len(normalize_text(s_text))
        dur = s_end - curr["start"]

        first_w_stretched = False
        if s_words:
            fw = s_words[0]
            fw_dur = float(fw.get("end", 0.0)) - float(fw.get("start", 0.0))
            fw_chars = max(1, len(normalize_text(fw.get("word", ""))))
            first_w_stretched = (fw_dur >= 1.20) and ((fw_dur / fw_chars) >= 0.45)

        # Rule 0: Retake restart detection against previous segment or current sentence head
        is_restart = False
        if prev_s and (is_fuzzy_prefix_restart(prev_s["text"], s_text) or is_earlier_sentence_ng_retake(prev_s["text"], s_text)):
            is_restart = True
        elif is_fuzzy_prefix_restart(curr_text, s_text) or is_earlier_sentence_ng_retake(curr_text, s_text):
            is_restart = True

        # Forward retake check: if s is followed by a restart of itself in s_next, split before s so s does not pollute curr
        is_s_retake_head = False
        if s_next and (is_fuzzy_prefix_restart(s_text, s_next["text"]) or is_earlier_sentence_ng_retake(s_text, s_next["text"])):
            is_s_retake_head = True

        should_split = False
        if is_restart or is_s_retake_head:
            should_split = True
        # Rule 1: Physical respiration pause or stretched onset when current clause has sufficient length
        elif gap >= 0.55 or ((gap >= max_gap or first_w_stretched) and curr_norm_len >= 5):
            should_split = True
        # Rule 2: Clause duration ceiling reached with micro-pause
        elif dur >= max_sentence_dur and gap >= 0.12:
            should_split = True
        # Rule 3: Closure punctuation with micro-pause
        elif any(curr_text.endswith(p) for p in CLOSURE_PUNCT) and gap >= 0.15:
            should_split = True
        # Rule 3b: Modal particles marking sentence question / exclamatory closure
        elif any(curr_text.endswith(p) for p in MODAL_PARTICLES) and curr_norm_len >= 5:
            should_split = True
        # Rule 3c: Natural clause boundary when accumulated clause exceeds 3.8s and 18 chars without conjunction
        elif (dur >= 3.8 and curr_norm_len >= 18) and s_norm_len >= 6 and not any(s_text.startswith(c) for c in CONJUNCTIONS):
            should_split = True

        # Rule 4: Slate cue or script language switch (e.g. alphanumeric slate vs Japanese/Chinese)
        is_curr_slate = bool(re.match(r"^[A-Za-z0-9\s\-_]+$", curr_text))
        is_s_japanese = bool(re.search(r"[぀-ヿ]", s_text))
        is_curr_japanese = bool(re.search(r"[぀-ヿ]", curr_text))
        if is_curr_slate and not bool(re.match(r"^[A-Za-z0-9\s\-_]+$", s_text)):
            should_split = True
        elif is_curr_japanese != is_s_japanese:
            should_split = True

        # Rule 5: Target speaker validity transition (crew audio vs target host)
        if curr.get("is_target_speaker", True) != s_is_target:
            should_split = True

        # Rule 6: Speaker turn-taking transition
        curr_spk = curr.get("speaker_id")
        if s_spk and curr_spk and curr_spk != s_spk:
            should_split = True

        # Rule 7: Unconditional split on pauses >= 1.0s
        if gap >= 1.0:
            should_split = True

        # Rule 8: Conjunction attachment protection (commit 0b579f4 invariant)
        if should_split and not is_restart and not is_s_retake_head and gap < 0.85 and dur < max_sentence_dur * 1.4:
            if curr.get("is_target_speaker", True) == s_is_target and curr.get("speaker_id") == s_spk:
                if any(s_text.startswith(c) for c in CONJUNCTIONS):
                    should_split = False

        # Rule 8b: Incomplete syntax protection (decimals, digits, prepositions)
        is_curr_incomplete_num = bool(re.search(r'\d+\.$', curr_text)) or bool(re.search(r'\d+$', curr_text) and re.match(r'^[0-9.%]', s_text))
        is_curr_incomplete_prep = any(curr_text.endswith(p) for p in ("與", "和", "或", "在", "的", "把", "將", "向", "對", "從", "自"))
        if should_split and not is_restart and not is_s_retake_head and gap < 1.2:
            if is_curr_incomplete_num or is_curr_incomplete_prep:
                should_split = False

        if should_split:
            _compute_sentence_speaker(curr)
            raw_sentences.append(curr)
            curr = {
                "id": len(raw_sentences) + 1,
                "start": s_start,
                "end": s_end,
                "text": s_text,
                "words": list(s_words),
                "orig_segment_ids": [s["id"]],
                "speaker_id": s_spk,
                "is_target_speaker": s_is_target,
            }
            _compute_sentence_speaker(curr)
        else:
            curr["end"] = s_end
            curr["text"] = (
                curr["text"] + " " + s_text
                if not curr["text"].endswith((" ", "，", "。"))
                else curr["text"] + s_text
            )
            curr["words"].extend(s_words)
            curr["orig_segment_ids"].append(s["id"])
            _compute_sentence_speaker(curr)

        prev_s = s

    if curr is not None:
        _compute_sentence_speaker(curr)
        raw_sentences.append(curr)

    final_sentences = []
    for sent in raw_sentences:
        for sub_sent in _split_sentence_on_paused_restarts(sent):
            sub_sent["id"] = len(final_sentences) + 1
            final_sentences.append(sub_sent)

    return final_sentences


def transcribe_video_whisper(
    video_path,
    out_json_path,
    model_name="small"
):
    """
    第一階段：微觀聲學時間戳對齊 (Word-level Ground Truth)
    在調用大模型之前，先在本地以 Whisper 生成帶有毫秒級字級時間戳的結構化劇本。
    返回: (raw_segments, merged_sentences)
    """
    out_json = Path(out_json_path)
    sentences_json = out_json.parent / f"{video_path.stem}_whisper_sentences.json"

    raw_results = []
    if out_json.exists():
        logger.info("載入既有之 Whisper 聲學時間戳: %s", out_json.name)
        with open(out_json, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
    elif HAS_WHISPER:
        temp_wav = out_json.parent / f"temp_{video_path.stem}_whisper.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if HAS_MLX_WHISPER:
            logger.info("本地調用 mlx-whisper (%s) [Apple Silicon Metal GPU 原生加速] 進行微觀字級時間戳轉錄...", model_name)
            mlx_repo = f"mlx-community/whisper-{model_name}-mlx"
            res = mlx_whisper.transcribe(str(temp_wav), path_or_hf_repo=mlx_repo, word_timestamps=True, language="zh")
            for i, s in enumerate(res.get("segments", [])):
                words_data = []
                for w in s.get("words", []):
                    words_data.append({
                        "word": w["word"].strip(),
                        "start": round(float(w["start"]), 2),
                        "end": round(float(w["end"]), 2)
                    })
                raw_results.append({
                    "id": i + 1,
                    "start": round(float(s["start"]), 2),
                    "end": round(float(s["end"]), 2),
                    "text": s["text"].strip(),
                    "words": words_data
                })
        else:
            logger.info("本地調用 faster-whisper (%s) 進行微觀字級時間戳轉錄...", model_name)
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(temp_wav), word_timestamps=True, language="zh")

            for i, s in enumerate(segments):
                words_data = []
                if s.words:
                    for w in s.words:
                        words_data.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2)
                        })
                raw_results.append({
                    "id": i + 1,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": s.text.strip(),
                    "words": words_data
                })

        temp_wav.unlink(missing_ok=True)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        logger.info("Whisper 轉錄完畢，共解析出 %d 個高精度時間戳片段。", len(raw_results))
    else:
        logger.info("[提示] 未安裝 mlx-whisper 或 faster-whisper，跳過本地字級轉錄。")
        return [], []

    # 執行語意句子合併（每次由 raw_results 即時計算，確保最新斷句與重講拆分規則生效）
    sentences = merge_whisper_segments_to_sentences(raw_results)
    with open(sentences_json, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    logger.info("語意合併完成：從 %d 個聲學碎片濃縮為 %d 個完整語意句子！", len(raw_results), len(sentences))

    return raw_results, sentences


def format_whisper_transcript_for_prompt(whisper_sentences):
    """將 Whisper 合併後的語意句子格式化為 Gemini 提示詞專用的文字剪輯清單"""
    lines = [
        "\n---",
        "## Whisper 語意句子時間戳劇本 (Ground-Truth Semantic Sentences)",
        "以下是由本地微觀語音模型（Whisper）對本片轉錄、並依據自然換氣停頓與重講邊界拆分之「句子清單」。",
        "每個 Sentence 均為一個候選表達單元，包含物理起訖秒數（start -> end）。",
        "【多模態發言人日誌審查與 Last Take Wins 鐵則】：",
        "請務必結合視訊畫面中主講人的嘴型、眼神方向與肢體動作：",
        "1. 僅挑選由畫面中央「目標主講人面對鏡頭正式發表」（target_host）的有效句子；",
        "2. 任何由場外小幫手/導播喊出的口令（如 Action、報幕代號 CDA82/CTA-S2、CDA84 等，主講人嘴巴閉著或在等待）屬於無效場外音，嚴禁選入 final_edl！",
        "3. 錄影空檔中主講人偏離鏡頭與工作人員之閒聊、自我檢討（如「這段不理想」），屬於 blooper/chatter，亦嚴禁選入 final_edl！",
        "4. 若相鄰或相近的多個 Sentence 講述相同或重複開頭的台詞（講者吃螺絲重錄），請務必在 `sentence_ids` 中徹底剔除前面的 NG 句，只保留最後一次完整流暢的 Sentence ID！",
        "5. 孤立短截殘句與棄用假開頭（Abandoned False Starts，話講到一半卡住放棄、只有連詞或片語、隨後重講者，如「雖然研究強烈支持」、「所以說人類」、「健康相關他也曾和口腔菌相」），100% 為無效廢料，絕對嚴禁選入 final_edl！",
        "請在輸出 final_edl 時，於 `sentence_ids` 明確列出保留的 Sentence ID 陣列（並填寫 `start_sentence_id` 與 `end_sentence_id`），",
        "並將 `source_in` 與 `source_out` 對齊起訖句子之時間：\n"
    ]
    for s in whisper_sentences:
        lines.append(f"[Sentence ID: {s['id']:3d}] {s['start']:6.2f}s -> {s['end']:6.2f}s | {s['text']}")
    return "\n".join(lines)


def _find_index(units, item):
    for i, u in enumerate(units):
        if u is item:
            return i
    return None


def _neighbor_bounds(whisper_units, first_item, last_item):
    """回傳 (prev_sentence_end, next_sentence_start)：相鄰句子的邊界，供聲學搜尋範圍上限使用。"""
    idx_first = _find_index(whisper_units, first_item)
    idx_last = _find_index(whisper_units, last_item)
    prev_end = None
    next_start = None
    if idx_first is not None and idx_first > 0:
        prev_end = whisper_units[idx_first - 1]["end"]
    if idx_last is not None and idx_last + 1 < len(whisper_units):
        next_start = whisper_units[idx_last + 1]["start"]
    return prev_end, next_start


def align_clip_with_whisper(whisper_units, clip_data, total_dur):
    """
    比照字幕方式：嚴格以 Whisper 物理時間為準，不猜測、不更動字尾！
    優先採用 start_sentence_id / end_sentence_id 或 start_segment_id / end_segment_id，
    次之採用文本子字串比對，再次之採用時間窗。
    In 點鎖定：嚴格跳過所有 is_target_speaker == False 的非目標人聲，直擊主講人真聲開口。

    回傳: (t_first, t_last, prev_sentence_end, next_sentence_start)
    prev_sentence_end / next_sentence_start 為相鄰 Whisper 句子的邊界（無相鄰句子時為 None），
    供 acoustic.refine_speech_bounds_locked 做結構性搜尋範圍上限。
    """
    if not whisper_units:
        return clip_data.get("source_in", 0), clip_data.get("source_out", total_dur), None, None

    start_id = clip_data.get("start_sentence_id")
    if start_id is None:
        start_id = clip_data.get("start_segment_id")

    end_id = clip_data.get("end_sentence_id")
    if end_id is None:
        end_id = clip_data.get("end_segment_id")

    # 1. 優先使用 ID
    if start_id is not None and end_id is not None:
        matched = [s for s in whisper_units if start_id <= s["id"] <= end_id]
        if matched:
            prev_end, next_start = _neighbor_bounds(whisper_units, matched[0], matched[-1])
            transcript = clip_data.get("transcript", "") or clip_data.get("content", "")
            # 若提供了明確台詞文字，且句子內有詳細字級時間戳，精確對齊至台詞起訖字
            if transcript:
                matched_words = []
                for s in matched:
                    matched_words.extend(s.get("words", []))
                char_timeline = []
                for w in matched_words:
                    w_norm = normalize_text(w["word"])
                    if not w_norm:
                        continue
                    w_is_target = w.get("is_target_speaker", True)
                    w_dur = (w["end"] - w["start"]) / len(w_norm)
                    for idx, ch in enumerate(w_norm):
                        ch_start = w["start"] + idx * w_dur
                        char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3), w_is_target))

                tgt_norm = normalize_text(transcript)
                whisper_str = ''.join([item[0] for item in char_timeline])
                if tgt_norm and whisper_str and char_timeline:
                    matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
                    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
                    if blocks and sum(b.size for b in blocks) >= max(2, len(tgt_norm) * 0.4):
                        # 聲紋鎖定：尋找匹配區間中第一個屬於合法主講人的字元
                        t_first = None
                        for b in blocks:
                            for c_idx in range(b.b, b.b + b.size):
                                if char_timeline[c_idx][3]:  # is_target_speaker == True
                                    t_first = char_timeline[c_idx][1]
                                    break
                            if t_first is not None:
                                break
                        if t_first is None:
                            t_first = char_timeline[blocks[0].b][1]

                        last_b = blocks[-1]
                        t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
                        return t_first, t_last, prev_end, next_start

            first_words = matched[0].get("words", [])
            last_words = matched[-1].get("words", [])
            # 聲紋鎖定：若首句開頭有場外雜音單詞，跳至第一個合法主講人單詞
            target_words = [w for w in first_words if w.get("is_target_speaker", True)]
            if target_words:
                t_first = target_words[0]["start"]
            elif first_words:
                t_first = first_words[0]["start"]
            else:
                t_first = matched[0]["start"]

            t_last = last_words[-1]["end"] if last_words else matched[-1]["end"]
            return t_first, t_last, prev_end, next_start

    raw_in = clip_data.get("source_in", 0)
    raw_out = clip_data.get("source_out", total_dur)
    transcript = clip_data.get("transcript", "")

    # 2. 透過時間窗附近篩選
    cand_segs = [s for s in whisper_units if not (s["end"] < raw_in - 2.5 or s["start"] > raw_out + 2.5)]
    if not cand_segs:
        return raw_in, raw_out, None, None

    prev_end, next_start = _neighbor_bounds(whisper_units, cand_segs[0], cand_segs[-1])

    # 3. 逐字元建立時間線並做 SequenceMatcher
    char_timeline = []
    for s in cand_segs:
        words = s.get("words", [])
        if words:
            for w in words:
                w_norm = normalize_text(w["word"])
                if not w_norm:
                    continue
                w_is_target = w.get("is_target_speaker", True)
                w_dur = (w["end"] - w["start"]) / len(w_norm)
                for idx, ch in enumerate(w_norm):
                    ch_start = w["start"] + idx * w_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3), w_is_target))
        else:
            s_norm = normalize_text(s["text"])
            if s_norm:
                s_is_target = s.get("is_target_speaker", True)
                s_dur = (s["end"] - s["start"]) / len(s_norm)
                for idx, ch in enumerate(s_norm):
                    ch_start = s["start"] + idx * s_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + s_dur, 3), s_is_target))

    tgt_norm = normalize_text(transcript)
    whisper_str = ''.join([item[0] for item in char_timeline])

    if tgt_norm and whisper_str and char_timeline:
        matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
        if blocks:
            # 聲紋鎖定：尋找第一個屬於合法主講人的字元
            t_first = None
            for b in blocks:
                for c_idx in range(b.b, b.b + b.size):
                    if char_timeline[c_idx][3]:  # is_target_speaker == True
                        t_first = char_timeline[c_idx][1]
                        break
                if t_first is not None:
                    break
            if t_first is None:
                t_first = char_timeline[blocks[0].b][1]

            last_b = blocks[-1]
            t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
            return t_first, t_last, prev_end, next_start

    # 若匹配落空，返回重疊 segment 範圍
    return cand_segs[0]["start"], cand_segs[-1]["end"], prev_end, next_start


def filter_ng_retake_sentences(sentences: list[dict], max_lookahead: int = 3, max_time_span: float = 25.0) -> list[dict]:
    """
    在候選句子序列中執行確定性的「Last Take Wins」過濾：
    若句子 S_i 與後續緊鄰的句子 S_j (j > i) 構成重講關係（S_i 為 NG 前綴或吃螺絲半截話），
    自動於自然句子邊界處剔除 S_i，僅保留最後一次完整句子。
    """
    if len(sentences) <= 1:
        return list(sentences)

    kept = []
    n = len(sentences)
    for i, s_curr in enumerate(sentences):
        is_ng = False
        limit = min(n, i + 1 + max_lookahead)
        for j in range(i + 1, limit):
            s_later = sentences[j]
            if float(s_later.get("start", 0.0)) - float(s_curr.get("end", 0.0)) > max_time_span:
                break
            if is_earlier_sentence_ng_retake(s_curr.get("text", ""), s_later.get("text", "")):
                logger.info(
                    "    [Last-Take-Wins 過濾] 剔除 NG 重複句 (Sentence %s: '%s') -> 保留正式句 (Sentence %s: '%s')",
                    s_curr.get("id"),
                    s_curr.get("text", ""),
                    s_later.get("id"),
                    s_later.get("text", ""),
                )
                is_ng = True
                break
        if not is_ng:
            kept.append(s_curr)
    return kept


def _trim_matched_words_by_transcript(matched_sentences: list[dict], transcript: str) -> list[dict]:
    """
    Trim leading and trailing words of matched sentences to match the LLM-selected transcript.

    ASD-STE100:
    Align the LLM transcript string with the Whisper word timeline.
    When the LLM omits a leading or trailing stumble from a selected sentence,
    trim the word list to the matched character boundaries.
    """
    if not transcript or not matched_sentences:
        return matched_sentences

    char_timeline = []
    for s_idx, s in enumerate(matched_sentences):
        for w_idx, w in enumerate(s.get("words", [])):
            w_norm = normalize_text(w.get("word", ""))
            if not w_norm:
                continue
            for ch in w_norm:
                char_timeline.append((s_idx, w_idx, ch))

    if not char_timeline:
        return matched_sentences

    tgt_norm = normalize_text(transcript)
    whisper_str = "".join(item[2] for item in char_timeline)
    if not tgt_norm or not whisper_str:
        return matched_sentences

    matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return matched_sentences

    matched_chars = sum(b.size for b in blocks)
    if matched_chars < max(3, len(tgt_norm) * 0.60):
        return matched_sentences

    first_char_pos = blocks[0].b
    last_char_pos = min(len(char_timeline) - 1, blocks[-1].b + blocks[-1].size - 1)

    # If full match across boundaries without any trimming needed
    if first_char_pos == 0 and last_char_pos == len(char_timeline) - 1:
        return matched_sentences

    first_s_idx, first_w_idx, _ = char_timeline[first_char_pos]
    last_s_idx, last_w_idx, _ = char_timeline[last_char_pos]

    trimmed = []
    for s_idx in range(first_s_idx, last_s_idx + 1):
        orig_s = matched_sentences[s_idx]
        words = list(orig_s.get("words", []))
        if not words:
            trimmed.append(orig_s)
            continue

        w_start = first_w_idx if s_idx == first_s_idx else 0
        # Protect tail of sentence: allow trimming tail stumble only if:
        # 1) It consists entirely of known blooper tokens (e.g. "好", "不好意思", "對"), OR
        # 2) There is a distinct acoustic pause (gap >= 0.18s) separating it from the sentence.
        # Continuous phonation (gap < 0.18s) must never be chopped mid-word due to LLM transcript omissions.
        omitted_tail = words[last_w_idx + 1 :]
        BLOOPER_TAIL_WORDS = {
            "好", "對", "嗯", "啊", "ok", "OK", "噯", "喂", "卡", "謝謝", "不好意思", "再一次", "重來", "幹嘛呢"
        }
        is_all_blooper = bool(omitted_tail) and all(
            normalize_text(w.get("word", "")).lower() in BLOOPER_TAIL_WORDS for w in omitted_tail
        )
        tail_gap = (omitted_tail[0]["start"] - words[last_w_idx]["end"]) if omitted_tail else 0.0
        if s_idx == last_s_idx and (is_all_blooper or tail_gap >= 0.18):
            w_end = last_w_idx + 1
        else:
            w_end = len(words)
        sliced_words = words[w_start:w_end]
        if not sliced_words:
            continue

        s_copy = dict(orig_s)
        s_copy["words"] = sliced_words
        s_copy["start"] = sliced_words[0]["start"]
        s_copy["end"] = sliced_words[-1]["end"]
        s_copy["text"] = "".join(w.get("word", "") for w in sliced_words).strip()
        trimmed.append(s_copy)

    return trimmed if trimmed else matched_sentences


def _split_sentence_words_by_internal_gap(sentence: dict, max_word_gap: float = 0.45) -> list[dict]:
    """
    Split a sentence into compact speech chunks when internal word gaps exceed max_word_gap.

    ASD-STE100:
    Divide a sentence at physical silence gaps (>= max_word_gap) so the acoustic engine
    can tighten dead air inside a single sentence.
    """
    words = [w for w in sentence.get("words", []) if w.get("is_target_speaker", True)]
    if not words:
        words = list(sentence.get("words", []))
    if not words:
        return [{
            "sentence_id": sentence.get("id", 0),
            "t_first": float(sentence.get("start", 0.0)),
            "t_last": float(sentence.get("end", 0.0)),
            "text": sentence.get("text", "").strip(),
        }]

    chunks = []
    cur_words = [words[0]]
    for w in words[1:]:
        gap = float(w.get("start", 0.0)) - float(cur_words[-1].get("end", 0.0))
        if gap >= max_word_gap:
            chunks.append({
                "sentence_id": sentence.get("id", 0),
                "t_first": float(cur_words[0]["start"]),
                "t_last": float(cur_words[-1]["end"]),
                "text": "".join(x.get("word", "") for x in cur_words).strip(),
            })
            cur_words = [w]
        else:
            cur_words.append(w)

    if cur_words:
        chunks.append({
            "sentence_id": sentence.get("id", 0),
            "t_first": float(cur_words[0]["start"]),
            "t_last": float(cur_words[-1]["end"]),
            "text": "".join(x.get("word", "") for x in cur_words).strip(),
        })

    return chunks


def resolve_clip_sub_units(
    whisper_units: list[dict],
    clip_data: dict,
    total_dur: float,
    max_internal_gap: float = 0.65,
    max_word_gap: float = 0.70,
) -> list[dict]:
    """
    Resolve a single EDL clip into one or more compact speech sub-units.

    ASD-STE100:
    1. Read `sentence_ids` first, or fall back to `start_sentence_id`..`end_sentence_id`.
    2. Align word boundaries to `clip_data["transcript"]` when the LLM trims a boundary stumble.
    3. Preserve all LLM-selected sentences without Python string-similarity deletion.
    4. Coalesce consecutive sentence IDs when the physical gap is below max_internal_gap (0.40s),
       and split when an ID is skipped or when a pause is >= max_internal_gap.
    """
    if not whisper_units:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "sentence_ids": [],
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": clip_data.get("transcript", "") or clip_data.get("content", ""),
        }]

    matched = []
    raw_sent_ids = clip_data.get("sentence_ids")
    if isinstance(raw_sent_ids, list) and raw_sent_ids:
        valid_ids = {int(x) for x in raw_sent_ids if isinstance(x, (int, str)) and str(x).isdigit()}
        matched = [s for s in whisper_units if s.get("id") in valid_ids]

    if not matched:
        start_id = clip_data.get("start_sentence_id")
        if start_id is None:
            start_id = clip_data.get("start_segment_id")
        end_id = clip_data.get("end_sentence_id")
        if end_id is None:
            end_id = clip_data.get("end_segment_id")
        if start_id is not None and end_id is not None:
            matched = [s for s in whisper_units if start_id <= s.get("id", 0) <= end_id]

    if not matched and ("source_in" in clip_data or "source_out" in clip_data):
        s_in = float(clip_data.get("source_in", 0.0))
        s_out = float(clip_data.get("source_out", total_dur))
        matched = [
            u for u in whisper_units
            if u.get("is_target_speaker", True)
            and float(u.get("end", 0.0)) >= s_in - 0.20
            and float(u.get("start", 0.0)) <= s_out + 0.20
        ]

    if not matched:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "sentence_ids": [],
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": clip_data.get("transcript", "") or clip_data.get("content", ""),
        }]

    # Filter out off-screen crew sentences (is_target_speaker == False)
    target_matched = [s for s in matched if s.get("is_target_speaker", True)]
    if target_matched:
        matched = target_matched

    # Trim leading/trailing words if LLM transcript excluded a boundary stumble
    clip_transcript = clip_data.get("transcript", "") or clip_data.get("content", "")
    if clip_transcript:
        matched = _trim_matched_words_by_transcript(matched, clip_transcript)

    # Split sentences at internal dead-air pauses (>= max_word_gap)
    fine_chunks = []
    for s in matched:
        fine_chunks.extend(_split_sentence_words_by_internal_gap(s, max_word_gap=max_word_gap))

    if not fine_chunks:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "sentence_ids": [],
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": clip_transcript,
        }]

    # Coalesce consecutive chunks when gap < max_internal_gap and no Sentence ID was skipped
    grouped_units = []
    cur_group = {
        "sentence_ids": [fine_chunks[0]["sentence_id"]],
        "t_first": fine_chunks[0]["t_first"],
        "t_last": fine_chunks[0]["t_last"],
        "texts": [fine_chunks[0]["text"]],
    }

    for ch in fine_chunks[1:]:
        gap = ch["t_first"] - cur_group["t_last"]
        prev_sid = cur_group["sentence_ids"][-1]
        cur_sid = ch["sentence_id"]
        skipped_sentence = cur_sid > prev_sid + 1

        if gap >= max_internal_gap or skipped_sentence:
            grouped_units.append(cur_group)
            cur_group = {
                "sentence_ids": [cur_sid],
                "t_first": ch["t_first"],
                "t_last": ch["t_last"],
                "texts": [ch["text"]],
            }
        else:
            if cur_sid not in cur_group["sentence_ids"]:
                cur_group["sentence_ids"].append(cur_sid)
            cur_group["t_last"] = ch["t_last"]
            cur_group["texts"].append(ch["text"])

    grouped_units.append(cur_group)

    all_bounds = [(float(u.get("start", 0.0)), float(u.get("end", 0.0))) for u in whisper_units]
    sub_units = []
    for idx, g in enumerate(grouped_units):
        t_f = g["t_first"]
        t_l = g["t_last"]

        prev_ends = [b_end for (_, b_end) in all_bounds if b_end <= t_f + 0.02]
        if idx > 0:
            prev_ends.append(grouped_units[idx - 1]["t_last"])
        prev_end = max(prev_ends) if prev_ends else None

        next_starts = [b_start for (b_start, _) in all_bounds if b_start >= t_l - 0.02]
        if idx + 1 < len(grouped_units):
            next_starts.append(grouped_units[idx + 1]["t_first"])
        next_start = min(next_starts) if next_starts else None

        sub_text = " ".join(t for t in g["texts"] if t).strip()
        sub_units.append({
            "sentence_ids": list(g["sentence_ids"]),
            "t_first": t_f,
            "t_last": t_l,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": sub_text or clip_transcript,
        })

    return sub_units


def coalesce_adjacent_sub_units(
    expanded_units: list[dict],
    max_internal_gap: float = 0.65,
) -> list[dict]:
    """
    Coalesce consecutive sub-units across clip boundaries when no sentence ID is skipped
    and the physical pause is below max_internal_gap.

    ASD-STE100:
    Prevent artificial jump-cuts and margin collisions when the LLM splits a continuous take
    across two adjacent EDL clip entries.
    """
    if len(expanded_units) <= 1:
        return list(expanded_units)

    coalesced = [dict(expanded_units[0])]
    for unit in expanded_units[1:]:
        prev = coalesced[-1]
        prev_sids = prev.get("sentence_ids") or []
        curr_sids = unit.get("sentence_ids") or []

        gap = float(unit.get("t_first", 0.0)) - float(prev.get("t_last", 0.0))
        is_contiguous_sid = (
            bool(prev_sids)
            and bool(curr_sids)
            and 0 <= (int(curr_sids[0]) - int(prev_sids[-1])) <= 1
        )

        if is_contiguous_sid and 0.0 <= gap < max_internal_gap:
            merged_sids = list(prev_sids)
            for sid in curr_sids:
                if sid not in merged_sids:
                    merged_sids.append(sid)
            prev["sentence_ids"] = merged_sids
            prev["t_last"] = float(unit["t_last"])
            prev["next_sentence_start"] = unit.get("next_sentence_start")
            prev_text = (prev.get("transcript") or "").strip()
            curr_text = (unit.get("transcript") or "").strip()
            if curr_text and curr_text not in prev_text:
                prev["transcript"] = f"{prev_text} {curr_text}".strip()
        else:
            coalesced.append(dict(unit))

    return coalesced


def calculate_active_script_window(
    whisper_units: list[dict],
    script_text: str,
    total_duration: float,
    padding_seconds: float = 2.0,
) -> tuple[float, float, list[dict]]:
    """
    Align spoken clauses from a reference script against Whisper sentences to locate
    the active recording window in a multi-episode or long camera roll.

    ASD-STE100:
    This function matches script clauses with transcript sentences.
    It returns the start time, end time, and sentences inside the active window.
    """
    if not whisper_units or not script_text or not script_text.strip():
        return 0.0, total_duration, whisper_units

    # Extract meaningful spoken clauses from script (ignore markdown headers and stage brackets)
    raw_lines = re.split(r"[\n。！？!?；;]+", script_text)
    script_clauses = []
    for line in raw_lines:
        cleaned = re.sub(r"^\s*[#*>\-\d.]+\s*", "", line)
        cleaned = re.sub(r"（.*?）|\(.*?\)|【.*?】|\[.*?\]", "", cleaned)
        norm = normalize_text(cleaned)
        if len(norm) >= 6:
            script_clauses.append(norm)

    if not script_clauses:
        return 0.0, total_duration, whisper_units

    matched_unit_indices = set()
    matched_clause_count = 0

    norm_units = [(idx, normalize_text(u.get("text", ""))) for idx, u in enumerate(whisper_units)]

    for clause in script_clauses:
        clause_matched = False
        for idx, u_norm in norm_units:
            if len(u_norm) < 4:
                continue
            if clause in u_norm or (len(u_norm) >= 6 and u_norm in clause):
                matched_unit_indices.add(idx)
                clause_matched = True
                continue
            matcher = difflib.SequenceMatcher(None, clause, u_norm)
            longest = matcher.find_longest_match(0, len(clause), 0, len(u_norm))
            if longest.size >= 6 or (min(len(clause), len(u_norm)) >= 8 and matcher.ratio() >= 0.58):
                matched_unit_indices.add(idx)
                clause_matched = True
        if clause_matched:
            matched_clause_count += 1

    # Require at least 40% script clause coverage to activate windowing
    coverage = matched_clause_count / max(1, len(script_clauses))
    if not matched_unit_indices or coverage < 0.40:
        return 0.0, total_duration, whisper_units

    sorted_indices = sorted(matched_unit_indices)
    first_idx = sorted_indices[0]
    last_idx = sorted_indices[-1]

    win_start = max(0.0, round(float(whisper_units[first_idx].get("start", 0.0)) - padding_seconds, 2))
    win_end = min(total_duration, round(float(whisper_units[last_idx].get("end", total_duration)) + padding_seconds, 2))

    # If the window already covers >= 85% of the video, keep the full video
    if (win_end - win_start) >= total_duration * 0.85:
        return 0.0, total_duration, whisper_units

    filtered_units = whisper_units[first_idx : last_idx + 1]
    return win_start, win_end, (filtered_units or whisper_units)


def trim_cross_clip_seam_overlaps(
    units: list[dict],
    whisper_units: list[dict],
    min_overlap_chars: int = 4,
) -> list[dict]:
    """
    Trim acoustic tail-to-head text overlaps and head stutters across adjacent EDL cut seams.

    ASD-STE100:
    Compare adjacent clips across cut seams.
    1. If the tail of clip A shares a matching phrase (>= min_overlap_chars) with the head of clip B,
       trim clip A's t_last/source_out to end before the duplicated words.
    2. If clip B begins with an immediate stutter (repeating its own opening words),
       trim clip B's t_first/source_in to start at the final take of that opening phrase.
    This eliminates repeated speech across cuts and aligns with the editor's cuts.
    """
    if not units or not whisper_units:
        return units

    all_words = sorted(
        [w for s in whisper_units for w in s.get("words", []) if "start" in w and "end" in w],
        key=lambda x: float(x["start"]),
    )
    if not all_words:
        return units

    # 1. Intra-clip opening stutter check: clean each clip's head if it repeats opening words
    for u in units:
        in_t = float(u.get("t_first", u.get("source_in", 0.0)))
        out_t = float(u.get("t_last", u.get("source_out", 0.0)))
        w = [w for w in all_words if in_t - 0.05 <= float(w["start"]) and float(w["end"]) <= out_t + 0.05]
        if not w:
            continue
        u_words = w[:25]
        norm_words = [normalize_text(w.get("word", "")) for w in u_words]
        full_u_norm = "".join(norm_words)

        trimmed_start_w_idx = None

        # Case A: Immediate single-word or repeated phrase stutter at opening (e.g. "它 它", "這 這")
        if len(norm_words) >= 2 and norm_words[0] and norm_words[0] == norm_words[1]:
            trimmed_start_w_idx = 1

        # Case B: Prefix string repeated within the head window
        if trimmed_start_w_idx is None and len(full_u_norm) >= 6:
            for p_w_cnt in range(1, min(5, len(u_words))):
                prefix_str = "".join(norm_words[:p_w_cnt])
                if len(prefix_str) < 3:
                    continue
                rep_pos = full_u_norm.find(prefix_str, len(prefix_str))
                if rep_pos != -1 and rep_pos <= 60:
                    # If repeated immediately or within 6 chars (immediate restart of opening clause)
                    if rep_pos <= len(prefix_str) + 6:
                        c_acc = 0
                        for w_i, nw in enumerate(norm_words):
                            if c_acc >= rep_pos:
                                trimmed_start_w_idx = w_i
                                break
                            c_acc += len(nw)
                    else:
                        # Stutter abandoned opening before another clause
                        trimmed_start_w_idx = p_w_cnt
                    break

        # Case C: Short 1~2 char opening token followed by large dead air (>= 0.8s) or stretch (>= 1.5s)
        if trimmed_start_w_idx is None and len(u_words) >= 3:
            for k in (1, 2):
                if k < len(u_words):
                    gap_after = float(u_words[k].get("start", 0.0)) - float(u_words[k - 1].get("end", 0.0))
                    dur_k = float(u_words[k].get("end", 0.0)) - float(u_words[k].get("start", 0.0))
                    prefix_len = sum(len(norm_words[i]) for i in range(k))
                    if (gap_after >= 0.80 or dur_k >= 1.50) and prefix_len <= 3:
                        trimmed_start_w_idx = k if gap_after >= 0.80 else k + 1
                        break

        if trimmed_start_w_idx is not None and trimmed_start_w_idx > 0 and trimmed_start_w_idx < len(w):
            trimmed_w = w[trimmed_start_w_idx:]
            if trimmed_w:
                new_in = float(trimmed_w[0]["start"])
                if "t_first" in u:
                    u["t_first"] = new_in
                if "source_in" in u:
                    u["source_in"] = round(new_in, 2)
                    if "source_out" in u:
                        u["duration"] = round(u["source_out"] - u["source_in"], 2)
                u["transcript"] = "".join(w.get("word", "") for w in trimmed_w).strip()
        elif len(w) >= 6:
            curr_opening = normalize_text("".join(w.get("word", "") for w in w[0:4]))
            cand_opening = normalize_text("".join(w.get("word", "") for w in w[4:8]))
            if len(curr_opening) >= 3 and len(cand_opening) >= 3 and cand_opening.startswith(curr_opening[:3]):
                trimmed_w = w[4:]
                if trimmed_w:
                    new_in = float(trimmed_w[0]["start"])
                    if "t_first" in u:
                        u["t_first"] = new_in
                    if "source_in" in u:
                        u["source_in"] = round(new_in, 2)
                        if "source_out" in u:
                            u["duration"] = round(u["source_out"] - u["source_in"], 2)
                    u["transcript"] = "".join(w.get("word", "") for w in trimmed_w).strip()

    # 2. Cross-clip seam overlaps and prefix duplicate check
    dropped_indices = set()

    for i in range(len(units) - 1):
        if i in dropped_indices:
            continue
        u1 = units[i]
        u2 = units[i + 1]

        t1_raw = u1.get("transcript", "")
        t2_raw = u2.get("transcript", "")
        t1_norm = normalize_text(t1_raw)
        t2_norm = normalize_text(t2_raw)

        # Seam Prefix Duplicate Check:
        # If u1 is an abandoned prefix of u2 (e.g. u1 = "所以說人類", u2 = "所以說人類不是從..."):
        if len(t1_norm) >= 2 and len(t2_norm) >= len(t1_norm):
            prefix_check_len = min(len(t1_norm), 8)
            if t2_norm.startswith(t1_norm[:prefix_check_len]) and len(t1_norm) <= 10:
                logger.info(
                    "    [Seam Overlap 剔除] 發現片段 %d ('%s') 為下一鏡頭 ('%s') 之棄用假開頭，自動剔除！",
                    i + 1,
                    t1_raw,
                    t2_raw[:25],
                )
                dropped_indices.add(i)
                continue

        in_1 = float(u1.get("t_first", u1.get("source_in", 0.0)))
        out_1 = float(u1.get("t_last", u1.get("source_out", 0.0)))
        in_2 = float(u2.get("t_first", u2.get("source_in", 0.0)))
        out_2 = float(u2.get("t_last", u2.get("source_out", 0.0)))

        w1 = [w for w in all_words if in_1 - 0.05 <= float(w["start"]) and float(w["end"]) <= out_1 + 0.05]
        w2 = [w for w in all_words if in_2 - 0.05 <= float(w["start"]) and float(w["end"]) <= out_2 + 0.05]

        if not w1 or not w2:
            continue

        tail_w = w1[-20:]
        head_w = w2[:20]

        tail_norm = normalize_text("".join(w.get("word", "") for w in tail_w))
        head_norm = normalize_text("".join(w.get("word", "") for w in head_w))

        if len(tail_norm) >= 2 and len(head_norm) >= 2:
            matcher = difflib.SequenceMatcher(None, tail_norm, head_norm)
            match = matcher.find_longest_match(0, len(tail_norm), 0, len(head_norm))

            # Anchor condition: match reaches the end of tail_norm and is located at the head of head_norm
            is_seam_anchor = (
                match.size >= 2
                and (match.a + match.size >= len(tail_norm) - 1)
                and (match.b <= 3)
            )

            if match.size >= min_overlap_chars or is_seam_anchor:
                cut_char_pos = match.a
                # Expand backward if shared demonstrative / particle prefix (e.g. '這' vs '這個')
                if cut_char_pos > 0 and match.b > 0 and tail_norm[cut_char_pos - 1] == head_norm[match.b - 1]:
                    cut_char_pos -= 1
                elif cut_char_pos > 0 and tail_norm[cut_char_pos - 1] in ("這", "那") and any(c in head_norm[:match.b] for c in ("這", "那")):
                    cut_char_pos -= 1

                char_count = 0
                cut_idx = None
                for idx, w in enumerate(tail_w):
                    wn = normalize_text(w.get("word", ""))
                    if not wn:
                        continue
                    if char_count + len(wn) > cut_char_pos:
                        cut_idx = idx
                        break
                    char_count += len(wn)

                if cut_idx == 0 and len(w1) <= 12:
                    dropped_indices.add(i)
                    continue
                elif cut_idx is not None and cut_idx > 0:
                    kept_tail = tail_w[:cut_idx]
                    trimmed_w1 = w1[:-len(tail_w)] + kept_tail
                    if trimmed_w1:
                        new_out = float(trimmed_w1[-1]["end"])
                        if "t_last" in u1:
                            u1["t_last"] = new_out
                        if "source_out" in u1:
                            u1["source_out"] = round(new_out, 2)
                            if "source_in" in u1:
                                u1["duration"] = round(u1["source_out"] - u1["source_in"], 2)
                        u1["next_sentence_start"] = in_2
                        u1["transcript"] = "".join(w.get("word", "") for w in trimmed_w1).strip()

    if dropped_indices:
        units = [u for idx, u in enumerate(units) if idx not in dropped_indices]

    return units


def _are_sentences_retake_related(text_a: str, text_b: str) -> bool:
    """
    Check if two sentences belong to the same retake cluster.
    """
    if is_earlier_sentence_ng_retake(text_a, text_b) or is_earlier_sentence_ng_retake(text_b, text_a):
        return True
    na = normalize_text(text_a)
    nb = normalize_text(text_b)
    if len(na) >= 4 and len(nb) >= 4 and na[:4] == nb[:4]:
        return True
    return False


def _backtrack_before_retake_cluster(
    whisper_units: list[dict],
    candidate_idx: int,
    min_idx: int,
) -> int:
    """
    Backtrack the split index so that a retake cluster is never split across two chunks.

    ASD-STE100:
    This function checks if sentences after the split repeat sentences before the split.
    If a repeat exists, it moves the split index backward before the first take.
    """
    idx = candidate_idx
    while idx > min_idx:
        after_slice = whisper_units[idx + 1 : min(len(whisper_units), idx + 5)]
        before_start = max(min_idx, idx - 4)
        earliest_overlap_idx = None

        for b_idx in range(before_start, idx + 1):
            b_text = whisper_units[b_idx].get("text", "")
            for a_unit in after_slice:
                a_text = a_unit.get("text", "")
                if _are_sentences_retake_related(b_text, a_text):
                    earliest_overlap_idx = b_idx
                    break
            if earliest_overlap_idx is not None:
                break

        if earliest_overlap_idx is not None and earliest_overlap_idx > min_idx:
            # Step back to the sentence immediately before Take 1 of the retake cluster
            idx = earliest_overlap_idx - 1
        else:
            break

    return idx


def detect_chunk_boundaries(
    whisper_units: list[dict],
    total_duration: float,
    target_chunk_duration: float = 480.0,
    min_pause_duration: float = 2.0,
    max_chunk_duration: float = 660.0,
    window_start: float | None = None,
) -> list[tuple[float, float, list[dict]]]:
    """
    Split the timeline into logical chunks at natural speech pauses (>= 2.0s).
    Backtracks before any retake cluster so Take 1 and Take N stay in the same chunk.

    ASD-STE100:
    This function finds pauses between spoken sentences.
    It groups sentences into chunks between 6 and 11 minutes long.

    Returns:
        List of tuples: (chunk_start_sec, chunk_end_sec, chunk_sentences)
    """
    if not whisper_units:
        start_t = 0.0 if window_start is None else window_start
        return [(start_t, total_duration, [])]

    first_unit_start = float(whisper_units[0].get("start", 0.0))
    chunk_start = window_start if window_start is not None else (0.0 if first_unit_start < 30.0 else max(0.0, first_unit_start - 5.0))
    effective_span = total_duration - chunk_start

    if effective_span <= max_chunk_duration:
        return [(chunk_start, total_duration, whisper_units)]

    chunks = []
    start_idx = 0
    n_units = len(whisper_units)

    while start_idx < n_units:
        # Check if remaining duration fits in a single final chunk
        if (total_duration - chunk_start) <= max_chunk_duration:
            chunks.append((chunk_start, total_duration, whisper_units[start_idx:]))
            break

        split_idx = None
        min_backtrack_idx = start_idx

        for i in range(start_idx, n_units - 1):
            s_curr_end = float(whisper_units[i].get("end", 0.0))
            chunk_elapsed = s_curr_end - chunk_start
            if chunk_elapsed >= target_chunk_duration * 0.65 and min_backtrack_idx == start_idx:
                min_backtrack_idx = i

            s_next_start = float(whisper_units[i + 1].get("start", 0.0))
            pause = s_next_start - s_curr_end

            if chunk_elapsed >= target_chunk_duration and pause >= min_pause_duration:
                safe_idx = _backtrack_before_retake_cluster(whisper_units, i, min_backtrack_idx)
                safe_pause = float(whisper_units[safe_idx + 1].get("start", 0.0)) - float(whisper_units[safe_idx].get("end", 0.0))
                if safe_idx == i or safe_pause >= 0.40:
                    split_idx = safe_idx
                    break

            if chunk_elapsed >= max_chunk_duration:
                # Backtrack within [min_backtrack_idx, i] to find the largest non-retake pause
                best_idx = i
                best_score = -1.0
                search_start = max(start_idx + 1, min_backtrack_idx)
                for cand in range(search_start, i + 1):
                    safe_cand = _backtrack_before_retake_cluster(whisper_units, cand, search_start)
                    cand_pause = (
                        float(whisper_units[safe_cand + 1].get("start", 0.0))
                        - float(whisper_units[safe_cand].get("end", 0.0))
                    )
                    if cand_pause > best_score:
                        best_score = cand_pause
                        best_idx = safe_cand
                split_idx = best_idx
                break

        if split_idx is None or split_idx >= n_units - 1:
            chunks.append((chunk_start, total_duration, whisper_units[start_idx:]))
            break

        s_curr_end = float(whisper_units[split_idx].get("end", 0.0))
        s_next_start = float(whisper_units[split_idx + 1].get("start", 0.0))
        pause = max(0.0, s_next_start - s_curr_end)
        split_point = round(s_curr_end + (pause / 2.0), 2)

        chunk_sentences = whisper_units[start_idx : split_idx + 1]
        chunks.append((chunk_start, split_point, chunk_sentences))

        chunk_start = split_point
        start_idx = split_idx + 1

    return chunks



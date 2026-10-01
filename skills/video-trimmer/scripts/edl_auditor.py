"""
EDL quality auditor, deterministic timing sanitizer, and surgical micro-window repair.

ASD-STE100:
1. Detect missing script blocks, unanchored clips, and adjacent retake candidates before rendering.
2. Re-scan only the affected 15-to-90-second video micro-window via Vertex AI VideoMetadata.
3. Sanitize physical clip boundaries to eliminate overlaps and micro-clips (< 0.45s).
4. Generate an 8-dimension rough-cut audit report (.json and .md) with a top-level agent_verdict.
"""

from __future__ import annotations

import difflib
import logging
import math
import re
from pathlib import Path

from .acoustic import calculate_clip_cps
from .transcribe import _are_sentences_retake_related, normalize_text

logger = logging.getLogger(__name__)

MAX_MICRO_WINDOW_REPAIRS = 5
MIN_MICRO_WINDOW_SEC = 15.0
MAX_MICRO_WINDOW_SEC = 90.0


# Document metadata prefixes that represent non-spoken headers (entire line is skipped)
SCRIPT_METADATA_LINE_PREFIXES = (
    "標題：", "標題:", "主題：", "主題:", "大綱：", "大綱:",
    "備註：", "備註:", "說明：", "說明:", "專案：", "專案:",
    "title:", "subject:", "topic:", "outline:", "author:",
    "date:", "duration:", "scene:", "note:", "notes:",
)

# Section label prefixes where the prefix itself is non-spoken, but spoken text may follow on the same line
SCRIPT_SECTION_LABEL_PREFIXES = (
    "內文：", "內文:", "講稿：", "講稿:", "腳本：", "腳本:",
    "台詞：", "台詞:", "旁白：", "旁白:", "口白：", "口白:",
    "body:", "script:", "narration:", "voiceover:", "vo:",
)


def extract_script_blocks(script_text: str | None) -> list[dict]:
    """
    Parse raw script text into sequential numbered script block dicts.

    ASD-STE100:
    1. Strip YAML frontmatter (--- ... ---), markdown headings (#), and horizontal rules.
    2. Exclude non-spoken metadata lines (such as Title, Subject, Outline, Notes).
    3. Strip section label prefixes (such as Body: or Script:) and stage directions.
    4. Extract numbered spoken script blocks for Prompt and Auditor alignment.
    """
    if not script_text or not script_text.strip():
        return []

    raw_lines = script_text.splitlines()
    blocks: list[dict] = []
    block_idx = 1
    in_frontmatter = False

    for line_num, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if line_num == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped in ("---", "..."):
                in_frontmatter = False
            continue

        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[-=*_]{3,}$", stripped):
            continue

        # Remove leading markdown list/quote markers before checking metadata prefixes
        unmarked = re.sub(r"^\s*[>*\-\d.]+\s*", "", stripped).strip()
        unmarked_lower = unmarked.lower()

        if any(unmarked_lower.startswith(p) for p in SCRIPT_METADATA_LINE_PREFIXES):
            continue

        for label_prefix in SCRIPT_SECTION_LABEL_PREFIXES:
            if unmarked_lower.startswith(label_prefix):
                unmarked = unmarked[len(label_prefix):].strip()
                break

        if not unmarked:
            continue

        # Strip inline stage directions / camera cues in brackets
        cleaned = re.sub(r"（.*?）|\(.*?\)|【.*?】|\[.*?\]", "", unmarked).strip()
        norm = normalize_text(cleaned)
        if len(norm) >= 4:
            blocks.append({
                "block_id": block_idx,
                "label": f"[Script Block {block_idx:02d}]",
                "raw_text": unmarked,
                "norm_text": norm,
            })
            block_idx += 1

    return blocks


# Backward-compatible alias for internal callers and tests
_extract_script_blocks = extract_script_blocks


def _script_block_coverage_score(block_norm: str, target_norm: str) -> float:
    """
    Compute the fraction of `block_norm` characters covered in order within `target_norm`.

    ASD-STE100:
    Always use `len(block_norm)` as the denominator so short aborted NG fragments
    cannot produce high similarity scores, while complete takes with minor ASR homophone
    differences achieve high coverage.
    """
    if not block_norm or not target_norm:
        return 0.0
    if block_norm in target_norm:
        return 1.0

    block_len = len(block_norm)
    target_len = len(target_norm)
    min_match_size = 2 if any("\u4e00" <= ch <= "\u9fff" for ch in block_norm) else 3

    def _window_coverage(win_text: str) -> float:
        if not win_text:
            return 0.0
        if block_norm in win_text:
            return 1.0
        matcher = difflib.SequenceMatcher(None, block_norm, win_text, autojunk=False)
        matched_chars = sum(
            m.size for m in matcher.get_matching_blocks() if m.size >= min_match_size
        )
        return min(1.0, matched_chars / block_len)

    if target_len <= block_len * 2:
        return _window_coverage(target_norm)

    win_size = min(target_len, max(block_len + 24, int(block_len * 1.8)))
    stride = max(4, block_len // 3)
    best_score = 0.0

    # Evaluate sliding windows across target_norm
    for start_idx in range(0, max(1, target_len - win_size + 1), stride):
        score = _window_coverage(target_norm[start_idx : start_idx + win_size])
        if score > best_score:
            best_score = score
            if best_score >= 0.95:
                return best_score

    # Evaluate tail window and anchor window around longest contiguous match
    tail_score = _window_coverage(target_norm[max(0, target_len - win_size) :])
    best_score = max(best_score, tail_score)

    global_matcher = difflib.SequenceMatcher(None, block_norm, target_norm, autojunk=False)
    longest = global_matcher.find_longest_match(0, block_len, 0, target_len)
    if longest.size >= min_match_size:
        anchor_start = max(0, longest.b - longest.a - 12)
        anchor_end = min(target_len, anchor_start + win_size)
        best_score = max(best_score, _window_coverage(target_norm[anchor_start:anchor_end]))

    return best_score


def _match_text_similarity(norm_a: str, norm_b: str) -> float:
    """Return script coverage score of norm_a inside norm_b."""
    return _script_block_coverage_score(norm_a, norm_b)


def _unit_overlaps_script_block(block_norm: str, unit_norm: str) -> bool:
    """
    Return True if a single Whisper sentence unit belongs to a script block.

    ASD-STE100:
    Accept a sentence unit when it covers >= 35% of the script block or when >= 65%
    of the sentence unit (at least 5 chars) matches a contiguous clause of the script block.
    """
    if not block_norm or not unit_norm:
        return False
    if _script_block_coverage_score(block_norm, unit_norm) >= 0.35:
        return True
    if len(unit_norm) >= 5 and _script_block_coverage_score(unit_norm, block_norm) >= 0.65:
        return True
    return False


def _resolve_clip_sentence_ids(clip: dict, whisper_units: list[dict]) -> list[int]:
    """Return sorted valid Whisper sentence IDs referenced by a clip."""
    if not whisper_units:
        return []
    valid_unit_ids = {int(u["id"]) for u in whisper_units if "id" in u}
    raw_ids = clip.get("sentence_ids")
    resolved = []
    if isinstance(raw_ids, list) and raw_ids:
        for x in raw_ids:
            if isinstance(x, (int, str)) and str(x).isdigit() and int(x) in valid_unit_ids:
                resolved.append(int(x))
    if not resolved:
        start_id = clip.get("start_sentence_id") or clip.get("start_segment_id")
        end_id = clip.get("end_sentence_id") or clip.get("end_segment_id")
        if isinstance(start_id, int) and isinstance(end_id, int):
            resolved = [uid for uid in sorted(valid_unit_ids) if start_id <= uid <= end_id]
    return sorted(set(resolved))


def _clip_time_bounds(clip: dict, unit_by_id: dict[int, dict], whisper_units: list[dict]) -> tuple[float, float]:
    """Return physical (start, end) time bounds for a clip using resolved sentence IDs when available."""
    sids = _resolve_clip_sentence_ids(clip, whisper_units)
    t_in = float(clip.get("source_in", 0.0))
    t_out = float(clip.get("source_out", t_in))
    if sids:
        first_u = unit_by_id.get(sids[0])
        last_u = unit_by_id.get(sids[-1])
        if first_u is not None:
            t_in = float(first_u.get("start", t_in))
        if last_u is not None:
            t_out = float(last_u.get("end", t_out))
    return t_in, max(t_in, t_out)


def _clip_dual_texts(clip: dict, unit_by_id: dict[int, dict], whisper_units: list[dict]) -> tuple[str, str]:
    """
    Return `(llm_transcript, whisper_transcript)` for a clip so anomaly detectors and
    quality auditors can inspect both LLM-cleaned text and physical Whisper words.
    """
    llm_text = (clip.get("transcript") or "").strip()
    w_text = (clip.get("whisper_transcript") or "").strip()
    if not w_text:
        sids = _resolve_clip_sentence_ids(clip, whisper_units)
        t_first = clip.get("t_first")
        t_last = clip.get("t_last")
        if sids:
            collected_words: list[str] = []
            used_word_bounds = False
            for sid in sids:
                u = unit_by_id.get(sid)
                if not u:
                    continue
                u_words = u.get("words") or []
                if u_words and t_first is not None and t_last is not None:
                    used_word_bounds = True
                    for w in u_words:
                        w_start = float(w.get("start", 0.0))
                        w_end = float(w.get("end", 0.0))
                        if w_end >= float(t_first) - 0.05 and w_start <= float(t_last) + 0.05:
                            collected_words.append(str(w.get("word", "")))
                else:
                    u_start = float(u.get("start", 0.0))
                    u_end = float(u.get("end", 0.0))
                    # If t_first/t_last trimmed this unit and no word list is attached, prefer llm_text
                    if t_first is not None and t_last is not None and (
                        float(t_first) > u_start + 0.20 or float(t_last) < u_end - 0.20
                    ):
                        continue
                    collected_words.append(str(u.get("text", "")).strip())
            if collected_words:
                sep = "" if used_word_bounds else " "
                w_text = sep.join(x for x in collected_words if x).strip()
    if not llm_text:
        llm_text = w_text
    if not w_text:
        w_text = llm_text
    return llm_text, w_text


def _detect_tail_head_overlap(
    text_a: str,
    text_b: str,
    min_exact_chars: int = 6,
    min_stumble_chars: int = 4,
    max_trailing_stumble: int = 2,
) -> str | None:
    """
    Detect when the tail of `text_a` contains an aborted false start that repeats at the
    beginning of `text_b` (`TAIL_HEAD_RETAKE`).

    ASD-STE100:
    1. Check the last 28 normalized characters of `text_a` against the first 28 normalized
       characters of `text_b`.
    2. Flag an overlap when a contiguous suffix-to-prefix clause is >= 6 chars, or >= 4 chars
       followed by 1-to-2 trailing stumble characters at the end of `text_a`.
    """
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if len(norm_a) < min_stumble_chars or len(norm_b) < min_stumble_chars:
        return None

    tail_a = norm_a[-32:]
    head_b = norm_b[:36]

    # 1. Suffix-to-prefix check with tolerance for 1-2 trailing stumble chars at the end of tail_a
    for stumble_tail in range(0, max_trailing_stumble + 1):
        a_end = len(tail_a) - stumble_tail
        min_k = min_stumble_chars if stumble_tail > 0 else min_exact_chars
        if a_end < min_k:
            continue
        max_lead_b = 0 if stumble_tail > 0 else 2
        for lead_b in range(0, max_lead_b + 1):
            max_k = min(a_end, len(head_b) - lead_b)
            for k in range(max_k, min_k - 1, -1):
                sub_a = tail_a[a_end - k : a_end]
                sub_b = head_b[lead_b : lead_b + k]
                if sub_a == sub_b:
                    # Ensure trailing stumble chars actually diverged from head_b continuation
                    if stumble_tail > 0 and (lead_b + k) < len(head_b):
                        if tail_a[a_end : a_end + 1] == head_b[lead_b + k : lead_b + k + 1]:
                            continue
                    return sub_a

    # 2. Shared clause near the tail-to-head boundary (e.g., speaker inserts a short bridge phrase
    #    at the start of Clip B before repeating the stumbled tail clause of Clip A)
    matcher = difflib.SequenceMatcher(None, tail_a, head_b, autojunk=False)
    longest = matcher.find_longest_match(0, len(tail_a), 0, len(head_b))
    if longest.size >= 8 or (
        longest.size >= min_exact_chars and (longest.a + longest.size) >= (len(tail_a) - 7)
    ):
        return tail_a[longest.a : longest.a + longest.size]

    return None


def _detect_intra_clip_repeat(
    text: str,
    min_repeat_chars: int = 6,
    max_gap_chars: int = 12,
) -> str | None:
    """
    Detect when a single clip contains an immediate internal repetition (`A + A + B`).

    ASD-STE100:
    Scan normalized text for any substring of length >= 6 characters that repeats within
    12 intervening characters (`ERR_INTRA_CLIP_REPEAT`).
    """
    norm = normalize_text(text)
    n = len(norm)
    if n < min_repeat_chars * 2:
        return None

    max_k = min(24, n // 2)
    for k in range(max_k, min_repeat_chars - 1, -1):
        for i in range(0, n - 2 * k + 1):
            sub = norm[i : i + k]
            search_end = min(n, i + 2 * k + max_gap_chars)
            next_pos = norm.find(sub, i + k, search_end)
            if next_pos != -1:
                return sub
    return None


def _detect_script_take_collisions(
    clips: list[dict],
    whisper_units: list[dict],
    script_text: str | None,
    min_coverage: float = 0.45,
) -> list[dict]:
    """
    Detect when multiple clips in Mode A map to the same `[Script Block NN]` (`SCRIPT_TAKE_COLLISION`).

    ASD-STE100:
    Flag pairs of clips that both cover >= 45% of the same script block and share overlapping
    content from that block (indicating an unpruned partial take + full take collision).
    """
    if not script_text or len(clips) <= 1:
        return []

    script_blocks = extract_script_blocks(script_text)
    if not script_blocks:
        return []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    block_by_id = {b["block_id"]: b for b in script_blocks}
    clip_matches: list[tuple[int, dict, int, float, str]] = []

    for idx, c in enumerate(clips):
        llm_text, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        best_blk_id = None
        best_score = 0.0
        best_norm = normalize_text(llm_text or w_text)
        for cand_text in (llm_text, w_text):
            norm = normalize_text(cand_text)
            if not norm:
                continue
            for blk in script_blocks:
                score = _script_block_coverage_score(blk["norm_text"], norm)
                if score >= min_coverage and score > best_score and len(norm) <= len(blk["norm_text"]) * 1.85:
                    best_score = score
                    best_blk_id = blk["block_id"]
                    best_norm = norm
        if best_blk_id is not None:
            clip_matches.append((idx, c, best_blk_id, best_score, best_norm))

    collisions: list[dict] = []
    for i in range(len(clip_matches)):
        idx_a, clip_a, blk_a, score_a, norm_a = clip_matches[i]
        for j in range(i + 1, len(clip_matches)):
            idx_b, clip_b, blk_b, score_b, norm_b = clip_matches[j]
            if blk_a != blk_b:
                continue
            # Verify that the two clips overlap on the same clause (rather than splitting a block into two disjoint halves)
            matcher = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False)
            shared = matcher.find_longest_match(0, len(norm_a), 0, len(norm_b))
            if shared.size >= 4 or (score_a >= 0.65 and score_b >= 0.65):
                blk = block_by_id[blk_a]
                collisions.append({
                    "idx_a": idx_a,
                    "idx_b": idx_b,
                    "clip_a": clip_a.get("clip_id", idx_a + 1),
                    "clip_b": clip_b.get("clip_id", idx_b + 1),
                    "block_id": blk_a,
                    "label": blk["label"],
                    "script_block": blk,
                    "score_a": round(score_a, 2),
                    "score_b": round(score_b, 2),
                })
    return collisions


def deduplicate_and_sort_clips(
    clips: list[dict],
    whisper_units: list[dict],
    script_text: str | None = None,
) -> list[dict]:
    """
    Sort clips chronologically and enforce Single-Winner (Last-Take-Wins) deduplication.

    ASD-STE100:
    1. Sort all clips by their physical start time.
    2. When two clips share sentence IDs, overlap in physical time, or collide on the same
       Mode A script block (coverage >= 0.45 with shared clause), retain only the later clip.
    """
    if not clips:
        return []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}

    def _sort_key(c: dict) -> tuple[float, float]:
        t_in, t_out = _clip_time_bounds(c, unit_by_id, whisper_units)
        return (t_in, t_out)

    sorted_clips = sorted((dict(c) for c in clips), key=_sort_key)
    script_blocks = extract_script_blocks(script_text) if script_text else []

    def _matched_script_block_info(c: dict) -> tuple[int | None, float, str]:
        if not script_blocks:
            return None, 0.0, ""
        llm_text, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        best_blk_id = None
        best_score = 0.0
        best_norm = normalize_text(llm_text or w_text)
        for cand_text in (llm_text, w_text):
            norm = normalize_text(cand_text)
            if not norm:
                continue
            for blk in script_blocks:
                score = _script_block_coverage_score(blk["norm_text"], norm)
                if score >= 0.45 and score > best_score and len(norm) <= len(blk["norm_text"]) * 1.85:
                    best_score = score
                    best_blk_id = blk["block_id"]
                    best_norm = norm
        return best_blk_id, best_score, best_norm

    deduped: list[dict] = []
    for curr in sorted_clips:
        curr_sids = set(_resolve_clip_sentence_ids(curr, whisper_units))
        curr_in, curr_out = _clip_time_bounds(curr, unit_by_id, whisper_units)
        curr_blk_id, curr_score, curr_norm = _matched_script_block_info(curr)
        curr_text, _ = _clip_dual_texts(curr, unit_by_id, whisper_units)

        # Remove any earlier clip in `deduped` that is superseded by `curr` (Last Take Wins)
        kept_earlier: list[dict] = []
        for prev in deduped:
            prev_sids = set(_resolve_clip_sentence_ids(prev, whisper_units))
            prev_in, prev_out = _clip_time_bounds(prev, unit_by_id, whisper_units)
            prev_blk_id, prev_score, prev_norm = _matched_script_block_info(prev)
            prev_text, _ = _clip_dual_texts(prev, unit_by_id, whisper_units)

            shares_sids = bool(curr_sids and prev_sids and curr_sids.intersection(prev_sids))
            time_overlaps = (curr_in < prev_out - 0.05) and (curr_out > prev_in + 0.05)
            same_script_block = False
            if curr_blk_id is not None and curr_blk_id == prev_blk_id:
                if curr_score >= 0.65 and prev_score >= 0.65:
                    same_script_block = True
                elif curr_score >= prev_score:
                    matcher = difflib.SequenceMatcher(None, prev_norm, curr_norm, autojunk=False)
                    if matcher.find_longest_match(0, len(prev_norm), 0, len(curr_norm)).size >= 4:
                        same_script_block = True
            adjacent_retake = (
                bool(curr_text and prev_text)
                and (curr_in - prev_out) <= 25.0
                and _are_sentences_retake_related(prev_text, curr_text)
                and same_script_block
            )

            if shares_sids or time_overlaps or same_script_block or adjacent_retake:
                continue
            kept_earlier.append(prev)

        kept_earlier.append(curr)
        deduped = kept_earlier

    for idx, c in enumerate(deduped, start=1):
        c["clip_id"] = idx
    return deduped


def detect_micro_window_anomalies(
    model_edl: dict,
    whisper_units: list[dict],
    total_dur: float,
    script_text: str | None = None,
) -> list[dict]:
    """
    Inspect the initial LLM EDL before video rendering and locate 15-to-90s micro-windows
    that require targeted Gemini re-arbitration.

    ASD-STE100:
    Identify five classes of pre-render anomalies for Gemini micro-window repair:
    1. POTENTIAL_RESIDUAL_RETAKE / TAIL_HEAD_RETAKE: Adjacent clips share retake phrasing,
       tail-to-head stumbled clauses, or overlapping sentence IDs.
    2. INTRA_CLIP_REPEAT: A single clip contains an immediate internal repetition (`A + A + B`).
    3. SCRIPT_TAKE_COLLISION: Multiple clips attempt the same Mode A `[Script Block NN]`.
    4. UNANCHORED_CLIP: Clip lacks valid Whisper sentence IDs.
    5. MISSING_SCRIPT_BLOCK: A script block exists in Whisper sentences but is absent from the EDL.
    """
    clips = model_edl.get("final_edl", []) if isinstance(model_edl, dict) else []
    if not whisper_units or not clips:
        return []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    anomalies: list[dict] = []

    def _overlaps_existing_range(r_start: int, r_end: int) -> bool:
        return any(
            a.get("replace_range") is not None
            and max(r_start, a["replace_range"][0]) < min(r_end, a["replace_range"][1])
            for a in anomalies
        )

    # 1. Check adjacent clips for overlapping sentence IDs, opening retakes, or tail-to-head stumbles
    for idx in range(len(clips) - 1):
        if len(anomalies) >= MAX_MICRO_WINDOW_REPAIRS:
            break
        c_curr = clips[idx]
        c_next = clips[idx + 1]
        sids_curr = _resolve_clip_sentence_ids(c_curr, whisper_units)
        sids_next = _resolve_clip_sentence_ids(c_next, whisper_units)

        t_curr_in, t_curr_out = _clip_time_bounds(c_curr, unit_by_id, whisper_units)
        t_next_in, t_next_out = _clip_time_bounds(c_next, unit_by_id, whisper_units)

        span_sec = t_next_out - t_curr_in
        if span_sec > MAX_MICRO_WINDOW_SEC or (t_next_in - t_curr_out) > 30.0:
            continue

        overlap_ids = set(sids_curr).intersection(set(sids_next))
        llm_curr, w_curr = _clip_dual_texts(c_curr, unit_by_id, whisper_units)
        llm_next, w_next = _clip_dual_texts(c_next, unit_by_id, whisper_units)

        anomaly_type = None
        reason = ""
        if overlap_ids:
            anomaly_type = "POTENTIAL_RESIDUAL_RETAKE"
            reason = f"Adjacent clips {idx + 1} and {idx + 2} share overlapping Sentence IDs {sorted(overlap_ids)}."
        elif (llm_curr and llm_next and _are_sentences_retake_related(llm_curr, llm_next)) or (
            w_curr and w_next and _are_sentences_retake_related(w_curr, w_next)
        ):
            anomaly_type = "POTENTIAL_RESIDUAL_RETAKE"
            reason = (
                f"Adjacent clips {idx + 1} ('{llm_curr[:24]}') and {idx + 2} ('{llm_next[:24]}') "
                "contain overlapping opening clauses that indicate a possible unpruned retake."
            )
        else:
            tail_overlap = _detect_tail_head_overlap(llm_curr, llm_next) or _detect_tail_head_overlap(w_curr, w_next)
            if tail_overlap:
                anomaly_type = "TAIL_HEAD_RETAKE"
                reason = (
                    f"Clip {idx + 1} tail and Clip {idx + 2} head repeat the phrase '{tail_overlap}', "
                    "indicating a trailing false start before a sentence restart."
                )

        if anomaly_type is not None:
            win_start = max(0.0, t_curr_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_next_out + 3.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC and not _overlaps_existing_range(idx, idx + 2):
                anomalies.append({
                    "type": anomaly_type,
                    "replace_range": (idx, idx + 2),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "reason": reason,
                })

    # 2. Check for intra-clip immediate repetitions (`A + A + B` inside a single clip)
    for idx, c in enumerate(clips):
        if len(anomalies) >= MAX_MICRO_WINDOW_REPAIRS:
            break
        if _overlaps_existing_range(idx, idx + 1):
            continue
        llm_text, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        repeated_sub = _detect_intra_clip_repeat(llm_text)
        if repeated_sub is None and len(normalize_text(llm_text)) >= len(normalize_text(w_text)) * 0.92:
            repeated_sub = _detect_intra_clip_repeat(w_text)
        if repeated_sub:
            t_in, t_out = _clip_time_bounds(c, unit_by_id, whisper_units)
            win_start = max(0.0, t_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_out + 3.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                anomalies.append({
                    "type": "INTRA_CLIP_REPEAT",
                    "replace_range": (idx, idx + 1),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "reason": f"Clip {idx + 1} contains an internal repeated clause '{repeated_sub}' that requires clean transcript trimming.",
                })

    # 3. Mode A: Check for script block take collisions (`SCRIPT_TAKE_COLLISION`)
    if script_text and len(anomalies) < MAX_MICRO_WINDOW_REPAIRS:
        collisions = _detect_script_take_collisions(clips, whisper_units, script_text, min_coverage=0.45)
        for col in collisions:
            if len(anomalies) >= MAX_MICRO_WINDOW_REPAIRS:
                break
            idx_a, idx_b = col["idx_a"], col["idx_b"]
            if _overlaps_existing_range(idx_a, idx_b + 1):
                continue
            t_a_in, _ = _clip_time_bounds(clips[idx_a], unit_by_id, whisper_units)
            _, t_b_out = _clip_time_bounds(clips[idx_b], unit_by_id, whisper_units)
            win_start = max(0.0, t_a_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_b_out + 3.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                anomalies.append({
                    "type": "SCRIPT_TAKE_COLLISION",
                    "replace_range": (idx_a, idx_b + 1),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "script_block": col["script_block"],
                    "reason": (
                        f"Clips {col['clip_a']} and {col['clip_b']} both attempt {col['label']} "
                        f"(coverage {col['score_a']} vs {col['score_b']}); retain only the single complete winning take."
                    ),
                })

    # 4. Check for unanchored clips (missing sentence_ids)
    for idx, c in enumerate(clips):
        if len(anomalies) >= MAX_MICRO_WINDOW_REPAIRS:
            break
        if _overlaps_existing_range(idx, idx + 1):
            continue
        sids = _resolve_clip_sentence_ids(c, whisper_units)
        if not sids:
            raw_in = float(c.get("source_in", 0.0))
            raw_out = float(c.get("source_out", min(total_dur, raw_in + 10.0)))
            win_start = max(0.0, raw_in - 8.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, raw_out + 8.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                anomalies.append({
                    "type": "UNANCHORED_CLIP",
                    "replace_range": (idx, idx + 1),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "reason": f"Clip {idx + 1} has no valid Whisper sentence_ids and requires exact sentence anchoring.",
                })

    # 5. Mode A: Check for missed script blocks that exist in contiguous unselected Whisper clusters
    if script_text and len(anomalies) < MAX_MICRO_WINDOW_REPAIRS:
        blocks = extract_script_blocks(script_text)
        selected_sids = set()
        selected_texts: list[str] = []
        for c in clips:
            c_sids = _resolve_clip_sentence_ids(c, whisper_units)
            selected_sids.update(c_sids)
            c_text = c.get("transcript", "") or ""
            if not c_text and c_sids:
                c_text = " ".join(str(unit_by_id[sid].get("text", "")) for sid in c_sids if sid in unit_by_id)
            selected_texts.append(normalize_text(c_text))
        selected_norm_text = " ".join(t for t in selected_texts if t)

        for blk in blocks:
            if len(anomalies) >= MAX_MICRO_WINDOW_REPAIRS:
                break
            if _script_block_coverage_score(blk["norm_text"], selected_norm_text) >= 0.55:
                continue

            # Find unselected Whisper units that overlap this script block
            cand_units = [
                u for u in whisper_units
                if int(u.get("id", 0)) not in selected_sids
                and _unit_overlaps_script_block(blk["norm_text"], normalize_text(u.get("text", "")))
            ]
            if not cand_units:
                continue

            # Group candidate units into temporal clusters (split when gap > 12.0s)
            clusters: list[list[dict]] = []
            for u in cand_units:
                if not clusters:
                    clusters.append([u])
                else:
                    prev_end = float(clusters[-1][-1].get("end", 0.0))
                    curr_start = float(u.get("start", 0.0))
                    if (curr_start - prev_end) <= 12.0:
                        clusters[-1].append(u)
                    else:
                        clusters.append([u])

            # Evaluate each cluster's coverage of the missing script block; pick latest valid cluster
            valid_clusters: list[tuple[float, list[dict]]] = []
            for cl in clusters:
                cl_norm = "".join(normalize_text(u.get("text", "")) for u in cl)
                cov = _script_block_coverage_score(blk["norm_text"], cl_norm)
                if cov >= 0.50:
                    valid_clusters.append((cov, cl))

            if not valid_clusters:
                continue

            # Last-Take-Wins: prefer the latest qualifying cluster
            _, chosen_cluster = valid_clusters[-1]
            win_start = max(0.0, float(chosen_cluster[0].get("start", 0.0)) - 4.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, float(chosen_cluster[-1].get("end", 0.0)) + 4.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                anomalies.append({
                    "type": "MISSING_SCRIPT_BLOCK",
                    "replace_range": None,
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "script_block": blk,
                    "reason": f"{blk['label']} was omitted from the initial EDL but matches Whisper sentences in {win_start:.1f}s..{win_end:.1f}s.",
                })

    return anomalies[:MAX_MICRO_WINDOW_REPAIRS]


def repair_edl_micro_windows(
    model_edl: dict,
    whisper_units: list[dict],
    total_dur: float,
    run_window_inference_fn,
    script_text: str | None = None,
) -> tuple[dict, list[dict]]:
    """
    Execute targeted micro-window re-scans for detected pre-render anomalies and splice
    repaired clips back into model_edl with Single-Winner (Last-Take-Wins) arbitration.

    ASD-STE100:
    Re-scan only short 15-to-90s video windows to avoid gateway timeouts.
    """
    anomalies = detect_micro_window_anomalies(
        model_edl=model_edl,
        whisper_units=whisper_units,
        total_dur=total_dur,
        script_text=script_text,
    )
    if not anomalies:
        return model_edl, []

    clips = [dict(c) for c in model_edl.get("final_edl", [])]
    applied_repairs = []

    # Process anomalies in reverse order of replace_range so indices remain valid
    indexed_anomalies = sorted(
        anomalies,
        key=lambda a: (a["replace_range"][0] if a.get("replace_range") is not None else 999999),
        reverse=True,
    )

    for anomaly in indexed_anomalies:
        win_start = anomaly["win_start"]
        win_end = anomaly["win_end"]
        win_units = [
            u for u in whisper_units
            if float(u.get("end", 0.0)) >= win_start and float(u.get("start", 0.0)) <= win_end
        ]
        if not win_units:
            continue

        start_offset = f"{int(math.floor(win_start))}s"
        end_offset = f"{int(math.ceil(win_end))}s"
        logger.info(
            "    [局部視窗精準重掃] %s | 區間: %s ~ %s (%d 句候選台詞)",
            anomaly["type"],
            start_offset,
            end_offset,
            len(win_units),
        )

        try:
            repaired_edl = run_window_inference_fn(
                win_units=win_units,
                win_start=win_start,
                win_end=win_end,
                start_offset=start_offset,
                end_offset=end_offset,
                anomaly=anomaly,
            )
        except Exception as exc:
            logger.warning("    [局部視窗重掃跳過] 區間 %s~%s 推論失敗，保留原結果: %s", start_offset, end_offset, exc)
            continue

        new_clips = repaired_edl.get("final_edl", []) if isinstance(repaired_edl, dict) else []
        valid_new_clips = [
            c for c in new_clips
            if _resolve_clip_sentence_ids(c, win_units)
        ]
        if not valid_new_clips:
            continue

        replace_range = anomaly.get("replace_range")
        if replace_range is not None:
            r_start, r_end = replace_range
            clips[r_start:r_end] = valid_new_clips
        else:
            clips.extend(valid_new_clips)

        applied_repairs.append({
            "type": anomaly["type"],
            "window": f"{start_offset}~{end_offset}",
            "reason": anomaly["reason"],
            "repaired_clip_count": len(valid_new_clips),
        })

    if applied_repairs:
        clips = deduplicate_and_sort_clips(
            clips=clips,
            whisper_units=whisper_units,
            script_text=script_text,
        )

    updated_edl = dict(model_edl)
    updated_edl["final_edl"] = clips
    return updated_edl, list(reversed(applied_repairs))


def sanitize_refined_edl(
    refined_edl: list[dict],
    total_dur: float,
    min_clip_dur: float = 0.45,
    merge_gap_thresh: float = 0.35,
) -> tuple[list[dict], dict]:
    """
    Sanitize physical clip timings after acoustic boundary locking and before NLE/MP4 export.

    ASD-STE100:
    1. Sort clips chronologically and enforce source_out >= t_last so final consonants are preserved.
    2. Prune contained or backward-jumping redundant clips and resolve micro-overlaps at their midpoint.
    3. Merge flash-frame micro-clips (< 0.45s) into adjacent clips when gap < 0.35s.
    4. Guarantee strict forward monotonicity (source_in < source_out and c[i].source_out <= c[i+1].source_in).
    """
    stats = {
        "overlaps_resolved": 0,
        "plosive_tails_restored": 0,
        "micro_clips_merged": 0,
    }
    if not refined_edl:
        return [], stats

    sanitized = [dict(c) for c in refined_edl]

    # 1. Enforce source_bounds and plosive tail protection (source_out >= t_last)
    for c in sanitized:
        s_in = max(0.0, min(total_dur, float(c.get("source_in", 0.0))))
        s_out = max(0.0, min(total_dur, float(c.get("source_out", s_in + 0.5))))
        t_last = c.get("t_last")
        if t_last is not None and s_out < float(t_last):
            s_out = min(total_dur, round(float(t_last) + 0.04, 2))
            stats["plosive_tails_restored"] += 1
        if s_out <= s_in:
            s_out = min(total_dur, round(s_in + 0.50, 2))
        c["source_in"] = round(s_in, 2)
        c["source_out"] = round(s_out, 2)
        c["duration"] = round(c["source_out"] - c["source_in"], 2)

    # Sort strictly by (source_in, source_out) before overlap resolution
    sanitized.sort(key=lambda c: (float(c["source_in"]), float(c["source_out"])))

    # 2. De-conflict adjacent overlaps and prune nested/contained redundant clips
    deconflicted: list[dict] = []
    for curr_c in sanitized:
        if not deconflicted:
            if curr_c["source_out"] > curr_c["source_in"]:
                deconflicted.append(curr_c)
            continue

        prev_c = deconflicted[-1]
        # If curr_c is completely contained within prev_c, drop the redundant inner clip
        if curr_c["source_out"] <= prev_c["source_out"] + 0.05:
            stats["overlaps_resolved"] += 1
            continue

        if curr_c["source_in"] < prev_c["source_out"]:
            mid = round((prev_c["source_out"] + curr_c["source_in"]) / 2.0, 2)
            prev_t_last = float(prev_c.get("t_last", prev_c["source_in"] + 0.2))
            curr_t_first = float(curr_c.get("t_first", curr_c["source_in"]))
            boundary = max(prev_t_last, min(curr_t_first, mid))

            # Clamp boundary so both prev_c and curr_c maintain strictly positive durations (>= 0.15s)
            min_valid_boundary = prev_c["source_in"] + 0.15
            max_valid_boundary = curr_c["source_out"] - 0.15
            if max_valid_boundary < min_valid_boundary:
                # Span is too narrow for two distinct clips; drop the redundant second clip
                stats["overlaps_resolved"] += 1
                continue

            boundary = round(max(min_valid_boundary, min(max_valid_boundary, boundary)), 2)
            prev_c["source_out"] = boundary
            curr_c["source_in"] = boundary
            prev_c["duration"] = round(prev_c["source_out"] - prev_c["source_in"], 2)
            curr_c["duration"] = round(curr_c["source_out"] - curr_c["source_in"], 2)
            stats["overlaps_resolved"] += 1

        if curr_c["source_out"] - curr_c["source_in"] >= 0.15:
            deconflicted.append(curr_c)

    # 3. Coalesce micro-clips (< min_clip_dur) into adjacent clips if gap < merge_gap_thresh
    merged: list[dict] = []
    for c in deconflicted:
        if (
            merged
            and (c["duration"] < min_clip_dur or merged[-1]["duration"] < min_clip_dur)
            and c["source_out"] > merged[-1]["source_out"]
            and 0.0 <= (c["source_in"] - merged[-1]["source_out"]) <= merge_gap_thresh
        ):
            prev = merged[-1]
            prev["source_out"] = c["source_out"]
            prev["duration"] = round(prev["source_out"] - prev["source_in"], 2)
            if c.get("t_last") is not None:
                prev["t_last"] = max(float(prev.get("t_last") or 0.0), float(c["t_last"]))
            prev_sids = list(prev.get("sentence_ids") or [])
            for sid in (c.get("sentence_ids") or []):
                if sid not in prev_sids:
                    prev_sids.append(sid)
            prev["sentence_ids"] = prev_sids
            prev["transcript"] = f"{prev.get('transcript', '').strip()} {c.get('transcript', '').strip()}".strip()
            prev["cps"], _ = calculate_clip_cps(prev["transcript"], prev["duration"])
            stats["micro_clips_merged"] += 1
        else:
            merged.append(c)

    final_clips: list[dict] = []
    for c in merged:
        dur = round(float(c["source_out"]) - float(c["source_in"]), 2)
        if dur <= 0.0:
            continue
        c["duration"] = dur
        c["clip_id"] = len(final_clips) + 1
        c["cps"], _ = calculate_clip_cps(c.get("transcript", ""), dur)
        final_clips.append(c)

    return final_clips, stats


def audit_edl_quality(
    refined_edl: list[dict],
    whisper_units: list[dict],
    total_dur: float,
    video_path: Path,
    script_text: str | None = None,
    script_path: Path | None = None,
    sanitization_stats: dict | None = None,
    micro_window_repairs: list[dict] | None = None,
) -> dict:
    """
    Execute an 8-dimension rough-cut quality audit and compute the top-level agent_verdict.

    ASD-STE100:
    Evaluate Whisper lock rate, timeline monotonicity, retake hygiene, script coverage,
    flash-frame micro-clips, CPS dead-air anomalies, acoustic margins, and compression ratio.
    """
    clip_count = len(refined_edl)
    fatal_violations: list[str] = []
    warnings: list[str] = []

    if clip_count == 0:
        fatal_violations.append("Final EDL contains 0 clips.")
        return {
            "agent_verdict": {
                "pass_quality_gate": False,
                "overall_grade": "FAIL",
                "fatal_violations": fatal_violations,
                "suggested_action": "ONE_SHOT_REMEDIATE",
                "remediation_reason": "Final EDL is empty.",
                "remediation_cmd": None,
            },
            "summary": {"clip_count": 0, "source_duration_sec": round(total_dur, 2)},
        }

    # Dimension 1: Whisper Word-Level Lock Rate
    if whisper_units:
        locked_clips = [c for c in refined_edl if c.get("sentence_ids")]
        whisper_lock_rate_pct = round(len(locked_clips) * 100.0 / clip_count, 1)
    else:
        whisper_lock_rate_pct = 100.0

    if whisper_units and whisper_lock_rate_pct < 80.0:
        fatal_violations.append(
            f"Whisper sentence lock rate {whisper_lock_rate_pct:.1f}% is below the 80.0% minimum threshold."
        )
    elif whisper_units and whisper_lock_rate_pct < 90.0:
        warnings.append(f"Whisper sentence lock rate is {whisper_lock_rate_pct:.1f}% (< 90.0%).")

    # Dimension 2: Timeline Monotonicity & Overlap Check
    overlap_count = 0
    out_of_order_count = 0
    out_of_bounds_count = 0
    for i, c in enumerate(refined_edl):
        s_in = float(c.get("source_in", 0.0))
        s_out = float(c.get("source_out", 0.0))
        if s_in < -0.01 or s_out > total_dur + 0.05 or s_out <= s_in:
            out_of_bounds_count += 1
        if i > 0:
            prev_in = float(refined_edl[i - 1].get("source_in", 0.0))
            prev_out = float(refined_edl[i - 1].get("source_out", 0.0))
            if s_in < prev_in:
                out_of_order_count += 1
            elif s_in < prev_out - 0.005:
                overlap_count += 1

    if overlap_count > 0:
        fatal_violations.append(f"Detected {overlap_count} overlapping clip boundary pair(s).")
    if out_of_order_count > 0:
        fatal_violations.append(f"Detected {out_of_order_count} non-monotonic out-of-order clip(s).")
    if out_of_bounds_count > 0:
        fatal_violations.append(f"Detected {out_of_bounds_count} clip(s) outside [0, {total_dur:.2f}s].")

    # Dimension 3: Residual Retake, Tail-to-Head Stumble, Intra-Clip Repeat & Off-Screen Crew Check
    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    residual_retake_pairs = []
    tail_head_retake_pairs = []
    for i in range(clip_count - 1):
        llm_a, w_a = _clip_dual_texts(refined_edl[i], unit_by_id, whisper_units)
        llm_b, w_b = _clip_dual_texts(refined_edl[i + 1], unit_by_id, whisper_units)
        gap_sec = float(refined_edl[i + 1]["source_in"]) - float(refined_edl[i]["source_out"])
        if gap_sec <= 25.0:
            if _are_sentences_retake_related(llm_a, llm_b) or _are_sentences_retake_related(w_a, w_b):
                residual_retake_pairs.append({
                    "type": "OPENING_RETAKE",
                    "clip_a": refined_edl[i]["clip_id"],
                    "clip_b": refined_edl[i + 1]["clip_id"],
                    "text_a": (llm_a or w_a)[:40],
                    "text_b": (llm_b or w_b)[:40],
                })
            else:
                tail_overlap = _detect_tail_head_overlap(llm_a, llm_b) or _detect_tail_head_overlap(w_a, w_b)
                if tail_overlap:
                    pair_entry = {
                        "type": "ERR_TAIL_HEAD_RETAKE",
                        "clip_a": refined_edl[i]["clip_id"],
                        "clip_b": refined_edl[i + 1]["clip_id"],
                        "overlap": tail_overlap,
                        "text_a": (llm_a or w_a)[-40:],
                        "text_b": (llm_b or w_b)[:40],
                    }
                    residual_retake_pairs.append(pair_entry)
                    tail_head_retake_pairs.append(pair_entry)

    intra_clip_repeats = []
    off_screen_clip_ids = []
    for c in refined_edl:
        sids = c.get("sentence_ids") or []
        if sids and all(not unit_by_id.get(int(sid), {}).get("is_target_speaker", True) for sid in sids if int(sid) in unit_by_id):
            off_screen_clip_ids.append(c["clip_id"])
        llm_c, w_c = _clip_dual_texts(c, unit_by_id, whisper_units)
        rep_phrase = _detect_intra_clip_repeat(llm_c)
        if rep_phrase is None and len(normalize_text(llm_c)) >= len(normalize_text(w_c)) * 0.92:
            rep_phrase = _detect_intra_clip_repeat(w_c)
        if rep_phrase:
            intra_clip_repeats.append({
                "type": "ERR_INTRA_CLIP_REPEAT",
                "clip_id": c["clip_id"],
                "repeated_phrase": rep_phrase,
                "text": (llm_c or w_c)[:60],
            })

    if residual_retake_pairs:
        pair_desc = ", ".join(f"Clip {p['clip_a']}->{p['clip_b']}" for p in residual_retake_pairs[:3])
        err_tag = " [ERR_TAIL_HEAD_RETAKE]" if tail_head_retake_pairs else ""
        fatal_violations.append(f"Detected {len(residual_retake_pairs)} residual retake pair(s) ({pair_desc}){err_tag}.")
    if intra_clip_repeats:
        rep_desc = ", ".join(f"Clip {r['clip_id']} ('{r['repeated_phrase']}')" for r in intra_clip_repeats[:3])
        fatal_violations.append(f"Detected {len(intra_clip_repeats)} intra-clip repeat(s) ({rep_desc}) [ERR_INTRA_CLIP_REPEAT].")
    if off_screen_clip_ids:
        fatal_violations.append(f"Detected off-screen speaker audio in Clip(s) {off_screen_clip_ids}.")

    # Dimension 4: Script Clause Coverage & Single-Winner Collision Check (Mode A)
    script_blocks = _extract_script_blocks(script_text)
    script_take_collisions = []
    if script_blocks:
        combined_norm = " ".join(normalize_text(c.get("transcript", "")) for c in refined_edl)
        matched_blocks = []
        missing_blocks = []
        for blk in script_blocks:
            if _match_text_similarity(blk["norm_text"], combined_norm) >= 0.50:
                matched_blocks.append(blk["label"])
            else:
                missing_blocks.append(blk["label"])
        script_coverage_pct = round(len(matched_blocks) * 100.0 / len(script_blocks), 1)
        if script_coverage_pct < 75.0:
            fatal_violations.append(
                f"Mode A script block coverage {script_coverage_pct:.1f}% is below 75.0% (missing: {', '.join(missing_blocks[:4])})."
            )
        elif script_coverage_pct < 85.0:
            warnings.append(f"Mode A script block coverage is {script_coverage_pct:.1f}% (< 85.0%).")

        raw_collisions = _detect_script_take_collisions(refined_edl, whisper_units, script_text, min_coverage=0.45)
        for col in raw_collisions:
            script_take_collisions.append({
                "type": "ERR_SCRIPT_TAKE_COLLISION",
                "clip_a": col["clip_a"],
                "clip_b": col["clip_b"],
                "block_id": col["block_id"],
                "label": col["label"],
                "score_a": col["score_a"],
                "score_b": col["score_b"],
            })
        if script_take_collisions:
            col_desc = ", ".join(
                f"Clips {c['clip_a']}&{c['clip_b']} on {c['label']}" for c in script_take_collisions[:3]
            )
            fatal_violations.append(
                f"Detected {len(script_take_collisions)} Mode A script block collision(s) ({col_desc}) [ERR_SCRIPT_TAKE_COLLISION]."
            )
    else:
        script_coverage_pct = None
        missing_blocks = []

    # Dimension 5: Flash-Frame Micro-Clip Check (< 0.40s)
    micro_clips = [c["clip_id"] for c in refined_edl if float(c.get("duration", 0.0)) < 0.40]
    if micro_clips:
        warnings.append(f"Detected {len(micro_clips)} micro-clip(s) shorter than 0.40s: {micro_clips}.")

    # Dimension 6: Dead-Air & CPS Anomaly Check
    low_cps_clips = [
        c["clip_id"]
        for c in refined_edl
        if float(c.get("duration", 0.0)) >= 3.0 and float(c.get("cps", 3.0)) < 1.2
    ]
    high_cps_clips = [c["clip_id"] for c in refined_edl if float(c.get("cps", 3.0)) > 10.0]
    if low_cps_clips:
        warnings.append(f"Detected {len(low_cps_clips)} clip(s) with low CPS (< 1.2 chars/s, possible internal dead air): {low_cps_clips}.")

    # Dimension 7: Acoustic Margin & Plosive Tail Protection
    plosive_truncated_clips = [
        c["clip_id"]
        for c in refined_edl
        if c.get("t_last") is not None and float(c["source_out"]) < float(c["t_last"]) - 0.01
    ]
    if plosive_truncated_clips:
        fatal_violations.append(
            f"Detected {len(plosive_truncated_clips)} clip(s) where source_out truncates the final spoken word (t_last): {plosive_truncated_clips}."
        )

    avg_in_margin = round(sum(float(c.get("in_margin", 0.0)) for c in refined_edl) / clip_count, 3)
    avg_out_margin = round(sum(float(c.get("out_margin", 0.0)) for c in refined_edl) / clip_count, 3)

    # Dimension 8: Compression & Pacing Summary
    trimmed_dur = round(sum(float(c.get("duration", 0.0)) for c in refined_edl), 2)
    avg_cps = round(sum(float(c.get("cps", 0.0)) for c in refined_edl) / clip_count, 2)
    trim_ratio_pct = round(max(0.0, (total_dur - trimmed_dur) * 100.0 / max(0.01, total_dur)), 1)

    pass_gate = len(fatal_violations) == 0
    if pass_gate and not warnings:
        overall_grade = "EXCELLENT"
    elif pass_gate:
        overall_grade = "PASS"
    else:
        overall_grade = "FAIL"

    remediation_cmd = None
    remediation_reason = None
    if not pass_gate:
        script_flag = f' -s "{script_path}"' if script_path else ""
        remediation_reason = "; ".join(fatal_violations)
        remediation_cmd = (
            f'python3 skills/video-trimmer/scripts/video_trimmer.py -i "{video_path}"{script_flag}'
        )

    return {
        "agent_verdict": {
            "pass_quality_gate": pass_gate,
            "overall_grade": overall_grade,
            "fatal_violations": fatal_violations,
            "warnings": warnings,
            "suggested_action": "DELIVER" if pass_gate else "ONE_SHOT_REMEDIATE",
            "remediation_reason": remediation_reason,
            "remediation_cmd": remediation_cmd,
        },
        "summary": {
            "video_file": video_path.name,
            "mode": "Mode A (Script-Anchored)" if script_blocks else "Mode B (Unscripted Intent-Window)",
            "source_duration_sec": round(total_dur, 2),
            "trimmed_duration_sec": trimmed_dur,
            "trim_ratio_pct": trim_ratio_pct,
            "clip_count": clip_count,
            "average_cps": avg_cps,
            "micro_window_repairs_applied": len(micro_window_repairs or []),
            "sanitization_stats": sanitization_stats or {},
        },
        "dimensions": {
            "1_whisper_lock": {
                "whisper_lock_rate_pct": whisper_lock_rate_pct,
                "pass": (not whisper_units) or (whisper_lock_rate_pct >= 80.0),
            },
            "2_timeline_integrity": {
                "overlap_count": overlap_count,
                "out_of_order_count": out_of_order_count,
                "out_of_bounds_count": out_of_bounds_count,
                "pass": overlap_count == 0 and out_of_order_count == 0 and out_of_bounds_count == 0,
            },
            "3_retake_and_speaker_hygiene": {
                "residual_retake_count": len(residual_retake_pairs),
                "residual_retake_pairs": residual_retake_pairs,
                "intra_clip_repeat_count": len(intra_clip_repeats),
                "intra_clip_repeats": intra_clip_repeats,
                "off_screen_speaker_clip_count": len(off_screen_clip_ids),
                "pass": (
                    len(residual_retake_pairs) == 0
                    and len(intra_clip_repeats) == 0
                    and len(off_screen_clip_ids) == 0
                ),
            },
            "4_script_coverage": {
                "enabled": bool(script_blocks),
                "total_script_blocks": len(script_blocks),
                "script_coverage_pct": script_coverage_pct,
                "missing_blocks": missing_blocks,
                "script_take_collision_count": len(script_take_collisions),
                "script_take_collisions": script_take_collisions,
                "pass": (
                    (not script_blocks)
                    or (
                        script_coverage_pct is not None
                        and script_coverage_pct >= 75.0
                        and len(script_take_collisions) == 0
                    )
                ),
            },
            "5_micro_clip_check": {
                "micro_clip_count": len(micro_clips),
                "micro_clip_ids": micro_clips,
                "pass": len(micro_clips) == 0,
            },
            "6_cps_and_dead_air": {
                "low_cps_dead_air_count": len(low_cps_clips),
                "low_cps_clip_ids": low_cps_clips,
                "high_cps_anomaly_count": len(high_cps_clips),
                "pass": len(low_cps_clips) == 0,
            },
            "7_acoustic_margins": {
                "plosive_truncated_count": len(plosive_truncated_clips),
                "avg_in_margin_sec": avg_in_margin,
                "avg_out_margin_sec": avg_out_margin,
                "pass": len(plosive_truncated_clips) == 0,
            },
            "8_compression_summary": {
                "source_duration_sec": round(total_dur, 2),
                "trimmed_duration_sec": trimmed_dur,
                "trim_ratio_pct": trim_ratio_pct,
                "average_cps": avg_cps,
                "pass": trimmed_dur > 0.0,
            },
        },
        "micro_window_repairs": micro_window_repairs or [],
    }


def generate_edl_audit_markdown(report: dict, output_md_path: Path) -> None:
    """
    Write a plain technical Markdown audit report without decorative emojis.

    ASD-STE100:
    Format the 8-dimension audit results and agent_verdict into a structured Markdown table.
    """
    verdict = report.get("agent_verdict", {})
    summary = report.get("summary", {})
    dims = report.get("dimensions", {})
    repairs = report.get("micro_window_repairs", [])
    san = summary.get("sanitization_stats", {})

    d1 = dims.get("1_whisper_lock", {})
    d2 = dims.get("2_timeline_integrity", {})
    d3 = dims.get("3_retake_and_speaker_hygiene", {})
    d4 = dims.get("4_script_coverage", {})
    d5 = dims.get("5_micro_clip_check", {})
    d6 = dims.get("6_cps_and_dead_air", {})
    d7 = dims.get("7_acoustic_margins", {})

    script_cov_str = (
        f"{d4.get('script_coverage_pct')}% ({d4.get('total_script_blocks')} blocks, {d4.get('script_take_collision_count', 0)} collisions)"
        if d4.get("enabled")
        else "N/A (Mode B Unscripted)"
    )

    lines = [
        f"# Video Trimmer Quality Audit Report - {summary.get('video_file', '')}",
        "",
        "## 1. Executive Summary & Agent Verdict",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| **Pass Quality Gate** | `{verdict.get('pass_quality_gate')}` |",
        f"| **Overall Grade** | `{verdict.get('overall_grade')}` |",
        f"| **Suggested Action** | `{verdict.get('suggested_action')}` |",
        f"| **Arbitration Mode** | {summary.get('mode')} |",
        f"| **Source Duration** | {summary.get('source_duration_sec')} s |",
        f"| **Trimmed Duration** | {summary.get('trimmed_duration_sec')} s (Trim Ratio: {summary.get('trim_ratio_pct')}%) |",
        f"| **Retained Clips** | {summary.get('clip_count')} |",
        f"| **Average CPS** | {summary.get('average_cps')} chars/s |",
        "",
        "## 2. 8-Dimension Rough-Cut Quality Audit",
        "",
        "| Dimension | Metric | Result | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **1. Whisper Word-Level Lock** | Sentence ID Lock Rate | {d1.get('whisper_lock_rate_pct')}% | {'PASS' if d1.get('pass') else 'FAIL'} |",
        f"| **2. Timeline Monotonicity** | Overlaps / Out-of-Order | {d2.get('overlap_count')} / {d2.get('out_of_order_count')} | {'PASS' if d2.get('pass') else 'FAIL'} |",
        f"| **3. Retake & Speaker Hygiene** | Retakes / Intra Repeats / Crew Clips | {d3.get('residual_retake_count')} / {d3.get('intra_clip_repeat_count', 0)} / {d3.get('off_screen_speaker_clip_count')} | {'PASS' if d3.get('pass') else 'FAIL'} |",
        f"| **4. Script Block Coverage** | Mode A Coverage & Collisions | {script_cov_str} | {'PASS' if d4.get('pass') else 'FAIL'} |",
        f"| **5. Flash-Frame Micro-Clips** | Clips < 0.40s | {d5.get('micro_clip_count')} | {'PASS' if d5.get('pass') else 'WARN'} |",
        f"| **6. Dead-Air & CPS Check** | Low-CPS Clips (< 1.2 CPS) | {d6.get('low_cps_dead_air_count')} | {'PASS' if d6.get('pass') else 'WARN'} |",
        f"| **7. Acoustic Margins** | Mean In / Out Margin | {d7.get('avg_in_margin_sec')}s / {d7.get('avg_out_margin_sec')}s | {'PASS' if d7.get('pass') else 'FAIL'} |",
        f"| **8. Compression Summary** | Trimmed / Source Duration | {summary.get('trimmed_duration_sec')}s / {summary.get('source_duration_sec')}s | PASS |",
        "",
        "## 3. In-Pipeline Self-Healing Summary",
        "",
        f"- **Surgical Micro-Window Re-Scans Applied**: {len(repairs)}",
        f"- **Boundary Overlaps Resolved**: {san.get('overlaps_resolved', 0)}",
        f"- **Plosive Tails Restored (`source_out >= t_last`)**: {san.get('plosive_tails_restored', 0)}",
        f"- **Flash-Frame Micro-Clips Merged**: {san.get('micro_clips_merged', 0)}",
    ]

    if repairs:
        lines.extend([
            "",
            "### Micro-Window Repair Log",
            "",
            "| Type | Window | Repaired Clips | Reason |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for r in repairs:
            lines.append(
                f"| `{r.get('type')}` | `{r.get('window')}` | {r.get('repaired_clip_count')} | {r.get('reason')} |"
            )

    if verdict.get("fatal_violations"):
        lines.extend(["", "## 4. Fatal Violations", ""])
        for v in verdict["fatal_violations"]:
            lines.append(f"- {v}")

    if verdict.get("warnings"):
        lines.extend(["", "## 5. Audit Warnings", ""])
        for w in verdict["warnings"]:
            lines.append(f"- {w}")

    lines.append("")
    Path(output_md_path).write_text("\n".join(lines), encoding="utf-8")

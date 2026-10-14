"""Assembly pass: global, text-only arbitration across chunk results (Show Mode).

ASD-STE100:
Long recordings are analysed in several video chunks. Each chunk returns candidate
clips without knowledge of the other chunks. This module sends all candidate clips,
the complete sentence list (with mic roles) and the reference script to Gemini in a
single text-only request, and receives the final ordered EDL.

The pass never uploads video. It only resolves: cross-chunk duplicate takes, missing
sentences, off-screen voice-role decisions, segment labels and timeline order.

Output size control:
1. The model returns only decisions (candidate reference, sentence IDs, role, segment).
   Times come from the Whisper sentences; transcript, reasons and visual checks come from
   the chunk candidates. A response schema enforces this shape.
2. Thinking tokens count against `max_output_tokens`, and Gemini 3 models do not treat
   `thinking_budget` as a hard limit (a replay used 36.5k thought tokens with a budget of
   11.6k). Gemini 3 models get `thinking_level`; `max_output_tokens` keeps headroom for it.
3. `assemble_with_fallback` retries once with a minimal format when an attempt fails,
   then falls back to the chunk results in source order. Every attempt writes its raw
   output and usage to the output folder.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .exceptions import GeminiAPIError, InvalidInputError
from .gemini_client import (
    _extract_text_from_response,
    _generate_content,
    _usage_from_generate_content,
    build_prompt,
    merge_chunked_edl,
    parse_edl_json_text,
)
from .constants import GEMINI_MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)

ASSEMBLY_PROMPT_NAME = "assembly_prompt.md"
VALID_ROLES = ("host", "offscreen")

FORMAT_COMPACT = "compact"
FORMAT_MINIMAL = "minimal"
ASSEMBLY_ATTEMPT_FORMATS = (FORMAT_COMPACT, FORMAT_MINIMAL)
# Gemini 3 thinking levels per attempt. The retry thinks less so that it finishes.
ASSEMBLY_THINKING_LEVEL = {FORMAT_COMPACT: "MEDIUM", FORMAT_MINIMAL: "LOW"}

STATUS_ASSEMBLED = "assembled"
STATUS_ASSEMBLED_AFTER_RETRY = "assembled_after_retry"
STATUS_DEGRADED = "degraded"
DEGRADED_BANNER = "全片組裝失敗、已降級：改用各區段結果，照原片時間順序輸出（timeline_position 全片唯一重編）"

_ZERO_USAGE = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "tool_use_tokens": 0, "total_tokens": 0}


def assembly_prompt_candidates(base_candidates: list[Path]) -> list[Path]:
    """Map the video-cut prompt candidate paths to assembly prompt paths in the same folders."""
    seen: list[Path] = []
    for p in base_candidates:
        cand = Path(p).parent / ASSEMBLY_PROMPT_NAME
        if cand not in seen:
            seen.append(cand)
    return seen


class AssemblyTruncatedError(GeminiAPIError):
    """The assembly response stopped at max_output_tokens. Keeps the partial text and usage."""

    def __init__(self, message: str, raw_text: str = "", usage: dict | None = None,
                 duration: float = 0.0, finish_reason: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.usage = dict(usage or _ZERO_USAGE)
        self.duration = duration
        self.finish_reason = finish_reason


def _sentence_id_list(raw_ids) -> list[int]:
    return sorted({int(x) for x in (raw_ids or []) if isinstance(x, (int, str)) and str(x).strip().isdigit()})


def iter_candidates(chunk_results: list[dict]) -> list[tuple[str, int, int, dict]]:
    """Return (cand_ref, chunk_index, position, clip) for every chunk candidate.

    `cand_ref` is "<chunk>-<position>" (1-based). The assembly model uses it to refer to a candidate.
    """
    out = []
    for ci, chunk in enumerate(chunk_results, start=1):
        clips = chunk.get("final_edl", []) if isinstance(chunk, dict) else []
        for pos, clip in enumerate(clips, start=1):
            if isinstance(clip, dict):
                out.append((f"{ci}-{pos}", ci, pos, clip))
    return out


def format_candidates_for_assembly(chunk_results: list[dict]) -> str:
    """Render every chunk's candidate clips as a compact text list with their `cand` references."""
    lines = [
        "\n---",
        "## 各區段候選 clip（多模態模型的初步選擇；輸出時用 `cand` 編號引用）",
        "欄位：區段 / 候選序號 / cand / 角色 / 段落 / sentence_ids / 時間 / transcript / 選擇理由 / 視覺檢查",
    ]
    for ci, chunk in enumerate(chunk_results, start=1):
        clips = chunk.get("final_edl", []) if isinstance(chunk, dict) else []
        if not clips:
            lines.append(f"[區段 {ci}] （沒有候選）")
            continue
        for pos, c in enumerate(clips, start=1):
            if not isinstance(c, dict):
                continue
            sids = c.get("sentence_ids") or []
            lines.append(
                f"[區段 {ci} / 候選 {pos}] cand={ci}-{pos} role={c.get('role', 'host')} segment={c.get('segment', '')} "
                f"sentence_ids={sids} {float(c.get('source_in', 0.0) or 0.0):.2f}s->{float(c.get('source_out', 0.0) or 0.0):.2f}s\n"
                f"    transcript: {c.get('transcript', '')}\n"
                f"    reason: {c.get('take_selection_reason', '')}\n"
                f"    visual: {c.get('visual_check', '')}"
            )
    return "\n".join(lines)


def uncovered_host_sentences(chunk_results: list[dict], whisper_units: list[dict], min_duration: float = 3.0) -> list[dict]:
    """HOST sentences of at least `min_duration` seconds that no chunk candidate contains.

    The assembly model decides each of them (add as a new clip, or list in `dropped`), so that
    `find_unaccounted_sentences` only reports sentences that nobody decided on.
    """
    candidates = [{"sentence_ids": clip.get("sentence_ids") or []} for _, _, _, clip in iter_candidates(chunk_results)]
    return find_unaccounted_sentences(candidates, [], whisper_units, min_duration=min_duration)


def _output_directive(output_format: str, num_candidates: int, uncovered: list[dict] | None = None) -> str:
    if output_format == FORMAT_MINIMAL:
        return (
            "\n\n---\n"
            "## 重試：改用最精簡輸出（取代前面「輸出」一節）\n"
            "上一次組裝沒有成功（輸出過長被截斷，或格式錯誤）。這次每個 clip 只輸出 `cand`、`sentence_ids`、`role`、`segment` 四個欄位；"
            "不要輸出 `transcript`、`note`、`dropped`、`notes`。去重、場外角色、順序與段落標籤的判斷規則不變。\n"
            f"候選共 {num_candidates} 個；每個候選編號最多出現一次。\n"
            '格式：{"final_edl":[{"cand":"1-1","sentence_ids":[9,10,11],"role":"host","segment":"總開場"}]}\n'
        )
    text = ""
    if uncovered:
        text += (
            "\n\n---\n"
            f"## 候選沒有涵蓋的主持人長句（{len(uncovered)} 句）\n"
            "以下 [HOST] 句子不在任何候選裡。請逐句決定：是正式內容且沒有更好的版本 → 以新 clip 加入 `final_edl`（`cand` 填 null）；"
            '否則列在 `dropped`，寫成 {"sentence_ids":[3],"why":"NG 重講"}（相鄰而且理由相同的句子可以合併成一筆）。\n'
            + "\n".join(f"- Sentence ID {u['id']}（{u['start']:.2f}s->{u['end']:.2f}s）：{u['text']}" for u in uncovered)
            + "\n"
        )
    return text + (
        "\n\n---\n"
        "## 輸出提醒\n"
        f"候選共 {num_candidates} 個。依前面「輸出」一節的精簡格式回傳單一 JSON 物件：每個 clip 只有 `cand`、`sentence_ids`、`role`、`segment`，"
        "只有在刪減句子或新增 clip 時才加 `transcript`，只有在例外取捨時才加 `note`。不要回寫時間、理由、視覺或聲音檢查。\n"
    )


def build_assembly_prompt(
    prompt_file_candidates: list[Path],
    chunk_results: list[dict],
    whisper_units: list[dict],
    reference_script_path: Path | None = None,
    script_path: Path | None = None,
    output_format: str = FORMAT_COMPACT,
) -> str:
    """Assembly template + reference script + full sentence list + candidate clips + output directive."""
    base = build_prompt(
        assembly_prompt_candidates(prompt_file_candidates),
        script_path=script_path,
        whisper_units=whisper_units,
        reference_script_path=reference_script_path,
    )
    num_candidates = len(iter_candidates(chunk_results))
    uncovered = uncovered_host_sentences(chunk_results, whisper_units) if output_format == FORMAT_COMPACT else []
    return (base + format_candidates_for_assembly(chunk_results)
            + _output_directive(output_format, num_candidates, uncovered))


def _thinking_budget_for_assembly(num_candidates: int, num_sentences: int) -> int:
    return int(min(24576, max(4096, 60 * num_candidates + 20 * num_sentences)))


def _is_gemini3_or_newer(model: str) -> bool:
    m = re.match(r"(?:.*/)?gemini-(\d+)", str(model or "").strip().lower())
    return bool(m) and int(m.group(1)) >= 3


def _assembly_thinking_config(types_module, model: str, output_format: str, num_candidates: int, num_sentences: int):
    """Return (ThinkingConfig or None, description dict).

    Gemini 3 models: `thinking_level` (Google recommends it over the legacy `thinking_budget`,
    and the two cannot be combined). Older models: the legacy token budget.
    """
    if not hasattr(types_module, "ThinkingConfig"):
        return None, {}
    level_enum = getattr(types_module, "ThinkingLevel", None)
    if _is_gemini3_or_newer(model) and level_enum is not None:
        level_name = ASSEMBLY_THINKING_LEVEL.get(output_format, "MEDIUM")
        level = getattr(level_enum, level_name, None)
        if level is not None:
            return types_module.ThinkingConfig(thinking_level=level), {"thinking_level": level_name}
    budget = _thinking_budget_for_assembly(num_candidates, num_sentences)
    if output_format == FORMAT_MINIMAL:
        budget = max(2048, budget // 2)
    return types_module.ThinkingConfig(thinking_budget=budget), {"thinking_budget": budget}


def build_assembly_response_schema(types_module, output_format: str, num_candidates: int):
    """JSON schema of the assembly answer, or None when the types module has no Schema support.

    Short free-text fields have a length limit; `transcript` has none because the acoustic layer
    trims word boundaries to it. Arrays have no `max_items`: Vertex AI rejects gemini-3.8-flash
    requests with `max_items` 180 (400 INVALID_ARGUMENT, 2026-10-13 probe) while 5 is accepted.
    `num_candidates` is kept for the call signature.
    """
    schema_cls = getattr(types_module, "Schema", None)
    type_enum = getattr(types_module, "Type", None)
    if schema_cls is None or type_enum is None:
        return None
    S, T = schema_cls, type_enum
    clip_props = {
        "cand": S(type=T.STRING, nullable=True),
        "sentence_ids": S(type=T.ARRAY, items=S(type=T.INTEGER), min_items=1),
        "role": S(type=T.STRING, enum=list(VALID_ROLES)),
        "segment": S(type=T.STRING, max_length=40),
    }
    order = ["cand", "sentence_ids", "role", "segment"]
    if output_format == FORMAT_COMPACT:
        clip_props["transcript"] = S(type=T.STRING)
        clip_props["note"] = S(type=T.STRING, max_length=80)
        order += ["transcript", "note"]
    top = {
        "final_edl": S(
            type=T.ARRAY,
            items=S(type=T.OBJECT, properties=clip_props, required=["sentence_ids", "role", "segment"],
                    property_ordering=order),
        ),
    }
    if output_format == FORMAT_COMPACT:
        top["dropped"] = S(
            type=T.ARRAY,
            # A dropped candidate (`cand`) or dropped uncovered HOST sentences (`sentence_ids`)
            items=S(type=T.OBJECT,
                    properties={"cand": S(type=T.STRING, nullable=True),
                                "sentence_ids": S(type=T.ARRAY, items=S(type=T.INTEGER)),
                                "why": S(type=T.STRING, max_length=60)},
                    property_ordering=["cand", "sentence_ids", "why"]),
        )
        top["notes"] = S(type=T.STRING, max_length=300)
    return S(type=T.OBJECT, properties=top, required=["final_edl"], property_ordering=list(top))


def call_assembly_model(client, types_module, model: str, prompt: str, num_candidates: int, num_sentences: int,
                        output_format: str = FORMAT_COMPACT) -> dict:
    """One text-only Gemini call. Never raises on truncation; the caller reads `truncated`.

    Returns {raw_text, usage, duration, finish_reason, truncated, config}.
    Raises GeminiAPIError only when the API call itself fails.
    """
    config_kwargs = {
        "response_mime_type": "application/json",
        # Thinking tokens count against this limit and Gemini 3 does not cap them at the budget,
        # so keep the model maximum. The response format, not this limit, keeps the answer short.
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
    }
    config_desc: dict = {"format": output_format, "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS}
    if not _is_gemini3_or_newer(model):
        # Gemini 3: keep the default temperature 1.0. Google warns that a lower value can cause
        # looping on Gemini 3 models (https://ai.google.dev/gemini-api/docs/gemini-3).
        config_kwargs["temperature"] = 0.0
        config_desc["temperature"] = 0.0
    thinking_config, thinking_desc = _assembly_thinking_config(
        types_module, model, output_format, num_candidates, num_sentences
    )
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
        config_desc.update(thinking_desc)
    schema = build_assembly_response_schema(types_module, output_format, num_candidates)
    if schema is not None:
        config_kwargs["response_schema"] = schema
        config_desc["response_schema"] = True
    if hasattr(types_module, "AutomaticFunctionCallingConfig"):
        config_kwargs["automatic_function_calling"] = types_module.AutomaticFunctionCallingConfig(disable=True)

    logger.info("    [全片組裝推論] 純文字 | 格式=%s | 候選 clip=%d | 句子=%d | 思考=%s | max_output_tokens=%d",
                output_format, num_candidates, num_sentences,
                thinking_desc.get("thinking_level") or thinking_desc.get("thinking_budget") or "-",
                GEMINI_MAX_OUTPUT_TOKENS)
    t0 = time.time()
    try:
        response = _generate_content(
            client,
            model=model,
            contents=[prompt],
            config=types_module.GenerateContentConfig(**config_kwargs),
        )
    except Exception as e:  # noqa: BLE001
        raise GeminiAPIError(f"全片組裝推論失敗: {e}") from e

    raw_text = _extract_text_from_response(response) or ""
    usage = _usage_from_generate_content(response)
    finish_reason = None
    candidates = getattr(response, "candidates", None)
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        finish_reason = None if fr is None else str(fr)
    truncated = finish_reason is not None and finish_reason.upper().endswith("MAX_TOKENS")
    return {
        "raw_text": raw_text,
        "usage": usage,
        "duration": time.time() - t0,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "config": config_desc,
    }


def run_assembly_inference(client, types_module, model: str, prompt: str, num_candidates: int, num_sentences: int,
                           output_format: str = FORMAT_COMPACT):
    """Text-only Gemini call that returns (raw_json, usage, duration).

    Raises AssemblyTruncatedError (with the partial text and usage) at max_output_tokens.
    """
    res = call_assembly_model(client, types_module, model, prompt, num_candidates, num_sentences, output_format)
    if res["truncated"]:
        u = res["usage"]
        raise AssemblyTruncatedError(
            f"全片組裝輸出被 max_output_tokens 截斷 (finish_reason={res['finish_reason']}, "
            f"thoughts_tokens={u.get('thoughts_tokens', 0)}, candidates_tokens={u.get('candidates_tokens', 0)})",
            raw_text=res["raw_text"], usage=u, duration=res["duration"], finish_reason=res["finish_reason"],
        )
    return res["raw_text"], res["usage"], res["duration"]


def _candidate_index(chunk_results: list[dict]) -> tuple[dict, dict, dict]:
    """Index chunk candidates by `cand` reference, by exact sentence-ID tuple and by first sentence ID."""
    by_ref: dict[str, dict] = {}
    by_sids: dict[tuple, dict] = {}
    by_first_sid: dict[int, dict] = {}
    for ref, ci, pos, clip in iter_candidates(chunk_results):
        sids = _sentence_id_list(clip.get("sentence_ids"))
        entry = {"ref": ref, "chunk_index": ci, "position": pos, "clip": clip, "sids": sids}
        by_ref[ref] = entry
        if sids:
            by_sids.setdefault(tuple(sids), entry)
            by_first_sid.setdefault(sids[0], entry)
    return by_ref, by_sids, by_first_sid


def _exact_candidate_cover(sids: list[int], entries) -> list[dict] | None:
    """Return two or more non-overlapping candidates whose sentence IDs together equal `sids`, else None."""
    target = set(sids)
    subsets = sorted((e for e in entries if e["sids"] and set(e["sids"]) <= target), key=lambda e: e["sids"][0])
    remaining = set(target)
    chosen = []
    for e in subsets:
        if set(e["sids"]) <= remaining:
            chosen.append(e)
            remaining -= set(e["sids"])
    return chosen if (not remaining and len(chosen) > 1) else None


def normalize_assembled_edl(assembled: dict, whisper_units: list[dict], chunk_results: list[dict]) -> dict:
    """Validate and normalise the assembled EDL.

    1. Keep only clips whose `sentence_ids` exist and are not all CUE sentences.
    2. Recompute `source_in` / `source_out` from the sentences (never trust estimates).
    3. Take the clean `transcript`, `topic`, `take_selection_reason`, `visual_check` and `audio_check`
       from the chunk candidate with the same sentence IDs. A `transcript` from the assembly model
       wins (the model writes one only when it changes a candidate). Without both, join the
       transcripts of candidates that exactly cover the clip, else leave it empty (Whisper words).
    4. Inherit `visual_check` from a chunk candidate sharing the first sentence id when missing.
    5. Order by `timeline_position` when given, else by array order; renumber `clip_id` and
       `timeline_position` (1..N, unique).
    """
    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    by_ref, by_sids, by_first_sid = _candidate_index(chunk_results)

    clips_in = assembled.get("final_edl", []) if isinstance(assembled, dict) else []
    kept: list[dict] = []
    dropped: list[dict] = []
    for idx, raw in enumerate(clips_in):
        if not isinstance(raw, dict):
            dropped.append({"clip": raw, "reason": "not an object"})
            continue
        c = dict(raw)
        sids = [s for s in _sentence_id_list(c.get("sentence_ids")) if s in unit_by_id]
        if not sids:
            dropped.append({"clip": raw, "reason": "no valid sentence_ids"})
            continue
        units = [unit_by_id[s] for s in sids]
        if all(not u.get("is_target_speaker", True) for u in units):
            dropped.append({"clip": raw, "reason": "cue-only sentences"})
            continue
        role = str(c.get("role", "host")).lower()
        if role not in VALID_ROLES:
            role = "offscreen" if any(u.get("speaker_id") == "OFFSCREEN" for u in units) else "host"
        c["role"] = role
        c["sentence_ids"] = sids
        c["start_sentence_id"] = sids[0]
        c["end_sentence_id"] = sids[-1]
        c["source_in"] = round(float(units[0].get("start", 0.0)), 2)
        c["source_out"] = round(float(units[-1].get("end", 0.0)), 2)
        c["duration"] = round(c["source_out"] - c["source_in"], 2)

        ref = c.pop("cand", None)
        ref = str(ref).strip() if ref not in (None, "") else None
        note = str(c.pop("note", "") or "").strip()
        model_transcript = str(c.get("transcript") or "").strip()
        exact = by_sids.get(tuple(sids))
        if exact is not None:
            src = exact["clip"]
            if model_transcript:
                c["assembly_transcript_source"] = "assembly"
            else:
                c["transcript"] = src.get("transcript", "")
                c["assembly_transcript_source"] = "candidate"
            for field in ("topic", "take_selection_reason", "visual_check", "audio_check"):
                if not c.get(field) and src.get(field):
                    c[field] = src[field]
            anchor = exact
        else:
            cover = _exact_candidate_cover(sids, by_ref.values())
            if model_transcript:
                c["assembly_transcript_source"] = "assembly"
            elif cover:
                c["transcript"] = " ".join(str(e["clip"].get("transcript", "")).strip() for e in cover).strip()
                c["assembly_transcript_source"] = "candidates_joined"
            else:
                c["transcript"] = ""
                c["assembly_transcript_source"] = "whisper"
            if not c.get("topic"):
                topics = [str(e["clip"].get("topic", "")).strip() for e in (cover or [])]
                c["topic"] = " / ".join(t for t in topics if t) or str(c.get("segment", ""))
            anchor = (cover[0] if cover else None) or (by_ref.get(ref) if ref else None) or by_first_sid.get(sids[0])
        if not c.get("visual_check"):
            src_v = by_first_sid.get(sids[0])
            c["visual_check"] = src_v["clip"].get("visual_check", "未經視覺確認") if src_v else "未經視覺確認"
        if anchor is not None:
            c.setdefault("chunk_index", anchor["chunk_index"])
            c.setdefault("chunk_timeline_position", anchor["clip"].get("timeline_position"))
        c["assembly_candidate_ref"] = ref or (exact["ref"] if exact is not None else None)
        if note:
            c["assembly_note"] = note
            if not c.get("take_selection_reason"):
                c["take_selection_reason"] = note
        try:
            c["timeline_position"] = int(c.get("timeline_position", idx + 1))
        except (TypeError, ValueError):
            c["timeline_position"] = idx + 1
        c["_order"] = (c["timeline_position"], idx)
        kept.append(c)

    kept.sort(key=lambda c: c["_order"])
    for i, c in enumerate(kept, start=1):
        c.pop("_order", None)
        c["clip_id"] = i
        c["timeline_position"] = i

    # `dropped` holds candidate references (compact format) or sentence IDs (older format).
    model_dropped: list[dict] = []
    for d in (assembled.get("dropped", []) if isinstance(assembled, dict) else []) or []:
        if not isinstance(d, dict):
            continue
        entry = dict(d)
        d_ref = str(d.get("cand") or "").strip()
        if d_ref in by_ref and not d.get("sentence_ids"):
            entry["sentence_ids"] = list(by_ref[d_ref]["sids"])
        if "why" in entry and "reason" not in entry:
            entry["reason"] = entry["why"]
        model_dropped.append(entry)

    kept_sids: set[int] = set()
    for c in kept:
        kept_sids.update(c["sentence_ids"])
    dropped_refs = {str(d.get("cand")).strip() for d in model_dropped if d.get("cand")}
    unreferenced = [e["ref"] for e in by_ref.values()
                    if e["sids"] and e["ref"] not in dropped_refs and not (set(e["sids"]) & kept_sids)]
    if unreferenced:
        logger.warning("    [全片組裝] %d 個候選 clip 既未入選也未列在 dropped: %s",
                       len(unreferenced), ", ".join(unreferenced[:20]))

    unaccounted = find_unaccounted_sentences(kept, model_dropped, whisper_units)
    if unaccounted:
        logger.warning(
            "    [全片組裝] %d 句主持人正式長句既未入選也未列在 dropped，請人工確認是否遺漏: %s",
            len(unaccounted), ", ".join(f"S{u['id']}@{u['start']:.1f}s" for u in unaccounted[:12]),
        )

    out = {
        "project_title": assembled.get("project_title", "Show Mode 全片組裝") if isinstance(assembled, dict) else "Show Mode 全片組裝",
        "pacing_style": "text_based_whisper_grounded",
        "final_edl": kept,
        "assembly_notes": assembled.get("notes", "") if isinstance(assembled, dict) else "",
        "assembly_dropped": list(model_dropped) + dropped,
        "assembly_unaccounted": unaccounted,
        "assembly_unreferenced_candidates": unreferenced,
    }
    return out


def build_degraded_edl(chunk_results: list[dict], whisper_units: list[dict], reason: str) -> dict:
    """Fallback when the assembly pass fails: chunk results in source-time order.

    1. Merge every chunk candidate (keeps `chunk_index` and `chunk_timeline_position`).
    2. Sort by the start time of the first sentence (or `source_in`), stable.
    3. Renumber `clip_id` and `timeline_position` across the full project (1..N, unique).
    Cross-chunk duplicates stay; the Last-Take-Wins deduplication downstream removes shared sentences.
    """
    merged = merge_chunked_edl(chunk_results, project_title="Show Mode 全片組裝（降級：區段結果）")
    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}

    def _start(c: dict) -> float:
        sids = [s for s in _sentence_id_list(c.get("sentence_ids")) if s in unit_by_id]
        if sids:
            return float(unit_by_id[sids[0]].get("start", 0.0))
        try:
            return float(c.get("source_in", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    indexed = list(enumerate(merged["final_edl"]))
    indexed.sort(key=lambda t: (_start(t[1]), t[0]))
    clips = [c for _, c in indexed]
    for i, c in enumerate(clips, start=1):
        c["clip_id"] = i
        c["timeline_position"] = i
    return {
        "project_title": merged["project_title"],
        "pacing_style": "text_based_whisper_grounded",
        "final_edl": clips,
        "assembly_status": STATUS_DEGRADED,
        "assembly_error": reason,
    }


def _add_usage(total: dict, part: dict) -> None:
    for k in total:
        total[k] += int((part or {}).get(k, 0) or 0)


def assemble_with_fallback(
    client,
    types_module,
    model: str,
    prompt_file_candidates: list[Path],
    chunk_results: list[dict],
    whisper_units: list[dict],
    out_dir: Path,
    file_stem: str,
    reference_script_path: Path | None = None,
    script_path: Path | None = None,
    on_attempt=None,
) -> dict:
    """Run the assembly pass with one compact-format retry and a degraded fallback. Never raises for
    model failures.

    Attempt 1 uses the compact format. When it is truncated, fails, returns invalid JSON or keeps
    zero clips, attempt 2 uses the minimal format. When both fail, the chunk results are used in
    source-time order (`build_degraded_edl`).

    Every attempt writes `<file_stem>_assembly_raw.txt` (attempt 2: `_assembly_raw_retry.txt`) and
    updates `<file_stem>_assembly_meta.json` (status, finish_reason, usage, config, error).
    `on_attempt(record)` is called after each attempt (for example to append usage_log.jsonl).

    Returns {"model_edl", "info", "usage", "duration"}.
    """
    out_dir = Path(out_dir)
    num_candidates = len(iter_candidates(chunk_results))
    num_sentences = len(whisper_units)
    meta_path = out_dir / f"{file_stem}_assembly_meta.json"
    attempts: list[dict] = []
    total_usage = dict(_ZERO_USAGE)
    total_duration = 0.0
    model_edl = None
    status = STATUS_DEGRADED
    last_error = None

    def _write_meta(final: dict | None = None) -> None:
        meta = {"candidates": num_candidates, "sentences": num_sentences, "attempts": attempts}
        if final:
            meta.update(final)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for attempt_no, output_format in enumerate(ASSEMBLY_ATTEMPT_FORMATS, start=1):
        raw_name = f"{file_stem}_assembly_raw.txt" if attempt_no == 1 else f"{file_stem}_assembly_raw_retry.txt"
        rec: dict = {"attempt": attempt_no, "format": output_format, "raw_file": raw_name, "status": None,
                     "finish_reason": None, "usage": dict(_ZERO_USAGE), "duration_seconds": 0.0,
                     "config": {}, "error": None, "clips": 0}
        raw_text = ""
        try:
            prompt = build_assembly_prompt(prompt_file_candidates, chunk_results, whisper_units,
                                           reference_script_path=reference_script_path,
                                           script_path=script_path, output_format=output_format)
            res = call_assembly_model(client, types_module, model, prompt, num_candidates, num_sentences,
                                      output_format=output_format)
        except Exception as e:  # noqa: BLE001  (API failure: record it and continue with the next attempt)
            rec.update(status="api_error", error=f"{type(e).__name__}: {e}")
        else:
            raw_text = res["raw_text"]
            rec.update(finish_reason=res["finish_reason"], usage=res["usage"],
                       duration_seconds=round(res["duration"], 2), config=res["config"], raw_chars=len(raw_text))
            _add_usage(total_usage, res["usage"])
            total_duration += res["duration"]
            if res["truncated"]:
                u = res["usage"]
                rec.update(status="truncated",
                           error=(f"輸出被 max_output_tokens 截斷 (finish_reason={res['finish_reason']}, "
                                  f"thoughts_tokens={u.get('thoughts_tokens', 0)}, "
                                  f"candidates_tokens={u.get('candidates_tokens', 0)})"))
            else:
                try:
                    normalized = normalize_assembled_edl(parse_assembly_json(raw_text), whisper_units, chunk_results)
                except Exception as e:  # noqa: BLE001  (invalid JSON or shape: try the next format)
                    rec.update(status="parse_error", error=f"{type(e).__name__}: {str(e)[:500]}")
                else:
                    rec["clips"] = len(normalized.get("final_edl", []))
                    if rec["clips"] == 0:
                        rec.update(status="empty", error="組裝結果沒有任何有效 clip")
                    else:
                        rec["status"] = "ok"
                        model_edl = normalized
        (out_dir / raw_name).write_text(raw_text, encoding="utf-8")
        attempts.append(rec)
        _write_meta()
        if on_attempt is not None:
            on_attempt(dict(rec))
        if model_edl is not None:
            status = STATUS_ASSEMBLED if attempt_no == 1 else STATUS_ASSEMBLED_AFTER_RETRY
            break
        last_error = rec["error"]
        if attempt_no < len(ASSEMBLY_ATTEMPT_FORMATS):
            logger.warning("    [全片組裝] 第 %d 次失敗（%s）：%s；改用最精簡格式重試一次",
                           attempt_no, rec["status"], rec["error"])

    if model_edl is None:
        reason = f"兩次全片組裝都失敗；最後一次：{last_error}"
        logger.warning("    [全片組裝] %s", reason)
        model_edl = build_degraded_edl(chunk_results, whisper_units, reason)
    model_edl["assembly_status"] = status

    info = {
        "status": status,
        "degraded": status == STATUS_DEGRADED,
        "error": last_error if status == STATUS_DEGRADED else None,
        "candidates": num_candidates,
        "final_clips": len(model_edl.get("final_edl", [])),
        "attempts": attempts,
        "meta_file": meta_path.name,
        "raw_files": [a["raw_file"] for a in attempts],
        "usage": dict(total_usage),
        "duration_seconds": round(total_duration, 2),
    }
    _write_meta({"status": status, "degraded": info["degraded"], "error": info["error"],
                 "final_clips": info["final_clips"], "usage": info["usage"]})
    return {"model_edl": model_edl, "info": info, "usage": total_usage, "duration": total_duration}


_CHUNK_RAW_RE = re.compile(r"^(?P<prefix>.+_chunk)(?P<num>\d+)_raw\.txt$")


def load_chunk_results(path: Path) -> tuple[list[dict], list[Path]]:
    """Load saved chunk results so that the assembly step can run again without video inference.

    `path` is one of:
    1. A `*_chunks_raw.json` file (list of chunk EDL objects, written by an `--assemble` run).
    2. A `*_chunk1_raw.txt` file: chunk 1, 2, ... are read until a number is missing.
    3. A folder with exactly one set of `*_chunk<N>_raw.txt` files.
    Returns (chunk_results, source_files).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise InvalidInputError(f"找不到區段結果: {p}")
    if p.is_file() and p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            data = data["chunks"]
        if not isinstance(data, list) or not all(isinstance(c, dict) for c in data):
            raise InvalidInputError(f"區段結果 JSON 必須是區段物件的陣列: {p}")
        return [c if "final_edl" in c else {"final_edl": []} for c in data], [p]

    if p.is_dir():
        groups: dict[str, list[tuple[int, Path]]] = {}
        for f in p.iterdir():
            m = _CHUNK_RAW_RE.match(f.name)
            if m and not f.name.startswith("._"):
                groups.setdefault(m.group("prefix"), []).append((int(m.group("num")), f))
        if len(groups) != 1:
            raise InvalidInputError(
                f"資料夾內有 {len(groups)} 組區段結果（{', '.join(sorted(groups)) or '無'}），請改指定某一組的 *_chunk1_raw.txt"
            )
        prefix = next(iter(groups))
        first = p / f"{prefix}1_raw.txt"
    else:
        m = _CHUNK_RAW_RE.match(p.name)
        if not m:
            raise InvalidInputError(f"無法辨識的區段結果檔名（需要 *_chunks_raw.json 或 *_chunk1_raw.txt）: {p.name}")
        prefix = m.group("prefix")
        first = p.parent / f"{prefix}1_raw.txt"
    files: list[Path] = []
    n = 1
    while (first.parent / f"{prefix}{n}_raw.txt").exists():
        files.append(first.parent / f"{prefix}{n}_raw.txt")
        n += 1
    if not files:
        raise InvalidInputError(f"找不到 {prefix}1_raw.txt")
    return [parse_edl_json_text(f.read_text(encoding="utf-8")) for f in files], files


def find_unaccounted_sentences(
    final_edl: list[dict],
    dropped: list[dict],
    whisper_units: list[dict],
    min_duration: float = 3.0,
) -> list[dict]:
    """HOST sentences longer than min_duration that are neither selected nor explicitly dropped.

    The assembly model sometimes omits a sentence silently (no entry in `dropped`); such
    omissions are content losses until a human confirms them, so they are surfaced here.
    """
    selected: set[int] = set()
    for c in final_edl:
        selected.update(int(x) for x in (c.get("sentence_ids") or []) if str(x).isdigit())
    explicit: set[int] = set()
    for d in dropped or []:
        explicit.update(int(x) for x in (d.get("sentence_ids") or []) if str(x).isdigit())
    out = []
    for u in whisper_units:
        uid = int(u.get("id", 0))
        if uid in selected or uid in explicit:
            continue
        if u.get("speaker_id") not in (None, "HOST", "SPEAKER_00"):
            continue
        if not u.get("is_target_speaker", True):
            continue
        dur = float(u.get("end", 0.0)) - float(u.get("start", 0.0))
        if dur < min_duration:
            continue
        out.append({"id": uid, "start": round(float(u.get("start", 0.0)), 2), "end": round(float(u.get("end", 0.0)), 2),
                    "text": str(u.get("text", ""))[:80]})
    return out


def parse_assembly_json(raw_json: str) -> dict:
    """Parse the JSON text returned by the assembly pass (object, bare array or fenced)."""
    return parse_edl_json_text(raw_json)

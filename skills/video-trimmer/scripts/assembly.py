"""Assembly pass: global, text-only arbitration across chunk results (Show Mode).

ASD-STE100:
Long recordings are analysed in several video chunks. Each chunk returns candidate
clips without knowledge of the other chunks. This module sends all candidate clips,
the complete sentence list (with mic roles) and the reference script to Gemini in a
single text-only request, and receives the final ordered EDL.

The pass never uploads video. It only resolves: cross-chunk duplicate takes, missing
sentences, off-screen voice-role decisions, segment labels and timeline order.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .exceptions import GeminiAPIError
from .gemini_client import (
    _extract_text_from_response,
    _generate_content,
    _usage_from_generate_content,
    build_prompt,
    parse_edl_json_text,
)
from .constants import GEMINI_MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)

ASSEMBLY_PROMPT_NAME = "assembly_prompt.md"
VALID_ROLES = ("host", "offscreen")


def assembly_prompt_candidates(base_candidates: list[Path]) -> list[Path]:
    """Map the video-cut prompt candidate paths to assembly prompt paths in the same folders."""
    seen: list[Path] = []
    for p in base_candidates:
        cand = Path(p).parent / ASSEMBLY_PROMPT_NAME
        if cand not in seen:
            seen.append(cand)
    return seen


def format_candidates_for_assembly(chunk_results: list[dict]) -> str:
    """Render every chunk's candidate clips as a compact text list."""
    lines = [
        "\n---",
        "## 各區段候選 clip（多模態模型的初步選擇）",
        "欄位：區段 / 候選編號 / 角色 / 段落 / sentence_ids / 時間 / transcript / 選擇理由 / 視覺檢查",
    ]
    for ci, chunk in enumerate(chunk_results, start=1):
        clips = chunk.get("final_edl", []) if isinstance(chunk, dict) else []
        if not clips:
            lines.append(f"[區段 {ci}] （沒有候選）")
            continue
        for c in clips:
            sids = c.get("sentence_ids") or []
            lines.append(
                f"[區段 {ci} / 候選 {c.get('clip_id', '?')}] role={c.get('role', 'host')} segment={c.get('segment', '')} "
                f"sentence_ids={sids} {float(c.get('source_in', 0.0)):.2f}s->{float(c.get('source_out', 0.0)):.2f}s\n"
                f"    transcript: {c.get('transcript', '')}\n"
                f"    reason: {c.get('take_selection_reason', '')}\n"
                f"    visual: {c.get('visual_check', '')}"
            )
    return "\n".join(lines)


def build_assembly_prompt(
    prompt_file_candidates: list[Path],
    chunk_results: list[dict],
    whisper_units: list[dict],
    reference_script_path: Path | None = None,
    script_path: Path | None = None,
) -> str:
    """Assembly template + reference script + full sentence list + candidate clips."""
    base = build_prompt(
        assembly_prompt_candidates(prompt_file_candidates),
        script_path=script_path,
        whisper_units=whisper_units,
        reference_script_path=reference_script_path,
    )
    return base + format_candidates_for_assembly(chunk_results)


def _thinking_budget_for_assembly(num_candidates: int, num_sentences: int) -> int:
    return int(min(24576, max(4096, 60 * num_candidates + 20 * num_sentences)))


def run_assembly_inference(client, types_module, model: str, prompt: str, num_candidates: int, num_sentences: int):
    """Text-only Gemini call that returns (raw_json, usage, duration)."""
    budget = _thinking_budget_for_assembly(num_candidates, num_sentences)
    config_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
    }
    if hasattr(types_module, "ThinkingConfig"):
        config_kwargs["thinking_config"] = types_module.ThinkingConfig(thinking_budget=budget)
    if hasattr(types_module, "AutomaticFunctionCallingConfig"):
        config_kwargs["automatic_function_calling"] = types_module.AutomaticFunctionCallingConfig(disable=True)

    logger.info("    [全片組裝推論] 純文字 | 候選 clip=%d | 句子=%d | 思維預算=%d", num_candidates, num_sentences, budget)
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

    raw_json = _extract_text_from_response(response)
    usage = _usage_from_generate_content(response)
    candidates = getattr(response, "candidates", None)
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None and str(finish_reason).upper().endswith("MAX_TOKENS"):
            raise GeminiAPIError(f"全片組裝輸出被 max_output_tokens 截斷 (finish_reason={finish_reason})")
    return raw_json, usage, time.time() - t0


def normalize_assembled_edl(assembled: dict, whisper_units: list[dict], chunk_results: list[dict]) -> dict:
    """Validate and normalise the assembled EDL.

    1. Keep only clips whose `sentence_ids` exist and are not all CUE sentences.
    2. Recompute `source_in` / `source_out` from the sentences (never trust estimates).
    3. Inherit `visual_check` from a chunk candidate sharing the first sentence id when missing.
    4. Order by `timeline_position` (stable), renumber `clip_id` and `timeline_position`.
    """
    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    cand_by_first_sid: dict[int, dict] = {}
    for chunk in chunk_results:
        for c in (chunk.get("final_edl", []) if isinstance(chunk, dict) else []):
            sids = [int(x) for x in (c.get("sentence_ids") or []) if str(x).isdigit()]
            if sids:
                cand_by_first_sid.setdefault(sids[0], c)

    clips_in = assembled.get("final_edl", []) if isinstance(assembled, dict) else []
    kept: list[dict] = []
    dropped: list[dict] = []
    for idx, raw in enumerate(clips_in):
        c = dict(raw)
        sids = sorted({int(x) for x in (c.get("sentence_ids") or []) if isinstance(x, (int, str)) and str(x).isdigit()})
        sids = [s for s in sids if s in unit_by_id]
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
        if not c.get("visual_check"):
            src = cand_by_first_sid.get(sids[0])
            c["visual_check"] = src.get("visual_check", "未經視覺確認") if src else "未經視覺確認"
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

    out = {
        "project_title": assembled.get("project_title", "Show Mode 全片組裝") if isinstance(assembled, dict) else "Show Mode 全片組裝",
        "pacing_style": "text_based_whisper_grounded",
        "final_edl": kept,
        "assembly_notes": assembled.get("notes", "") if isinstance(assembled, dict) else "",
        "assembly_dropped": (assembled.get("dropped", []) if isinstance(assembled, dict) else []) + dropped,
    }
    return out


def parse_assembly_json(raw_json: str) -> dict:
    """Parse the JSON text returned by the assembly pass (object, bare array or fenced)."""
    return parse_edl_json_text(raw_json)

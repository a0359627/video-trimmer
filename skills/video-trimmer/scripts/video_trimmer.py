#!/usr/bin/env python3
"""
video-trimmer (AI-Powered Smart Video Trimmer)
---------------------------------------------
An end-to-end video rough-cut and trimming engine powered by:
- Whisper word-level acoustic ground truth (Apple Silicon mlx-whisper & faster-whisper)
- Vertex AI Gemini 3.8 Flash multimodal video understanding and dual-mode take selection
- Acoustic onset snapping (80ms pre-vocal onset) and plosive tail defense
- Adaptive noise floor tracking and 20ms equal-power audio micro-crossfades
- Export to Premiere Pro (.xml), DaVinci Resolve (.xml), Final Cut Pro (.fcpxml), CSV (.csv), and MP4.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np  # noqa: F401  (used transitively by acoustic.py; kept for early dependency check)
    from google.genai import types
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format="%(message)s", stream=sys.stderr)
    logging.error("Missing required package: %s", e)
    logging.error("Run: pip install -r requirements.txt")
    sys.exit(1)

if __package__ in (None, ""):
    _skill_root = Path(__file__).resolve().parent.parent
    if str(_skill_root) not in sys.path:
        sys.path.insert(0, str(_skill_root))
    __package__ = "scripts"

from .acoustic import refine_speech_bounds_locked
from .constants import ALLOWED_INPUT_EXTENSIONS
from .edl_auditor import (
    audit_edl_quality,
    deduplicate_and_sort_clips,
    generate_edl_audit_markdown,
    repair_edl_micro_windows,
    sanitize_refined_edl,
)
from .exceptions import GeminiAPIError, InvalidInputError, VideoTrimmerError
from .exporters import (
    XML_PROFILES,
    generate_edl_csv,
    generate_edl_report,
    generate_fcp7_xml,
    generate_fcpxml,
    get_video_timecode,
)
from .gcs_utils import delete_gcs_blob, is_gdrive_source, download_gdrive_file_with_cache
from .gemini_client import (
    build_micro_window_repair_prompt,
    build_prompt,
    get_gemini_client,
    load_env_file,
    merge_chunked_edl,
    parse_edl_json_text,
    run_gemini_inference,
    stage_video_to_gcs,
)
from .render import probe_video, render_cut_video
from .assembly import (
    DEGRADED_BANNER,
    STATUS_DEGRADED,
    assemble_with_fallback,
    iter_candidates,
    load_chunk_results,
)
from .transcribe import (
    calculate_active_script_window,
    coalesce_adjacent_sub_units,
    detect_chunk_boundaries,
    resolve_clip_sub_units,
    transcribe_video_whisper,
    trim_cross_clip_seam_overlaps,
    normalize_text,
)

logger = logging.getLogger(__name__)


def _parse_gemini_json_text(raw_json: str) -> dict:
    """Clean markdown fences and parse JSON returned by Gemini (object, bare array, bracketless objects or single clip)."""
    try:
        return parse_edl_json_text(raw_json)
    except GeminiAPIError as e:
        raise VideoTrimmerError(str(e)) from e


ORPHAN_WORDS = {"所以", "但是", "而且", "然而", "如果", "因為", "不過", "雖然", "或是", "或者", "好", "對", "ok", "OK"}


def _prune_orphan_units(units: list[dict]) -> list[dict]:
    """
    Drop isolated conjunction or particle fragments (Orphan Conjunction / Particle Filter).

    ASD-STE100:
    Remove a unit when its normalized text has at most 2 characters, is in ORPHAN_WORDS
    (for example 「所以」, 「但是」, 「好」), and lasts less than 2.5s.
    Keep Show Mode off-screen units (role == "offscreen"): the master prompt decides
    whether a short off-screen reaction stays in the cut.
    """
    pruned = []
    for u in units:
        if str(u.get("role", "host")).lower() == "offscreen":
            pruned.append(u)
            continue
        norm_t = normalize_text(u.get("transcript", ""))
        dur_est = float(u.get("t_last", 0.0)) - float(u.get("t_first", 0.0))
        if len(norm_t) <= 2 and norm_t in ORPHAN_WORDS and dur_est < 2.5:
            logger.info("    [Orphan Filter] Dropped isolated fragment ('%s', estimated %.2fs).", u.get("transcript"), dur_est)
            continue
        pruned.append(u)
    return pruned


def _setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def _validate_input_extension(video_path):
    if video_path.suffix.lower() not in ALLOWED_INPUT_EXTENSIONS:
        allowed = ", ".join(ALLOWED_INPUT_EXTENSIONS)
        raise InvalidInputError(f"Unsupported video format: {video_path.suffix} (allowed extensions: {allowed})")


def _append_usage_log(out_dir, video_path, model, usage, mode, duration, extra=None):
    """Append token usage metrics for each Gemini inference run to usage_log.jsonl.

    The lines add up to the total cost: an assembly attempt is logged on its own line
    (mode "assembly"), and the run line then excludes it (scope "chunks_and_repairs").
    """
    usage_log_path = out_dir / "usage_log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video": video_path.name,
        "model": model,
        "mode": mode,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "candidates_tokens": usage.get("candidates_tokens", 0),
        "thoughts_tokens": usage.get("thoughts_tokens", 0),
        "tool_use_tokens": usage.get("tool_use_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "duration_seconds": round(duration, 2),
    }
    if extra:
        entry.update(extra)
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _merge_chunk_results(chunk_edl_results: list[dict], project_title: str) -> dict:
    """Merge chunk EDLs; `timeline_position` becomes unique across the project (chunk value kept in
    `chunk_timeline_position`). A single chunk keeps its own top-level keys."""
    merged = merge_chunked_edl(chunk_edl_results, project_title=project_title)
    if len(chunk_edl_results) == 1 and isinstance(chunk_edl_results[0], dict):
        single = dict(chunk_edl_results[0])
        single["final_edl"] = merged["final_edl"]
        return single
    return merged


def _run_show_assembly(args, client, chunk_edl_results, whisper_units, prompt_candidates, reference_script_path,
                       script_path, out_dir, base_name, tag, video_path):
    """Step 4a: Show Mode full-programme assembly (text only) with retry and degraded fallback.

    Writes `<base>_<tag>_chunks_raw.json`, the raw assembly output and `<base>_<tag>_assembly_meta.json`,
    and appends one usage_log line per attempt. Returns (model_edl, info, usage, duration).
    """
    (out_dir / f"{base_name}_{tag}_chunks_raw.json").write_text(
        json.dumps(chunk_edl_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_candidates = len(iter_candidates(chunk_edl_results))

    def _log_attempt(rec):
        _append_usage_log(out_dir, video_path, args.model, rec.get("usage") or {}, "assembly",
                          float(rec.get("duration_seconds") or 0.0),
                          extra={"scope": "assembly", "attempt": rec.get("attempt"), "format": rec.get("format"),
                                 "status": rec.get("status"), "finish_reason": rec.get("finish_reason")})

    result = assemble_with_fallback(
        client, types, args.model, prompt_candidates, chunk_edl_results, whisper_units,
        out_dir=out_dir, file_stem=f"{base_name}_{tag}",
        reference_script_path=reference_script_path, script_path=script_path,
        on_attempt=_log_attempt,
    )
    info = result["info"]
    if info["degraded"]:
        logger.warning("    [全片組裝] %s（候選 %d -> %d clips）。原因：%s",
                       DEGRADED_BANNER, n_candidates, info["final_clips"], info["error"])
    else:
        logger.info("    [全片組裝] %s：候選 %d -> 最終 %d clips（%.1f 秒）",
                    info["status"], n_candidates, info["final_clips"], result["duration"])
    return result["model_edl"], info, result["usage"], result["duration"]


def _assembly_report_line(info: dict) -> str:
    if info.get("degraded"):
        return f"{DEGRADED_BANNER}。原因：{info.get('error')}"
    tries = len(info.get("attempts", []))
    return f"全片組裝成功（{info.get('status')}，第 {tries} 次嘗試）：候選 {info.get('candidates')} -> {info.get('final_clips')} clips"


def _prepend_assembly_banner(md_path: Path, info: dict) -> None:
    """Insert the assembly status under the first heading of a Markdown report."""
    if not md_path.exists():
        return
    lines = md_path.read_text(encoding="utf-8").splitlines()
    prefix = "> **警告：" if info.get("degraded") else "> "
    banner = [f"{prefix}{_assembly_report_line(info)}{'**' if info.get('degraded') else ''}",
              f"> 組裝原始輸出與 usage：{'、'.join([*info.get('raw_files', []), info.get('meta_file', '')])}", ""]
    insert_at = 1 if lines and lines[0].startswith("#") else 0
    if insert_at:
        banner = [""] + banner
    md_path.write_text("\n".join(lines[:insert_at] + banner + lines[insert_at:]) + "\n", encoding="utf-8")


def _video_cut_prompt_candidates(args) -> list[Path]:
    _resolved_file = Path(__file__).resolve()
    return [
        *([Path(args.prompt).expanduser().resolve()] if getattr(args, "prompt", None) else []),
        Path(__file__).parent / "prompts" / "video_cut_prompt.md",
        Path(__file__).parent.parent / "prompts" / "video_cut_prompt.md",
        Path.cwd() / "prompts" / "video_cut_prompt.md",
        Path(__file__).parent / "video-trimmer" / "prompts" / "video_cut_prompt.md",
        *[p / "prompts" / "video_cut_prompt.md" for p in _resolved_file.parents[:5]],
    ]


def _check_reused_chunk_sentences(chunk_files: list[Path], whisper_units: list[dict], base_name: str) -> None:
    """Fail closed when reused chunk results were made with different Sentence IDs.

    The chunk results refer to sentences by ID. When the saved `<stem>_whisper_sentences.json` next to
    them differs from the current sentences (id / start / end), the IDs point at different audio.
    """
    saved = chunk_files[0].parent / f"{base_name}_whisper_sentences.json" if chunk_files else None
    if saved is None or not saved.exists():
        logger.warning("    [重用區段] 找不到區段結果旁的 %s_whisper_sentences.json，無法核對句子編號；請確認與目前轉錄一致",
                       base_name)
        return
    try:
        old_units = json.loads(saved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InvalidInputError(f"無法讀取 {saved}: {e}") from e

    def _sig(units):
        return [(int(u.get("id", -1)), round(float(u.get("start", 0.0)), 2), round(float(u.get("end", 0.0)), 2))
                for u in units if isinstance(u, dict)]

    if _sig(old_units) != _sig(whisper_units):
        raise InvalidInputError(
            f"重用的區段結果與目前的 Whisper 句子不一致（{saved.name}: {len(old_units)} 句，目前 {len(whisper_units)} 句），"
            "句子編號會對到錯的時間。請沿用產生區段結果時的 _whisper_raw.json 與相同的轉錄設定。"
        )
    logger.info("    [重用區段] 句子編號核對一致（%d 句）", len(whisper_units))


def _resolve_output_dir(output_dir_arg: str | None, default_parent: Path) -> Path:
    """Resolve the output directory and isolate deliverables in <default_parent>/output by default."""
    if output_dir_arg:
        out_dir = Path(output_dir_arg).expanduser().resolve()
    else:
        out_dir = default_parent.resolve() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run(args):
    if is_gdrive_source(args.input):
        out_dir = _resolve_output_dir(args.output_dir, Path.cwd())
        try:
            video_path = download_gdrive_file_with_cache(
                args.input,
                target_dir=out_dir / "gdrive_inputs",
                project_id=args.project,
            )
        except Exception as e:
            raise InvalidInputError(f"Failed to download video from Google Drive: {e}") from e
    else:
        video_path = Path(args.input).expanduser().resolve()
        if not video_path.exists():
            raise InvalidInputError(f"Video file not found: {video_path}")
        out_dir = _resolve_output_dir(args.output_dir, video_path.parent)

    _validate_input_extension(video_path)
    base_name = video_path.stem

    tag = args.suffix if args.suffix else ("agentic" if args.agentic else "static")

    logger.info("==> 1. Probing input video: %s", video_path.name)
    logger.info("    Output directory: %s", out_dir)
    total_dur, width, height, fps = probe_video(video_path)
    logger.info("    Duration: %.1f s (~%.1f min) | Resolution: %dx%d | Frame rate: %.3f fps",
                total_dur, total_dur / 60, width, height, fps)

    # Layer 1: Whisper word-level acoustic transcription and clause segmentation
    whisper_segs = []
    whisper_sentences = []

    if not args.skip_whisper:
        whisper_json = out_dir / f"{base_name}_whisper_raw.json"
        logger.info("==> 2. Running Whisper word-level transcription and clause segmentation...")
        whisper_segs, whisper_sentences = transcribe_video_whisper(
            video_path,
            whisper_json,
            model_name=getattr(args, "whisper_model", None),
            offscreen_channel=getattr(args, "offscreen_channel", None),
        )

    whisper_units = whisper_sentences if whisper_sentences else whisper_segs

    if getattr(args, "transcribe_only", False):
        role_counts: dict[str, int] = {}
        for u in whisper_units:
            key = str(u.get("speaker_id", "SPEAKER_00"))
            role_counts[key] = role_counts.get(key, 0) + 1
        logger.info("==> [transcribe-only] 轉錄完成：%d 句 | 角色統計: %s | 輸出目錄: %s",
                    len(whisper_units), role_counts, out_dir)
        return

    # Show Mode 參數：Mode C 參考腳本、重排、代理分析檔
    reference_script_path = Path(args.reference_script).resolve() if getattr(args, "reference_script", None) else None
    reference_script_text = (
        reference_script_path.read_text(encoding="utf-8")
        if (reference_script_path is not None and reference_script_path.exists()) else None
    )
    allow_reorder = bool(getattr(args, "allow_reorder", False))

    # Layer 2: Gemini multimodal video understanding and dual-mode take arbitration
    inference_metrics = {}
    micro_window_repairs = []
    script_path = Path(args.script).resolve() if args.script else None
    script_text = script_path.read_text(encoding="utf-8") if (script_path is not None and script_path.exists()) else None
    # Mode C：腳本只作結構參考。Mode A 的覆蓋率檢查、碰撞修復、腳本去重與「講稿視窗鎖定」一律關閉，
    # 因為脫稿的開場或結尾不在腳本裡，鎖定視窗會把它們切掉。
    audit_script_text = None if reference_script_text is not None else script_text
    window_script_text = script_text

    assembly_info = None
    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            raise InvalidInputError(f"Cached JSON file not found: {cached_file}")
        logger.info("==> 3. Skipping Gemini inference and loading cached EDL: %s", cached_file.name)
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    elif getattr(args, "reuse_chunks", None):
        # Re-run only the assembly step from saved chunk results: no video upload and no chunk inference.
        chunk_edl_results, chunk_files = load_chunk_results(Path(args.reuse_chunks))
        logger.info("==> 3. 重用既有區段結果（不上傳影片、不重跑區段推論）: %s", ", ".join(f.name for f in chunk_files))
        _check_reused_chunk_sentences(chunk_files, whisper_units, base_name)
        usage = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "tool_use_tokens": 0, "total_tokens": 0}
        duration = 0.0
        if getattr(args, "assemble", False) and whisper_units:
            load_env_file()
            client = get_gemini_client(project_id=args.project, location=args.region)
            logger.info("==> 4. 只重跑全片組裝（%s，純文字）", args.model)
            model_edl, assembly_info, a_usage, a_dur = _run_show_assembly(
                args, client, chunk_edl_results, whisper_units, _video_cut_prompt_candidates(args),
                reference_script_path, script_path, out_dir, base_name, tag, video_path,
            )
            duration += a_dur
            for k in usage:
                usage[k] += a_usage.get(k, 0)
        else:
            model_edl = _merge_chunk_results(chunk_edl_results, project_title=f"{base_name} AI Video Trimmer ({tag})")
        logger.warning("    [重用區段] 沒有上傳影片，略過 Micro-Window Repair（局部影片重掃）")
        inference_metrics = {
            "mode": "reuse_chunks",
            "duration_seconds": round(duration, 2),
            **{k: usage[k] for k in usage},
            "micro_window_repairs": 0,
            "micro_window_repair_skipped": True,
            "reused_chunk_files": [f.name for f in chunk_files],
        }
    else:
        load_env_file()
        client = get_gemini_client(project_id=args.project, location=args.region)

        prompt_candidates = _video_cut_prompt_candidates(args)

        # Step 3a: Locate the active recording window when a reference script is provided on long footage
        win_start, win_end, active_units = 0.0, total_dur, whisper_units
        if window_script_text is not None and total_dur > 600.0 and whisper_units:
            win_start, win_end, active_units = calculate_active_script_window(
                whisper_units=whisper_units,
                script_text=window_script_text,
                total_duration=total_dur,
                padding_seconds=2.0,
            )
            if win_start > 0.0 or win_end < total_dur:
                logger.info(
                    "    [Active Script Window] Focused recording window: %.1fs ~ %.1fs (span %.1fs, %d sentences)",
                    win_start,
                    win_end,
                    win_end - win_start,
                    len(active_units),
                )

        # Step 3b: Split long recordings (>660s) at natural pauses (>=2.0s) without splitting retake clusters
        chunks = detect_chunk_boundaries(
            whisper_units=active_units,
            total_duration=win_end,
            target_chunk_duration=480.0,
            min_pause_duration=2.0,
            max_chunk_duration=660.0,
            window_start=win_start,
        )

        resolved_bucket = (
            args.bucket
            or os.environ.get("VIDEO_TRIMMER_BUCKET")
            or os.environ.get("VIDEO_STORAGE_BUCKET")
            or os.environ.get("MEETING_STORAGE_BUCKET")
            or os.environ.get("GCS_BUCKET")
            or ""
        ).removeprefix("gs://") or None

        analysis_source = video_path
        if getattr(args, "analysis_video", None):
            analysis_source = Path(args.analysis_video).expanduser().resolve()
            if not analysis_source.exists():
                raise InvalidInputError(f"找不到 --analysis-video 代理檔: {analysis_source}")
            logger.info("    [代理分析] 上傳與 Gemini 分析使用 %s；Whisper 時間戳與 XML 仍對原檔 %s",
                        analysis_source.name, video_path.name)

        logger.info("==> 3. Staging video to Cloud Storage...")
        gcs_uri, mime_type, is_ephemeral = stage_video_to_gcs(analysis_source, resolved_bucket)

        mode_str = "Agentic Video Understanding" if args.agentic else "Static Multimodal"
        logger.info("==> 4. Running %s [Mode: %s] with Whisper sentences for take arbitration...", args.model, mode_str)

        usage = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "tool_use_tokens": 0, "total_tokens": 0}
        assembly_usage = dict(usage)
        assembly_dur = 0.0
        duration = 0.0
        chunk_edl_results = []

        try:
            if len(chunks) > 1:
                logger.info("    [Auto-Chunking] Splitting %.1fs active span into %d natural-pause chunks...", win_end - win_start, len(chunks))

            for idx, (c_start, c_end, c_units) in enumerate(chunks, start=1):
                c_prompt = build_prompt(
                    prompt_candidates,
                    script_path=script_path,
                    whisper_units=c_units,
                    reference_script_path=reference_script_path,
                )
                use_window_meta = (c_start > 0.0) or (c_end < total_dur - 1.0) or (len(chunks) > 1)
                start_offset = f"{int(c_start)}s" if use_window_meta else None
                end_offset = f"{int(math.ceil(c_end))}s" if use_window_meta else None

                if len(chunks) > 1:
                    logger.info(
                        "    ---> Chunk %d/%d: Window %s ~ %s (%.1fs, %d sentences)",
                        idx,
                        len(chunks),
                        start_offset,
                        end_offset,
                        c_end - c_start,
                        len(c_units),
                    )

                c_raw_json, c_usage, c_dur = run_gemini_inference(
                    client,
                    types,
                    args.model,
                    gcs_uri,
                    mime_type,
                    c_prompt,
                    args.agentic,
                    num_sentences=max(1, len(c_units)),
                    video_duration=max(10.0, c_end - c_start),
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
                duration += c_dur
                for k in usage:
                    usage[k] += c_usage.get(k, 0)

                # 先把原始回傳落地再解析，解析失敗時仍可查核模型輸出
                (out_dir / f"{base_name}_{tag}_chunk{idx}_raw.txt").write_text(c_raw_json or "", encoding="utf-8")
                chunk_edl_results.append(_parse_gemini_json_text(c_raw_json))

            # Each chunk numbers timeline_position from 1: renumber across the project (chunk value kept)
            model_edl = _merge_chunk_results(chunk_edl_results, project_title=f"{base_name} AI Video Trimmer ({tag})")

            # Step 4a: Show Mode 全片組裝（純文字推論）：跨區段取捨、場外角色判定、段落標籤與順序。
            # 失敗時以最精簡格式重試一次，仍失敗就降級為區段結果（不中止整支片）。
            if getattr(args, "assemble", False) and whisper_units:
                model_edl, assembly_info, assembly_usage, assembly_dur = _run_show_assembly(
                    args, client, chunk_edl_results, whisper_units, prompt_candidates,
                    reference_script_path, script_path, out_dir, base_name, tag, video_path,
                )
                duration += assembly_dur
                for k in usage:
                    usage[k] += assembly_usage.get(k, 0)

            # Step 4b: Pre-Render Semantic Audit & Surgical Micro-Window Repair (15s-90s windows)
            if whisper_units:
                def _run_window_repair(win_units, win_start, win_end, start_offset, end_offset, anomaly):
                    nonlocal duration
                    w_prompt = build_micro_window_repair_prompt(
                        prompt_file_candidates=prompt_candidates,
                        whisper_units=win_units,
                        anomaly=anomaly,
                        script_path=script_path,
                        reference_script_path=reference_script_path,
                    )
                    w_raw_json, w_usage, w_dur = run_gemini_inference(
                        client,
                        types,
                        args.model,
                        gcs_uri,
                        mime_type,
                        w_prompt,
                        False,
                        num_sentences=max(1, len(win_units)),
                        video_duration=max(10.0, win_end - win_start),
                        start_offset=start_offset,
                        end_offset=end_offset,
                    )
                    duration += w_dur
                    for k in usage:
                        usage[k] += w_usage.get(k, 0)
                    return _parse_gemini_json_text(w_raw_json)

                model_edl, micro_window_repairs = repair_edl_micro_windows(
                    model_edl=model_edl,
                    whisper_units=whisper_units,
                    total_dur=total_dur,
                    run_window_inference_fn=_run_window_repair,
                    script_text=audit_script_text,
                    keep_order=allow_reorder,
                )
        finally:
            if is_ephemeral and not args.keep_gcs_upload:
                delete_gcs_blob(gcs_uri)

        logger.info("    Model inference complete (%.1f s).", duration)
        logger.info("    Token usage: prompt=%s | candidates=%s | thoughts=%s | tool_use=%s | total=%s",
                    f"{usage['prompt_tokens']:,}", f"{usage['candidates_tokens']:,}",
                    f"{usage['thoughts_tokens']:,}", f"{usage['tool_use_tokens']:,}", f"{usage['total_tokens']:,}")

        mode = "agentic" if args.agentic else "static"
        inference_metrics = {
            "mode": mode,
            "duration_seconds": round(duration, 2),
            "prompt_tokens": usage["prompt_tokens"],
            "candidates_tokens": usage["candidates_tokens"],
            "thoughts_tokens": usage["thoughts_tokens"],
            "tool_use_tokens": usage["tool_use_tokens"],
            "total_tokens": usage["total_tokens"],
            "micro_window_repairs": len(micro_window_repairs),
        }
        if assembly_info is None:
            _append_usage_log(out_dir, video_path, args.model, usage, mode, duration)
        else:
            # Assembly attempts are already logged one line each; log the rest so the lines add up to the total.
            rest = {k: usage[k] - int(assembly_usage.get(k, 0) or 0) for k in usage}
            _append_usage_log(out_dir, video_path, args.model, rest, mode, max(0.0, duration - assembly_dur),
                              extra={"scope": "chunks_and_repairs"})

    if assembly_info is not None:
        inference_metrics["assembly_status"] = assembly_info["status"]
        inference_metrics["assembly_tokens"] = assembly_info["usage"]

    # Layer 3 & Layer 4: Sub-unit word trimming, cross-clip coalescing, and acoustic onset snapping
    logger.info("==> 5. Running text-locked word boundary trimming and acoustic onset snapping...")
    temp_wav = out_dir / f"temp_{base_name}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, sr = sf.read(str(temp_wav))

    # Show Mode：雙麥克風時，主持人 clip 用主持人聲道、場外 clip 用場外聲道做聲學鎖定，
    # 否則另一支麥克風的聲音會讓入點提早到前一個人的尾音上（HEAD_BLEED）。
    role_audio = {}
    offscreen_setting = str(getattr(args, "offscreen_channel", "none") or "none").lower()
    if offscreen_setting != "none":
        try:
            from .roles import detect_offscreen_channel, extract_stereo_wav, load_stereo_audio

            temp_stereo = out_dir / f"temp_{base_name}_stereo.wav"
            extract_stereo_wav(video_path, temp_stereo)
            stereo, sr_st = load_stereo_audio(temp_stereo)
            temp_stereo.unlink(missing_ok=True)
            if stereo.shape[1] >= 2 and sr_st == sr:
                off_side = detect_offscreen_channel(stereo) if offscreen_setting == "auto" else offscreen_setting
                off_idx = 0 if off_side == "left" else 1
                role_audio = {"host": stereo[:, 1 - off_idx].copy(), "offscreen": stereo[:, off_idx].copy()}
                logger.info("    [聲道鎖定] 主持人 clip 用 %s 聲道、場外 clip 用 %s 聲道做聲學邊界",
                            "右" if off_idx == 0 else "左", "左" if off_idx == 0 else "右")
        except Exception as exc:  # noqa: BLE001
            logger.warning("    [聲道鎖定略過] %s", exc)

    if whisper_units and isinstance(model_edl, dict) and model_edl.get("final_edl"):
        model_edl["final_edl"] = deduplicate_and_sort_clips(
            clips=model_edl["final_edl"],
            whisper_units=whisper_units,
            script_text=audit_script_text,
            keep_order=allow_reorder,
        )

    # Show Mode 節奏參數：想法內部的停頓可以保留（keep_pause），想法結束後多留一點尾音（tail_pad）
    # Without --keep-pause-sec, keep the conform-tuned defaults in transcribe.py
    # (resolve_clip_sub_units: max_internal_gap=0.65 / max_word_gap=0.70; coalesce: 0.65).
    keep_pause_arg = getattr(args, "keep_pause_sec", None)
    keep_pause = float(keep_pause_arg) if keep_pause_arg is not None else None
    tail_pad = float(getattr(args, "tail_pad_sec", 0.0) or 0.0)
    resolve_gap_kwargs = (
        {"max_internal_gap": keep_pause, "max_word_gap": max(0.45, keep_pause)} if keep_pause is not None else {}
    )
    coalesce_gap_kwargs = {"max_internal_gap": keep_pause} if keep_pause is not None else {}

    expanded_units = []
    for c in model_edl.get("final_edl", []):
        sub_units = resolve_clip_sub_units(whisper_units, c, total_dur, **resolve_gap_kwargs)
        base_topic = str(c.get("topic", ""))
        if str(c.get("role", "host")).lower() == "offscreen":
            base_topic = f"[場外] {base_topic}"
        for idx, su in enumerate(sub_units):
            topic_label = base_topic if len(sub_units) == 1 else f"{base_topic} (#{idx + 1})"
            expanded_units.append({
                "parent_clip": c,
                "topic": topic_label,
                "sentence_ids": su.get("sentence_ids", []),
                "t_first": su["t_first"],
                "t_last": su["t_last"],
                "prev_sentence_end": su["prev_sentence_end"],
                "next_sentence_start": su["next_sentence_start"],
                "transcript": su["transcript"],
                "whisper_transcript": su.get("whisper_transcript", su["transcript"]),
                "trimmed_head": su.get("trimmed_head", False),
                "trimmed_tail": su.get("trimmed_tail", False),
                "role": str(c.get("role", "host")).lower(),
            })

    # Coalesce adjacent continuous sub-units across clip boundaries when gap < keep_pause and no ID is skipped
    filtered_units = coalesce_adjacent_sub_units(expanded_units, **coalesce_gap_kwargs)

    # Trim tail-to-head text overlaps and head stutters across adjacent EDL cut seams
    filtered_units = trim_cross_clip_seam_overlaps(filtered_units, whisper_units)

    # Drop isolated conjunction / particle fragments (Orphan Conjunction / Particle Filter)
    filtered_units = _prune_orphan_units(filtered_units)

    refined_edl = []
    for seq_id, su in enumerate(filtered_units, start=1):
        c = su["parent_clip"]
        t_first = su["t_first"]
        t_last = su["t_last"]
        transcript = su["transcript"]
        prev_sentence_end = su["prev_sentence_end"]
        next_sentence_start = su["next_sentence_start"]

        clip_role = "offscreen" if str(c.get("role", "host")).lower() == "offscreen" else "host"
        lock_audio = role_audio.get(clip_role, audio) if role_audio else audio
        # 出點上限＝下一段聲音的聲學起音（含另一支麥克風的下一句），不是 Whisper 偏早的下一句起點；
        # tail_pad（讓想法落地）也套同一個上限
        other_audio = [a for r, a in role_audio.items() if r != clip_role] if role_audio else None
        tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds_locked(
            lock_audio, sr, t_first, t_last, total_dur, transcript=transcript, pacing=args.pacing,
            prev_sentence_end=prev_sentence_end, next_sentence_start=next_sentence_start,
            tail_pad=tail_pad, other_audio=other_audio,
        )
        if tight_out <= tight_in:
            tight_out = tight_in + 1.0
        dur = round(tight_out - tight_in, 2)

        refined_edl.append({
            "clip_id": seq_id,
            "topic": su["topic"],
            "sentence_ids": su.get("sentence_ids", []),
            "t_first": round(t_first, 2),
            "t_last": round(t_last, 2),
            "source_in": tight_in,
            "source_out": tight_out,
            "duration": dur,
            "cps": clip_cps,
            "in_margin": in_m,
            "out_margin": out_m,
            "transcript": transcript,
            "whisper_transcript": su.get("whisper_transcript", transcript),
            "role": str(c.get("role", "host")).lower(),
            "segment": c.get("segment", ""),
            "timeline_position": c.get("timeline_position"),  # renumbered 1..N after sanitization
            "chunk_index": c.get("chunk_index"),
            "chunk_timeline_position": c.get("chunk_timeline_position"),
            "take_selection_reason": c.get("take_selection_reason", ""),
            "visual_check": c.get("visual_check", "Direct eye contact with camera, eyes open."),
            "audio_check": c.get("audio_check", f"Whisper locked (in_margin={in_m:.2f}s, out_margin={out_m:.2f}s)")
        })
        logger.info("    Clip %2d: In=%6.2fs, Out=%6.2fs (%5.2fs) | t_first: %.2fs, t_last: %.2fs | %s",
                    seq_id, tight_in, tight_out, dur, t_first, t_last, su["topic"])

    temp_wav.unlink(missing_ok=True)

    # Layer 5: Deterministic timeline sanitization and 8-dimension rough-cut quality audit
    refined_edl, sanitization_stats = sanitize_refined_edl(refined_edl, total_dur, allow_reorder=allow_reorder)
    # The list order is the final timeline. One parent clip can expand into several sub-clips and chunk
    # models number from 1, so timeline_position is renumbered here to stay unique (1..N) for consumers
    # that sort by it. The chunk-local value stays in chunk_timeline_position.
    for pos, c in enumerate(refined_edl, start=1):
        c["timeline_position"] = pos
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    avg_cps = round(sum(c["cps"] for c in refined_edl) / max(1, len(refined_edl)), 2)
    logger.info("    Retained clips: %d | Trimmed duration: %.1f s (~%.2f min) | Average CPS: %.2f chars/s",
                len(refined_edl), total_out_dur, total_out_dur / 60, avg_cps)

    # Flags carried into the audit remediation_cmd (Show Mode, render and XML profile options)
    show_mode_flags = []
    if getattr(args, "prompt", None):
        show_mode_flags.append(f'--prompt "{args.prompt}"')
    if reference_script_path is not None:
        show_mode_flags.append(f'--reference-script "{reference_script_path}"')
    if getattr(args, "offscreen_channel", None) and str(args.offscreen_channel).lower() != "none":
        show_mode_flags.append(f"--offscreen-channel {args.offscreen_channel}")
    if getattr(args, "analysis_video", None):
        show_mode_flags.append(f'--analysis-video "{args.analysis_video}"')
    if getattr(args, "assemble", False):
        show_mode_flags.append("--assemble")
    if allow_reorder:
        show_mode_flags.append("--allow-reorder")
    if getattr(args, "xml_only", False):
        show_mode_flags.append("--xml-only")
    if getattr(args, "repeats_as_warnings", False):
        show_mode_flags.append("--repeats-as-warnings")
    if getattr(args, "no_render", False):
        show_mode_flags.append("--no-render")
    xml_profile = str(getattr(args, "xml_profile", "premiere") or "premiere").lower()
    if xml_profile != "premiere":
        show_mode_flags.append(f"--xml-profile {xml_profile}")

    audit_report = audit_edl_quality(
        refined_edl=refined_edl,
        whisper_units=whisper_units,
        total_dur=total_dur,
        video_path=video_path,
        script_text=audit_script_text,
        script_path=script_path,
        sanitization_stats=sanitization_stats,
        micro_window_repairs=micro_window_repairs,
        allow_reorder=allow_reorder,
        mode_label=("Mode C (Script Reference, Show Mode)" if reference_script_text is not None else None),
        extra_cli_flags=" ".join(show_mode_flags),
        repeats_as_warnings=bool(getattr(args, "repeats_as_warnings", False)),
    )
    if assembly_info is not None:
        audit_report["assembly"] = {k: v for k, v in assembly_info.items() if k != "attempts"}
        audit_report["assembly"]["attempts"] = [
            {k: a.get(k) for k in ("attempt", "format", "status", "finish_reason", "usage", "error")}
            for a in assembly_info.get("attempts", [])
        ]
        if assembly_info.get("status") == STATUS_DEGRADED:
            audit_report.setdefault("agent_verdict", {}).setdefault("warnings", []).insert(
                0, _assembly_report_line(assembly_info))
    report_json_path = out_dir / f"{base_name}_{tag}_edl_report.json"
    report_md_path = out_dir / f"{base_name}_{tag}_edl_report.md"
    report_json_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_edl_audit_markdown(audit_report, report_md_path)
    if assembly_info is not None:
        _prepend_assembly_banner(report_md_path, assembly_info)
    verdict = audit_report["agent_verdict"]
    logger.info(
        "    [Quality Audit] Grade: %s | Pass Quality Gate: %s | Report: %s",
        verdict["overall_grade"],
        verdict["pass_quality_gate"],
        report_md_path.name,
    )

    json_path = out_dir / f"{base_name}_{tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI Video Trimmer ({tag})",
        "pacing_style": "text_based_whisper_grounded",
        "agent_verdict": verdict,
        "inference_metrics": inference_metrics,
        **({"assembly": audit_report["assembly"]} if assembly_info is not None else {}),
        "average_cps": avg_cps,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("==> 6. Wrote structured EDL JSON and audit report: %s, %s", json_path.name, report_json_path.name)

    csv_path = out_dir / f"{base_name}_{tag}_edl.csv"
    generate_edl_csv(refined_edl, csv_path)
    logger.info("==> 7. Wrote CSV cut table: %s", csv_path.name)

    # FCP 7 XML: premiere (default) = Premiere Pro structure; davinci = same structure with the camera
    # timecode passed through in <file><timecode> and no <frame> tag
    cam_tc = None
    if xml_profile == "davinci":
        cam_tc = get_video_timecode(video_path)
        if cam_tc != "00:00:00:00":
            logger.info("    [Camera Timecode] Embedded camera timecode %s written to the XML to prevent DaVinci Resolve media offline.", cam_tc)

    xml_path = out_dir / f"{base_name}_{tag}_edl.xml"
    generate_fcp7_xml(refined_edl, video_path, total_dur, xml_path, width, height, fps,
                      start_tc=cam_tc, xml_profile=xml_profile)
    logger.info("==> 8. Wrote FCP 7 XML timeline (profile=%s): %s", xml_profile, xml_path.name)

    fcpxml_path = out_dir / f"{base_name}_{tag}_edl.fcpxml"
    generate_fcpxml(refined_edl, video_path, total_dur, total_out_dur, fcpxml_path, fps, width, height)
    logger.info("==> 9. Wrote Final Cut Pro FCPXML timeline: %s", fcpxml_path.name)

    # Conform clip summary table. `_edl_report.md` is the 8-dimension audit report (agent_verdict),
    # so the summary goes to `_edl_summary.md` and never overwrites it.
    summary_path = out_dir / f"{base_name}_{tag}_edl_summary.md"
    generate_edl_report(refined_edl, total_dur, summary_path, base_name=f"{base_name} ({tag})")
    if assembly_info is not None:
        _prepend_assembly_banner(summary_path, assembly_info)
    logger.info("==> 10. Wrote Markdown clip summary: %s", summary_path.name)

    if getattr(args, "xml_only", False) or getattr(args, "no_render", False):
        logger.info("==> 11. --xml-only / --no-render set: skipping the FFmpeg MP4 render.")
        logger.info("==> [Done] NLE timelines written: %s", xml_path.name)
    else:
        out_mp4 = out_dir / f"{base_name}_{tag}_trimmed.mp4"
        logger.info("==> 11. Rendering rough-cut MP4 with FFmpeg: %s ...", out_mp4.name)
        render_cut_video(refined_edl, video_path, out_mp4, crf=args.crf)
        logger.info("==> [Done] Final MP4 rendered (%.1f MB): %s", out_mp4.stat().st_size / (1024 * 1024), out_mp4.name)

    if args.strict and not verdict["pass_quality_gate"]:
        logger.error("==> [Strict Quality Gate Failed] %s", "; ".join(verdict["fatal_violations"]))
        sys.exit(2)


def main():
    load_env_file()
    default_model = os.environ.get("MODEL_NAME") or os.environ.get("TRIMMER_MODEL") or "gemini-3.8-flash"

    parser = argparse.ArgumentParser(description="video-trimmer: AI-Powered Smart Video Trimmer (Whisper Ground-Truth + Vertex AI Gemini)")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file (.mp4/.mov) or Google Drive share link (https://drive.google.com/... / gdrive://...)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (defaults to <input_dir>/output/, or ./output/ for Google Drive links)")
    parser.add_argument("--model", "-m", default=default_model, help=f"Gemini model name (default: {default_model})")
    parser.add_argument("--project", default=None, help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT/GCP_PROJECT or ADC)")
    parser.add_argument("--region", default=None, help="Vertex AI region/location (defaults to GOOGLE_CLOUD_LOCATION/GCP_REGION or 'global')")
    parser.add_argument("--bucket", default=None, help="GCS bucket for video staging (defaults to VIDEO_TRIMMER_BUCKET)")
    parser.add_argument("--keep-gcs-upload", action="store_true", help="Retain staged video in GCS instead of deleting after inference")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 CRF quality parameter (default: 18)")
    parser.add_argument("--no-render", action="store_true", help="Skip the FFmpeg MP4 render; write timelines and tables only (same as --xml-only)")
    parser.add_argument("--xml-profile", choices=list(XML_PROFILES), default="premiere",
                        help="FCP 7 XML profile: 'premiere' (default, Premiere Pro structure) or 'davinci' "
                             "(same structure, xmeml v5, camera timecode in <file><timecode> without <frame>, for DaVinci Resolve)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="Pacing style: 'auto'/'dynamic' (adaptive CPS margin, default), 'compact', or 'breathing'")
    parser.add_argument("--agentic", action="store_true", help="Enable Gemini Agentic Video Understanding mode")
    parser.add_argument("--suffix", default=None, help="Custom filename tag suffix (defaults to 'agentic' or 'static')")
    parser.add_argument("--script", "-s", default=None, help="Optional reference shooting script or outline (.txt/.md)")
    parser.add_argument("--cached-json", default=None,
                        help="Path to an existing EDL JSON file to bypass cloud Gemini inference")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip local Whisper transcription and use energy-only fallback")
    parser.add_argument("--whisper-model", default=None,
                        help="Whisper model: small / medium / large-v3 / large-v3-turbo, or an mlx Hugging Face repo "
                             "(defaults to $WHISPER_MODEL or large-v3; mlx-whisper loads mlx-community/whisper-<model>-mlx)")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 after writing audit reports if agent_verdict.pass_quality_gate is False")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DEBUG-level logging")
    # Show Mode（節目型拍帶：主持人 + 場外聲音角色）。以下旗標預設全部關閉，不影響原有行為。
    show = parser.add_argument_group("Show Mode（主持人 + 場外聲音角色）")
    show.add_argument("--prompt", default=None,
                      help="指定提示詞檔案（例如 prompts/video_cut_prompt_show.md），預設使用 prompts/video_cut_prompt.md")
    show.add_argument("--reference-script", default=None,
                      help="Mode C：腳本只作結構參考（不做 Mode A 覆蓋率檢查與單向消耗），TXT/MD")
    show.add_argument("--offscreen-channel", choices=["none", "auto", "left", "right"], default="none",
                      help="雙麥克風拍攝時場外麥克風所在聲道；auto 取整體較小聲的聲道。會把每句標成 HOST/OFFSCREEN/CUE 並在角色切換處拆句")
    show.add_argument("--analysis-video", default=None,
                      help="上傳與 Gemini 分析改用此代理檔（例如 1080p），Whisper 時間戳、聲學鎖定與 XML 仍以 -i 原檔為準")
    show.add_argument("--assemble", action="store_true",
                      help="分段推論後再做一次純文字全片組裝：跨區段取捨、場外角色判定、段落標籤與順序。"
                           "失敗時以最精簡格式重試一次，仍失敗就降級為各區段結果（報告會標明）")
    show.add_argument("--reuse-chunks", default=None,
                      help="重用既有區段結果（*_chunks_raw.json、*_chunk1_raw.txt 或只含一組區段結果的資料夾），"
                           "不上傳影片、不重跑區段推論；配合 --assemble 只重跑全片組裝。沒有影片可重掃，略過 Micro-Window Repair。"
                           "區段結果旁若有 <stem>_whisper_sentences.json，句子編號不一致時會中止")
    show.add_argument("--allow-reorder", action="store_true",
                      help="允許成片順序與原片時間順序不同（依 timeline_position），順序顛倒改為警告而非 fatal")
    show.add_argument("--xml-only", action="store_true", help="只輸出 JSON/CSV/XML/FCPXML 與審計報告，不渲染 MP4")
    show.add_argument("--repeats-as-warnings", action="store_true",
                      help="句內重複字句改列為警告而非 fatal（節目型拍帶常有刻意排比與重複，模型已在局部重掃確認）")
    show.add_argument("--keep-pause-sec", type=float, default=None,
                      help="想法內部可保留的停頓上限（秒）；小於此值的停頓不切開、連續句子合併成一個 clip。"
                           "未指定時沿用預設調校（句間 0.65 / 字間 0.70），節目型建議 1.0")
    show.add_argument("--tail-pad-sec", type=float, default=0.0,
                      help="每個 clip 尾字之後多留的時間（秒），讓想法落地；不會跨進下一句開口。預設 0，節目型建議 0.25")
    show.add_argument("--transcribe-only", action="store_true",
                      help="只做 Whisper 轉錄與句子合併（含角色標籤）後結束，不上傳也不呼叫 Gemini")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    try:
        _run(args)
    except VideoTrimmerError as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

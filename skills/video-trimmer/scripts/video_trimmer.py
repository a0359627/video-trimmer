#!/usr/bin/env python3
"""
video-trimmer (AI-Powered Smart Video Trimmer)
---------------------------------------------
An intelligent, end-to-end video rough-cut & trimming engine powered by:
- Whisper word-level acoustic ground-truth (Apple Silicon mlx-whisper & faster-whisper)
- Gemini 3.8 Flash multimodal video understanding & take selection
- Acoustic Onset Snapping (Smart Gap Shortening: eliminating dead air & pre-speech pauses)
- Adaptive noise floor tracking & equal-power audio micro-crossfades
- Text-based locked timestamp precision
- Export to Premiere Pro (XML), DaVinci Resolve (XML), Final Cut Pro (FCPXML), and direct MP4.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np  # noqa: F401  (used transitively by acoustic.py; kept for early dependency check)
    from google.genai import types
except ImportError as e:
    # 此處早於 main() 的 logging 設定即需中止，且是模組載入期的硬性依賴檢查
    # （而非業務邏輯例外），故沿用 sys.exit()；仍改用 logging 而非 print()。
    logging.basicConfig(level=logging.ERROR, format="%(message)s", stream=sys.stderr)
    logging.error("缺少必要套件: %s", e)
    logging.error("請先執行: pip install -r requirements.txt")
    sys.exit(1)

import math
import os

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
from .exporters import generate_edl_csv, generate_fcp7_xml, generate_fcpxml
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
    build_assembly_prompt,
    normalize_assembled_edl,
    parse_assembly_json,
    run_assembly_inference,
)
from .transcribe import (
    align_clip_with_whisper,
    calculate_active_script_window,
    coalesce_adjacent_sub_units,
    detect_chunk_boundaries,
    resolve_clip_sub_units,
    transcribe_video_whisper,
)

logger = logging.getLogger(__name__)


def _parse_gemini_json_text(raw_json: str) -> dict:
    """Clean markdown fences and parse JSON returned by Gemini (object, bare array or single clip)."""
    try:
        return parse_edl_json_text(raw_json)
    except GeminiAPIError as e:
        raise VideoTrimmerError(str(e)) from e


def _setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def _validate_input_extension(video_path):
    if video_path.suffix.lower() not in ALLOWED_INPUT_EXTENSIONS:
        allowed = "、".join(ALLOWED_INPUT_EXTENSIONS)
        raise InvalidInputError(f"不支援的影片格式: {video_path.suffix}（允許的副檔名: {allowed}）")


def _append_usage_log(out_dir, video_path, model, usage, mode, duration):
    """獨立於 logging 系統之外，永遠記錄每次 Gemini 呼叫的 token 用量，供後續人工彙總成本。"""
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
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
            raise InvalidInputError(f"從 Google Drive 讀取影片失敗: {e}") from e
    else:
        video_path = Path(args.input).expanduser().resolve()
        if not video_path.exists():
            raise InvalidInputError(f"找不到影片檔案 {video_path}")
        out_dir = _resolve_output_dir(args.output_dir, video_path.parent)

    _validate_input_extension(video_path)
    base_name = video_path.stem

    # 決定輸出檔案標籤
    tag = args.suffix if args.suffix else ("agentic" if args.agentic else "static")

    logger.info("==> 1. 檢測影片資訊: %s", video_path.name)
    logger.info("    輸出目錄: %s", out_dir)
    total_dur, width, height, fps = probe_video(video_path)
    logger.info("    時長: %.1f 秒 (~%.1f 分鐘) | 解析度: %dx%d | 幀率: %.3f fps",
                total_dur, total_dur / 60, width, height, fps)

    # 第一階段：Whisper 微觀聲學時間戳轉錄 (Word-level Ground Truth) 與語意句子合併
    whisper_segs = []
    whisper_sentences = []

    if not args.skip_whisper:
        whisper_json = out_dir / f"{base_name}_whisper_raw.json"
        logger.info("==> 2. 執行 Whisper 微觀字級聲學時間戳轉錄與語意句子合併...")
        whisper_segs, whisper_sentences = transcribe_video_whisper(
            video_path,
            whisper_json,
            model_name="small",
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

    # 第二階段：Gemini 宏觀多模態視訊理解與選鏡決策
    inference_metrics = {}
    micro_window_repairs = []
    script_path = Path(args.script).resolve() if args.script else None
    script_text = script_path.read_text(encoding="utf-8") if (script_path is not None and script_path.exists()) else None
    # Mode C：腳本只作結構參考。Mode A 的覆蓋率檢查、碰撞修復、腳本去重與「講稿視窗鎖定」一律關閉，
    # 因為脫稿的開場或結尾不在腳本裡，鎖定視窗會把它們切掉。
    audit_script_text = None if reference_script_text is not None else script_text
    window_script_text = script_text

    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            raise InvalidInputError(f"找不到快取 JSON 檔案: {cached_file}")
        logger.info("==> 3. 跳過 Gemini 模型分析，直接載入快取初剪決策: %s", cached_file.name)
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    else:
        load_env_file()
        client = get_gemini_client(project_id=args.project, location=args.region)

        _resolved_file = Path(__file__).resolve()
        prompt_candidates = [
            *([Path(args.prompt).expanduser().resolve()] if getattr(args, "prompt", None) else []),
            Path(__file__).parent / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent.parent / "prompts" / "video_cut_prompt.md",
            Path.cwd() / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent / "video-trimmer" / "prompts" / "video_cut_prompt.md",
            *[p / "prompts" / "video_cut_prompt.md" for p in _resolved_file.parents[:5]],
        ]

        # Step 3a: 若提供講稿且影片為長素材，先依據講稿語意句子定位有效視訊視窗
        win_start, win_end, active_units = 0.0, total_dur, whisper_units
        if window_script_text is not None and total_dur > 600.0 and whisper_units:
            win_start, win_end, active_units = calculate_active_script_window(
                whisper_units=whisper_units,
                script_text=window_script_text,
                total_duration=total_dur,
                padding_seconds=20.0,
            )
            if win_start > 0.0 or win_end < total_dur:
                logger.info(
                    "    [講稿視窗鎖定] 自動聚焦有效拍攝區間: %.1fs ~ %.1fs (跨度 %.1fs, 台詞 %d 句)",
                    win_start,
                    win_end,
                    win_end - win_start,
                    len(active_units),
                )

        # Step 3b: 若有效區間超過 11 分鐘 (660s)，於自然停頓處 (>=2.0s) 自動切塊並避免切開重講群組
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

        logger.info("==> 3. 準備視訊至 Cloud Storage 暫存...")
        gcs_uri, mime_type, is_ephemeral = stage_video_to_gcs(analysis_source, resolved_bucket)

        mode_str = "Agentic Video Understanding" if args.agentic else "Static Multimodal"
        logger.info("==> 4. 調用 %s [模式: %s] 結合 Whisper 劇本進行文字剪輯與選鏡決策...", args.model, mode_str)

        usage = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "tool_use_tokens": 0, "total_tokens": 0}
        duration = 0.0
        chunk_edl_results = []

        try:
            if len(chunks) > 1:
                logger.info("    [長片自動分段] 有效跨度 %.1fs 分割為 %d 個自然停頓區段依序推論...", win_end - win_start, len(chunks))

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
                        "    ---> Chunk %d/%d: 視訊區間 %s ~ %s (%.1fs, %d 句台詞)",
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

            if len(chunk_edl_results) == 1:
                model_edl = chunk_edl_results[0]
            else:
                model_edl = merge_chunked_edl(
                    chunk_edl_results,
                    project_title=f"{base_name} AI Video Trimmer ({tag})",
                )

            # Step 4a: Show Mode 全片組裝（純文字推論）：跨區段取捨、場外角色判定、段落標籤與順序
            if getattr(args, "assemble", False) and whisper_units:
                a_prompt = build_assembly_prompt(
                    prompt_candidates,
                    chunk_edl_results,
                    whisper_units,
                    reference_script_path=reference_script_path,
                    script_path=script_path,
                )
                n_candidates = sum(len(ch.get("final_edl", [])) for ch in chunk_edl_results if isinstance(ch, dict))
                a_raw_json, a_usage, a_dur = run_assembly_inference(
                    client, types, args.model, a_prompt, n_candidates, len(whisper_units)
                )
                duration += a_dur
                for k in usage:
                    usage[k] += a_usage.get(k, 0)
                (out_dir / f"{base_name}_{tag}_assembly_raw.json").write_text(a_raw_json, encoding="utf-8")
                (out_dir / f"{base_name}_{tag}_chunks_raw.json").write_text(
                    json.dumps(chunk_edl_results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                model_edl = normalize_assembled_edl(parse_assembly_json(a_raw_json), whisper_units, chunk_edl_results)
                logger.info("    [全片組裝] 候選 %d -> 最終 %d clips（%.1f 秒）",
                            n_candidates, len(model_edl.get("final_edl", [])), a_dur)

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

        logger.info("    模型推論完成 (耗時 %.1f 秒)！", duration)
        logger.info("    Token 統計: 輸入=%s | 輸出=%s | 思維=%s | 工具探索=%s | 總計=%s",
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
        _append_usage_log(out_dir, video_path, args.model, usage, mode, duration)

    # 第三階段：文字剪輯時間鎖定與聲學收緊 (Text-Based Sub-Clip Locking)
    logger.info("==> 5. 執行文字剪輯時間鎖定與聲學收緊 (Text-Based Sub-Clip Locking)...")
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
    keep_pause = float(getattr(args, "keep_pause_sec", 0.40) or 0.40)
    tail_pad = float(getattr(args, "tail_pad_sec", 0.0) or 0.0)

    expanded_units = []
    for c in model_edl.get("final_edl", []):
        sub_units = resolve_clip_sub_units(
            whisper_units, c, total_dur,
            max_internal_gap=keep_pause, max_word_gap=max(0.45, keep_pause),
        )
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
            })

    # Coalesce adjacent continuous sub-units across clip boundaries when gap < keep_pause and no ID is skipped
    filtered_units = coalesce_adjacent_sub_units(expanded_units, max_internal_gap=keep_pause)

    # 5c. 針對每個保留子片段獨立執行聲學包絡收緊與微呼吸邊界鎖定
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
        tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds_locked(
            lock_audio, sr, t_first, t_last, total_dur, transcript=transcript, pacing=args.pacing,
            prev_sentence_end=prev_sentence_end, next_sentence_start=next_sentence_start
        )
        if tight_out <= tight_in:
            tight_out = tight_in + 1.0
        if tail_pad > 0.0:
            # 讓想法落地：想法結束後多留一點尾音與停頓，但不跨進下一句的開口
            limit = total_dur
            if next_sentence_start is not None:
                limit = min(limit, float(next_sentence_start) - 0.02)
            tight_out = round(max(tight_out, min(limit, tight_out + tail_pad)), 2)
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
            "timeline_position": c.get("timeline_position"),
            "take_selection_reason": c.get("take_selection_reason", ""),
            "visual_check": c.get("visual_check", "眼神直視鏡頭就緒，無眨眼閉眼"),
            "audio_check": c.get("audio_check", f"Whisper精準錨定 (In前置氣息={in_m:.2f}s, Out俐落收口={out_m:.2f}s)")
        })
        logger.info("    Clip %2d: In=%6.2fs, Out=%6.2fs (%5.2fs) | 首字: %.2fs, 尾字: %.2fs | %s",
                    seq_id, tight_in, tight_out, dur, t_first, t_last, su["topic"])

    temp_wav.unlink(missing_ok=True)

    # 5d. 物理時間軸確定性自癒 (Overlap De-confliction, Plosive Tail Floor, Micro-Clip Coalescing)
    refined_edl, sanitization_stats = sanitize_refined_edl(refined_edl, total_dur, allow_reorder=allow_reorder)
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    avg_cps = round(sum(c["cps"] for c in refined_edl) / max(1, len(refined_edl)), 2)
    logger.info("    初剪片段數: %d | 成片預計長度: %.1f 秒 (~%.2f 分鐘) | 全片平均語速: %.2f 字/秒",
                len(refined_edl), total_out_dur, total_out_dur / 60, avg_cps)

    # 5e. 執行 8 維度粗剪品質審計 (8-Dimension Rough-Cut Quality Audit & agent_verdict)
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
    report_json_path = out_dir / f"{base_name}_{tag}_edl_report.json"
    report_md_path = out_dir / f"{base_name}_{tag}_edl_report.md"
    report_json_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_edl_audit_markdown(audit_report, report_md_path)
    verdict = audit_report["agent_verdict"]
    logger.info(
        "    [品質審計] 等級: %s | 通過品質閘門: %s | 報告: %s",
        verdict["overall_grade"],
        verdict["pass_quality_gate"],
        report_md_path.name,
    )

    # 輸出資料
    json_path = out_dir / f"{base_name}_{tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI Video Trimmer ({tag})",
        "pacing_style": "text_based_whisper_grounded",
        "agent_verdict": verdict,
        "inference_metrics": inference_metrics,
        "average_cps": avg_cps,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("==> 6. 輸出結構化資料與審計報告: %s, %s", json_path.name, report_json_path.name)

    csv_path = out_dir / f"{base_name}_{tag}_edl.csv"
    generate_edl_csv(refined_edl, csv_path)
    logger.info("==> 7. 輸出表格清單: %s", csv_path.name)

    xml_path = out_dir / f"{base_name}_{tag}_edl.xml"
    generate_fcp7_xml(refined_edl, video_path, total_dur, xml_path, width, height, fps)
    logger.info("==> 8. 輸出通用剪輯工程檔: %s", xml_path.name)

    fcpxml_path = out_dir / f"{base_name}_{tag}_edl.fcpxml"
    generate_fcpxml(refined_edl, video_path, total_dur, total_out_dur, fcpxml_path, fps)
    logger.info("==> 9. 輸出 Final Cut Pro X 工程檔: %s", fcpxml_path.name)

    if getattr(args, "xml_only", False):
        logger.info("==> 10. 已指定 --xml-only，略過 FFmpeg 渲染成片。")
        logger.info("==> [完成] 剪輯工程檔已產出: %s", xml_path.name)
    else:
        out_mp4 = out_dir / f"{base_name}_{tag}_trimmed.mp4"
        logger.info("==> 10. FFmpeg 渲染成片: %s ...", out_mp4.name)
        render_cut_video(refined_edl, video_path, out_mp4, crf=args.crf)
        logger.info("==> [完成] 最終成片已產出！檔案大小: %.1f MB", out_mp4.stat().st_size / (1024 * 1024))

    if args.strict and not verdict["pass_quality_gate"]:
        logger.error("==> [嚴格品質閘門未通過] %s", "; ".join(verdict["fatal_violations"]))
        sys.exit(2)


def main():
    load_env_file()
    default_model = os.environ.get("MODEL_NAME") or os.environ.get("TRIMMER_MODEL") or "gemini-3.8-flash"

    parser = argparse.ArgumentParser(description="video-trimmer: AI-Powered Smart Video Trimmer (Whisper Ground-Truth + Vertex AI Gemini)")
    parser.add_argument("--input", "-i", required=True, help="輸入影片檔案路徑 (MP4/MOV) 或 Google Drive 分享連結 (https://drive.google.com/... / gdrive://...)")
    parser.add_argument("--output-dir", "-o", default=None, help="輸出資料夾 (預設為 <影片所在目錄>/output/，Google Drive 連結則為 ./output/)")
    parser.add_argument("--model", "-m", default=default_model, help=f"使用的 Gemini 模型名稱 (預設: {default_model})")
    parser.add_argument("--project", default=None, help="Google Cloud 專案 ID (預設讀取 GOOGLE_CLOUD_PROJECT/GCP_PROJECT 或 ADC)")
    parser.add_argument("--region", default=None, help="Vertex AI 區域/位置 (預設讀取 GOOGLE_CLOUD_LOCATION/GCP_REGION 或 'global')")
    parser.add_argument("--bucket", default=None, help="用於暫存視訊的 GCS Bucket (預設讀取 VIDEO_TRIMMER_BUCKET)")
    parser.add_argument("--keep-gcs-upload", action="store_true", help="保留上傳至 GCS 的暫存視訊，不於推論後自動清理")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 畫質參數 (預設 18)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="剪輯節奏風格: 'auto'/'dynamic' (文字剪輯微氣息鎖定, 推薦預設)")
    parser.add_argument("--agentic", action="store_true", help="啟用 Gemini Agentic Video Understanding 動態探索模式")
    parser.add_argument("--suffix", default=None, help="自訂輸出檔案名稱標籤後綴 (預設為 agentic 或 static)")
    parser.add_argument("--script", "-s", default=None, help="可選的分鏡講稿或文本檔案路徑 (TXT/MD)")
    parser.add_argument("--cached-json", default=None,
                        help="指定既有之初剪決策 JSON 檔案路徑，跳過 Gemini 上傳與雲端分析")
    parser.add_argument("--skip-whisper", action="store_true", help="跳過本地 Whisper 轉錄，僅使用純能量檢測")
    parser.add_argument("--strict", action="store_true", help="啟用嚴格品質閘門 (若 agent_verdict.pass_quality_gate 為 false 則於輸出報告後以 exit code 2 結束)")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細除錯訊息 (DEBUG level)")
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
                      help="分段推論後再做一次純文字全片組裝：跨區段取捨、場外角色判定、段落標籤與順序")
    show.add_argument("--allow-reorder", action="store_true",
                      help="允許成片順序與原片時間順序不同（依 timeline_position），順序顛倒改為警告而非 fatal")
    show.add_argument("--xml-only", action="store_true", help="只輸出 JSON/CSV/XML/FCPXML 與審計報告，不渲染 MP4")
    show.add_argument("--repeats-as-warnings", action="store_true",
                      help="句內重複字句改列為警告而非 fatal（節目型拍帶常有刻意排比與重複，模型已在局部重掃確認）")
    show.add_argument("--keep-pause-sec", type=float, default=0.40,
                      help="想法內部可保留的停頓上限（秒）；小於此值的停頓不切開、連續句子合併成一個 clip。預設 0.40，節目型建議 1.0")
    show.add_argument("--tail-pad-sec", type=float, default=0.0,
                      help="每個 clip 尾字之後多留的時間（秒），讓想法落地；不會跨進下一句開口。預設 0，節目型建議 0.25")
    show.add_argument("--transcribe-only", action="store_true",
                      help="只做 Whisper 轉錄與句子合併（含角色標籤）後結束，不上傳也不呼叫 Gemini")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    try:
        _run(args)
    except VideoTrimmerError as e:
        logger.error("錯誤: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()


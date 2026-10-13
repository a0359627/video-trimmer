"""Cut review: render the assembled cut's audio, re-transcribe it, find weak seams, write markers.

ASD-STE100:
A rough cut can be verified like an editor does: listen to the cut from start to end.
This module renders the EDL's audio (both channels, frame-exact to the XML), runs Whisper on
the rendered audio, maps every cut boundary back to the source, and flags seams that are likely
to sound wrong:

- CUT_MID_SENTENCE: the kept text does not close a sentence and the next source sentence (which
  was dropped) continues the same thought within a short pause.
- TAIL_TRIMMED: the engine trimmed trailing words because the model's transcript was shorter
  than what was actually said.
- HEAD_TRIMMED: leading words were trimmed the same way.
- HAS_ALTERNATIVE: a dropped sentence nearby says the same thing (another take exists).
- MANUAL: the clip was restored or edited by hand after the model pass.
- LOW_CONFIDENCE: the model's own reason mentions uncertainty, or the clip is an off-screen line.

The result is a Markdown list with timeline and source timecodes, a JSON file, and a list of
sequence markers that `exporters.generate_fcp7_xml(markers=...)` writes into the XML so the
editor can jump to each seam inside Premiere Pro.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

CLOSURE_PUNCT = ("。", "！", "？", "!", "?", "…", "...", "～", "~")
UNCERTAIN_WORDS = ("可能", "或許", "不確定", "推測", "疑似", "未經視覺確認", "人工補回", "請確認", "需確認", "猜測")


def frames(sec: float, fps: float) -> int:
    return int(round(sec * fps))


def tc(sec: float, fps: float) -> str:
    """Seconds -> HH:MM:SS:FF timecode (non-drop) for the editor."""
    total = frames(sec, fps)
    tb = int(round(fps))
    f = total % tb
    s = total // tb
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}:{f:02d}"


def timeline_layout(edl: list[dict], fps: float) -> list[dict]:
    """Frame-exact timeline placement identical to the XML exporter."""
    out = []
    cursor = 0
    for c in edl:
        in_f = frames(float(c["source_in"]), fps)
        out_f = frames(float(c["source_out"]), fps)
        dur = max(1, out_f - in_f)
        out.append({"clip_id": c["clip_id"], "in_f": in_f, "out_f": in_f + dur, "start_f": cursor, "end_f": cursor + dur})
        cursor += dur
    return out


def render_cut_audio(edl: list[dict], video_path: Path, out_wav: Path, out_mp3: Path | None, fps: float) -> Path:
    """Concatenate the EDL's audio segments (all channels, frame-exact) into a WAV and optional MP3."""
    layout = timeline_layout(edl, fps)
    parts = []
    for i, p in enumerate(layout):
        s = p["in_f"] / fps
        e = p["out_f"] / fps
        parts.append(f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]")
    concat = "".join(f"[a{i}]" for i in range(len(layout))) + f"concat=n={len(layout)}:v=0:a=1[out]"
    filter_complex = ";".join(parts + [concat])
    filter_file = out_wav.with_suffix(".filter.txt")
    filter_file.write_text(filter_complex, encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
           "-filter_complex_script", str(filter_file), "-map", "[out]",
           "-c:a", "pcm_s16le", "-ar", "48000", str(out_wav)]
    subprocess.run(cmd, check=True)
    if out_mp3 is not None:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(out_wav),
                        "-c:a", "libmp3lame", "-b:a", "192k", str(out_mp3)], check=True)
    filter_file.unlink(missing_ok=True)
    return out_wav


def _norm(text: str) -> str:
    return re.sub(r"[^\w一-鿿]", "", text or "").lower()


def _bigram_sim(a: str, b: str) -> float:
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / max(1, len(A | B))


def analyze_seams(edl: list[dict], sentences: list[dict], fps: float, manual_sids: set[int] | None = None) -> list[dict]:
    """Flag clips whose boundaries are likely to sound wrong; returns one record per flagged clip."""
    by_id = {int(s["id"]): s for s in sentences}
    used: set[int] = set()
    for c in edl:
        used.update(int(x) for x in c.get("sentence_ids", []))
    layout = {p["clip_id"]: p for p in timeline_layout(edl, fps)}
    manual_sids = manual_sids or set()
    records = []
    for idx, c in enumerate(edl):
        flags: list[dict] = []
        sids = [int(x) for x in c.get("sentence_ids", [])]
        if not sids:
            continue
        last_s = by_id.get(sids[-1])
        first_s = by_id.get(sids[0])
        kept_text = (c.get("whisper_transcript") or c.get("transcript") or "").strip()

        # 1. Mid-sentence cut: no closing punctuation and the next source sentence continues within 0.8s
        if last_s is not None and not kept_text.endswith(CLOSURE_PUNCT):
            nxt = by_id.get(sids[-1] + 1)
            if nxt is not None and int(nxt["id"]) not in used and nxt.get("speaker_id") in (None, "HOST", "SPEAKER_00"):
                gap = float(nxt["start"]) - float(last_s["end"])
                nxt_norm = _norm(nxt["text"])
                is_retake = _bigram_sim(_norm(last_s["text"])[-12:], nxt_norm[:12]) >= 0.35
                if 0.0 <= gap <= 0.8 and not is_retake and len(nxt_norm) >= 3:
                    flags.append({"type": "CUT_MID_SENTENCE",
                                  "detail": f"下一句（S{nxt['id']} {nxt['text'][:30]}）在 {gap:.2f}s 後接著講，卻沒被選入"})

        # 2. Trailing / leading words trimmed by a shorter model transcript
        if last_s is not None:
            full_tail = _norm(last_s["text"])
            kept_norm = _norm(kept_text)
            if full_tail and kept_norm and not full_tail.endswith(kept_norm[-min(6, len(kept_norm)):]):
                t_last = float(c.get("t_last", 0.0))
                if float(last_s["end"]) - t_last > 0.35:
                    flags.append({"type": "TAIL_TRIMMED",
                                  "detail": f"原句尾還有 {float(last_s['end']) - t_last:.2f}s 的話被截掉：…{last_s['text'][-24:]}"})
        if first_s is not None:
            full_head = _norm(first_s["text"])
            kept_norm = _norm(kept_text)
            if full_head and kept_norm and not full_head.startswith(kept_norm[:min(6, len(kept_norm))]):
                t_first = float(c.get("t_first", 0.0))
                if t_first - float(first_s["start"]) > 0.35:
                    flags.append({"type": "HEAD_TRIMMED",
                                  "detail": f"原句開頭有 {t_first - float(first_s['start']):.2f}s 被截掉：{first_s['text'][:24]}…"})

        # 3. Alternatives: dropped HOST sentences within 90s that say the same thing
        alts = []
        kept_norm = _norm(kept_text)
        for s in sentences:
            sid = int(s["id"])
            if sid in used or s.get("speaker_id") not in (None, "HOST", "SPEAKER_00"):
                continue
            if abs(float(s["start"]) - float(c["source_in"])) > 90.0:
                continue
            sn = _norm(s["text"])
            if len(sn) >= 8 and _bigram_sim(sn, kept_norm) >= 0.30:
                alts.append(f"S{sid} {tc(float(s['start']), fps)} {s['text'][:26]}")
        if alts:
            flags.append({"type": "HAS_ALTERNATIVE", "detail": "另有版本：" + "；".join(alts[:3])})

        # 4. Manual edits and model uncertainty
        if any(sid in manual_sids for sid in sids) or "人工" in str(c.get("take_selection_reason", "")):
            flags.append({"type": "MANUAL", "detail": "人工補回或修改過的片段"})
        reason = str(c.get("take_selection_reason", "")) + " " + str(c.get("visual_check", ""))
        if any(w in reason for w in UNCERTAIN_WORDS):
            flags.append({"type": "LOW_CONFIDENCE", "detail": f"模型理由含不確定用語：{reason[:60]}"})
        if str(c.get("role", "host")) == "offscreen":
            flags.append({"type": "OFFSCREEN", "detail": "場外聲音角色，請確認要不要留"})

        if flags:
            p = layout[c["clip_id"]]
            records.append({
                "clip_id": c["clip_id"],
                "timeline_tc": tc(p["start_f"] / fps, fps),
                "timeline_end_tc": tc(p["end_f"] / fps, fps),
                "timeline_start_frame": p["start_f"],
                "source_tc": tc(float(c["source_in"]), fps),
                "source_end_tc": tc(float(c["source_out"]), fps),
                "sentence_ids": sids,
                "text": kept_text[:60],
                "flags": flags,
            })
    return records


def analyze_boundaries_acoustically(edl: list[dict], video_path: Path, temp_dir: Path, fps: float,
                                    threshold_db: float = -35.0) -> list[dict]:
    """Ground-truth check of every in/out point against the source audio.

    HEAD_BLEED: speech energy right before the in-point (the previous phrase runs straight
    into this clip, so the first 100 ms may carry the tail of a dropped sentence).
    TAIL_CONTINUES: speech energy right after the out-point while the clip ends in speech
    (the host keeps talking, so the last syllable may be clipped or the cut lands mid-flow).
    Both are review hints for the editor, not automatic fixes.
    """
    import numpy as np

    from .roles import extract_stereo_wav, load_stereo_audio

    temp_dir.mkdir(parents=True, exist_ok=True)
    wav = temp_dir / f"temp_{Path(video_path).stem}_boundaries.wav"
    extract_stereo_wav(video_path, wav)
    try:
        audio, sr = load_stereo_audio(wav)
    finally:
        wav.unlink(missing_ok=True)

    def level(t0: float, t1: float) -> float:
        i0, i1 = max(0, int(t0 * sr)), min(audio.shape[0], int(t1 * sr))
        if i1 <= i0:
            return -120.0
        seg = audio[i0:i1]
        return float(max(20.0 * np.log10(max(float(np.sqrt(np.mean(seg[:, ch] ** 2))), 1e-9)) for ch in range(seg.shape[1])))

    out = []
    for c in edl:
        si, so = float(c["source_in"]), float(c["source_out"])
        flags = []
        pre = level(si - 0.12, si - 0.02)
        if pre > threshold_db:
            flags.append({"type": "HEAD_BLEED", "detail": f"入點前 0.1 秒仍有語音（{pre:.0f} dBFS），開頭可能帶到前一句尾音，請聽第一個字"})
        last_in = level(so - 0.12, so - 0.02)
        post = level(so + 0.02, so + 0.12)
        if last_in > threshold_db and post > threshold_db:
            flags.append({"type": "TAIL_CONTINUES", "detail": f"出點後語音仍在持續（{post:.0f} dBFS），請聽最後一個字是否完整"})
        if flags:
            out.append({"clip_id": c["clip_id"], "flags": flags})
    return out


def merge_flag_records(base: list[dict], extra: list[dict], edl: list[dict], fps: float) -> list[dict]:
    """Merge extra per-clip flags into seam records (creating records for clips not yet flagged)."""
    layout = {p["clip_id"]: p for p in timeline_layout(edl, fps)}
    by_clip = {r["clip_id"]: r for r in base}
    clips = {c["clip_id"]: c for c in edl}
    for e in extra:
        cid = e["clip_id"]
        if cid in by_clip:
            by_clip[cid]["flags"].extend(e["flags"])
            continue
        c = clips[cid]
        p = layout[cid]
        by_clip[cid] = {
            "clip_id": cid,
            "timeline_tc": tc(p["start_f"] / fps, fps),
            "timeline_end_tc": tc(p["end_f"] / fps, fps),
            "timeline_start_frame": p["start_f"],
            "source_tc": tc(float(c["source_in"]), fps),
            "source_end_tc": tc(float(c["source_out"]), fps),
            "sentence_ids": [int(x) for x in c.get("sentence_ids", [])],
            "text": (c.get("whisper_transcript") or c.get("transcript") or "")[:60],
            "flags": list(e["flags"]),
        }
    return sorted(by_clip.values(), key=lambda r: r["timeline_start_frame"])


def records_to_markers(records: list[dict]) -> list[dict]:
    """Sequence markers for the XML exporter: one per flagged clip, at the clip's timeline start."""
    markers = []
    for r in records:
        kinds = "/".join(sorted({f["type"] for f in r["flags"]}))
        comment = " | ".join(f"{f['type']}: {f['detail']}" for f in r["flags"])
        markers.append({
            "name": f"確認 {kinds} #{r['clip_id']}",
            "comment": f"原片 {r['source_tc']}-{r['source_end_tc']} | {comment}",
            "frame": int(r["timeline_start_frame"]),
        })
    return markers


def write_review_markdown(records: list[dict], out_md: Path, title: str, extra_sections: list[str] | None = None) -> None:
    lines = [f"# {title}", "", "時碼為成片時間軸（Premiere 的 sequence 時間），括號內為原片時碼。", ""]
    lines.append("| # | 成片時碼 | 原片時碼 | 類型 | 說明 | 文字 |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        kinds = ", ".join(sorted({f["type"] for f in r["flags"]}))
        detail = "<br>".join(f["detail"] for f in r["flags"])
        lines.append(f"| {r['clip_id']} | {r['timeline_tc']} | {r['source_tc']}–{r['source_end_tc']} | {kinds} | {detail} | {r['text']} |")
    for sec in extra_sections or []:
        lines += ["", sec]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_clip_exact_readthrough(edl: list[dict], cut_segments: list[dict], fps: float,
                                 timestamp_bias_sec: float = 0.20) -> str:
    """Read-through text with one line per clip and a ‖ mark at every cut.

    Words from the re-transcribed cut audio are assigned to the clip that contains the
    word's midpoint. Whisper word timestamps on the cut start ~0.2 s early (measured median
    0.18 s on C7411), so `timestamp_bias_sec` is added before assignment; without it the
    first character of a clip lands on the previous line and reads like a word cut in half.
    """
    layout = timeline_layout(edl, fps)

    def clip_at(t: float) -> dict:
        f = int(round(t * fps))
        for p in layout:
            if p["start_f"] <= f < p["end_f"]:
                return p
        return layout[-1]

    words: list[tuple[float, float, str]] = []
    for seg in cut_segments:
        ws = seg.get("words") or [{"word": seg.get("text", ""), "start": seg.get("start", 0.0), "end": seg.get("end", 0.0)}]
        for w in ws:
            words.append((float(w.get("start", 0.0)), float(w.get("end", 0.0)), str(w.get("word", ""))))

    lines = ["說明：每行是成片中的一個 clip；‖ 代表剪接點。文字來自成片音軌重新轉錄，字在哪個 clip 以字的中點時間判定。"]
    cur = None
    buf: list[str] = []

    def flush():
        if cur is not None:
            lines.append(
                f"[clip {cur['clip_id']:2d} | 成片 {tc(cur['start_f'] / fps, fps)} | 原片 {tc(cur['in_f'] / fps, fps)}-{tc(cur['out_f'] / fps, fps)}] "
                + "".join(buf).strip() + " ‖"
            )

    for ws, we, w in words:
        p = clip_at((ws + we) / 2.0 + timestamp_bias_sec)
        if cur is None or p["clip_id"] != cur["clip_id"]:
            flush()
            cur = p
            buf = []
        buf.append(w)
    flush()
    return "\n".join(lines) + "\n"


def build_readthrough_review_prompt(prompt_path: Path, readthrough_text: str, edl: list[dict],
                                    sentences: list[dict], fps: float) -> str:
    """Assemble the Gemini read-through review prompt (text only)."""
    used: set[int] = set()
    for c in edl:
        used.update(int(x) for x in c.get("sentence_ids", []))
    layout = {p["clip_id"]: p for p in timeline_layout(edl, fps)}
    clip_lines = []
    for c in edl:
        p = layout[c["clip_id"]]
        role = "場外" if str(c.get("role", "host")) == "offscreen" else "主持"
        clip_lines.append(
            f"clip {c['clip_id']:2d} 成片 {tc(p['start_f'] / fps, fps)} | 原片 {tc(float(c['source_in']), fps)}-{tc(float(c['source_out']), fps)} "
            f"| {role} | S{c.get('sentence_ids', [])} | {(c.get('whisper_transcript') or c.get('transcript') or '')[:90]}"
        )
    dropped_lines = []
    for s in sentences:
        sid = int(s["id"])
        if sid in used:
            continue
        if s.get("speaker_id") not in (None, "HOST", "SPEAKER_00"):
            continue
        if float(s["end"]) - float(s["start"]) < 0.8:
            continue
        dropped_lines.append(f"S{sid:3d} 原片 {tc(float(s['start']), fps)}-{tc(float(s['end']), fps)} | {s['text'][:100]}")
    base = Path(prompt_path).read_text(encoding="utf-8")
    return (
        base
        + "\n\n---\n## 成片通讀稿（Whisper 重新轉錄成片音軌）\n" + readthrough_text
        + "\n\n---\n## clip 清單\n" + "\n".join(clip_lines)
        + "\n\n---\n## 沒有被選入的主持人句子（可建議加回）\n" + "\n".join(dropped_lines)
    )


def run_readthrough_review(prompt: str, model: str, project_id=None, location=None) -> tuple[dict, dict, float]:
    """Text-only Gemini call; returns (parsed_json, usage, seconds)."""
    import time

    from google.genai import types

    from .gemini_client import (
        _extract_text_from_response,
        _generate_content,
        _usage_from_generate_content,
        get_gemini_client,
        load_env_file,
        parse_edl_json_text,
    )
    from .constants import GEMINI_MAX_OUTPUT_TOKENS

    load_env_file()
    client = get_gemini_client(project_id=project_id, location=location)
    config_kwargs = {"response_mime_type": "application/json", "temperature": 0.0, "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS}
    if hasattr(types, "ThinkingConfig"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=16384)
    t0 = time.time()
    response = _generate_content(client, model=model, contents=[prompt], config=types.GenerateContentConfig(**config_kwargs))
    raw = _extract_text_from_response(response)
    usage = _usage_from_generate_content(response)
    data = parse_edl_json_text(raw)  # tolerant JSON parser; `final_edl` key is irrelevant here
    data.pop("final_edl", None)
    return data, usage, time.time() - t0


def review_to_markers(review: dict, fps: float) -> list[dict]:
    """Sequence markers from the Gemini seam list."""
    markers = []
    for s in review.get("seams", []) or []:
        tcs = str(s.get("timeline_tc", "00:00:00:00"))
        try:
            hh, mm, ss, ff = (int(x) for x in tcs.split(":"))
            frame = ((hh * 3600 + mm * 60 + ss) * int(round(fps))) + ff
        except Exception:  # noqa: BLE001
            continue
        markers.append({
            "name": f"評審 {s.get('severity', '')} {s.get('type', '')} #{s.get('clip_id', '?')}",
            "comment": f"{s.get('description', '')} | 修法: {s.get('fix', '')} | 原片 {s.get('source_tc', '')}",
            "frame": frame,
        })
    return markers


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="成片通讀評審：輸出音軌、重新轉錄、接點分析、Gemini 連貫性評讀、標記 XML")
    parser.add_argument("--edl-json", required=True, help="<base>_<tag>_edl.json（含 final_edl）")
    parser.add_argument("--sentences-json", required=True, help="<base>_whisper_sentences.json（含角色）")
    parser.add_argument("--video", required=True, help="原片路徑（XML 連結的檔案）")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=24000 / 1001)
    parser.add_argument("--readthrough", default=None, help="已產生的通讀稿 .txt；省略則不做 Gemini 評讀（除非加 --render）")
    parser.add_argument("--render", action="store_true",
                        help="輸出成片音軌（WAV 與 MP3）、用 Whisper 重新轉錄、產生以 clip 為單位對齊的通讀稿")
    parser.add_argument("--gemini", action="store_true", help="執行 Gemini 連貫性評讀（純文字，不上傳影片）")
    parser.add_argument("--model", default=None)
    parser.add_argument("--xml-out", default=None, help="寫出帶 marker 的 FCP7 XML")
    parser.add_argument("--xml-profile", choices=["premiere", "davinci"], default="premiere",
                        help="FCP7 XML 口味（同 video_trimmer.py --xml-profile）：premiere（預設）或 davinci")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edl = load_json(Path(args.edl_json))["final_edl"]
    sentences = load_json(Path(args.sentences_json))
    fps = args.fps

    records = analyze_seams(edl, sentences, fps)
    try:
        acoustic = analyze_boundaries_acoustically(edl, Path(args.video), out_dir, fps)
        records = merge_flag_records(records, acoustic, edl, fps)
        logger.info("聲學邊界檢查: %d 個 clip 的入點或出點貼著語音", len(acoustic))
    except Exception as exc:  # noqa: BLE001
        logger.warning("聲學邊界檢查略過: %s", exc)
    markers = records_to_markers(records)
    review = None

    readthrough_text = Path(args.readthrough).read_text(encoding="utf-8") if args.readthrough else None
    if args.render:
        from .transcribe import transcribe_video_whisper

        stem = Path(args.edl_json).stem.replace("_edl", "")
        wav = out_dir / f"{stem}_cut.wav"
        mp3 = out_dir / f"{stem}_cut.mp3"
        render_cut_audio(edl, Path(args.video), wav, mp3, fps)
        logger.info("成片音軌: %s", mp3)
        cut_raw, _cut_sents = transcribe_video_whisper(wav, out_dir / f"{stem}_cut_whisper_raw.json", model_name="large-v3")
        readthrough_text = build_clip_exact_readthrough(edl, cut_raw, fps)
        (out_dir / f"{stem}_cut_readthrough.txt").write_text(readthrough_text, encoding="utf-8")
        logger.info("通讀稿: %s", out_dir / f"{stem}_cut_readthrough.txt")

    if args.gemini and readthrough_text:
        prompt_path = Path(__file__).parent.parent / "prompts" / "cut_review_prompt.md"
        prompt = build_readthrough_review_prompt(prompt_path, readthrough_text, edl, sentences, fps)
        model = args.model or __import__("os").environ.get("MODEL_NAME") or "gemini-3.8-flash"
        review, usage, secs = run_readthrough_review(prompt, model)
        (out_dir / "cut_review_gemini.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Gemini 評讀完成 %.1fs | tokens=%s", secs, usage.get("total_tokens"))
        markers += review_to_markers(review, fps)

    (out_dir / "cut_review_seams.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    extra = []
    if review:
        extra.append(f"## Gemini 通讀評估\n\n分數：{review.get('coherence_score_0_10')} / 10\n\n{review.get('overall_assessment', '')}\n\n開頭搶拍主因：{review.get('opening_problem_cause', '')}")
        rows = ["| 優先 | 成片時碼 | clip | 動作 | 理由 |", "|---|---|---|---|---|"]
        for p in review.get("prioritized_changes", []) or []:
            rows.append(f"| {p.get('priority')} | {p.get('timeline_tc', '')} | {p.get('clip_id', '')} | {p.get('action', '')} | {p.get('why', '')} |")
        extra.append("## Gemini 優先修改清單\n\n" + "\n".join(rows))
    write_review_markdown(records, out_dir / "cut_review.md", "成片接點檢查清單", extra)

    if args.xml_out:
        from .exporters import generate_fcp7_xml

        total_dur = max(float(c["source_out"]) for c in edl) + 1.0
        try:
            from .render import probe_video
            total_dur, w, h, probed_fps = probe_video(Path(args.video))
            args.width, args.height = w, h
        except Exception:  # noqa: BLE001
            pass
        markers.sort(key=lambda m: m["frame"])
        generate_fcp7_xml(edl, Path(args.video), total_dur, Path(args.xml_out), args.width, args.height, fps,
                          markers=markers, xml_profile=args.xml_profile)
        logger.info("帶 %d 個 marker 的 XML: %s", len(markers), args.xml_out)
    logger.info("接點檢查: %d 個 clip 需確認 -> %s", len(records), out_dir / "cut_review.md")
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        import sys as _sys

        _skill_root = Path(__file__).resolve().parent.parent
        if str(_skill_root) not in _sys.path:
            _sys.path.insert(0, str(_skill_root))
        __package__ = "scripts"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

# Video Trimmer — Workspace & Development Rules (AGENTS.md)

This file defines the authoritative rules for AI Coding Agents (Google Antigravity / Jetski, Claude Code, Codex, etc.) working in this repository. It covers both **Client Execution Invariants** (when running the video trimming pipeline for end users) and **Repository Engineering Standards** (when developing, maintaining, or extending this project).

---

## Part I: Operational Invariants (When Executing Video Trimmer Tasks)

1. **Strict Toolset Execution Only (No Ad-Hoc Scripts)**:
   - Execute all video rough-cutting, take selection, acoustic onset snapping, NLE XML/FCPXML/EDL exporting, and video rendering exclusively via the official scripts in `skills/video-trimmer/scripts/`.
   - Writing temporary Python scripts, ad-hoc regex deduplication, or custom audio/video trimming logic is **STRICTLY FORBIDDEN**.
2. **Mandatory 3-Step Gated Workflow (Direct CLI Invocation)**:
   - Resolve `<PLUGIN_ROOT>` as two directory levels above `skills/video-trimmer/SKILL.md` (`../../`, e.g., `/Users/sylph/.gemini/config/plugins/video-trimmer`).
   - Follow the 3-Step Runbook defined in [SKILL.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/SKILL.md):
     - **Step 1 (Environment & Cloud Auth Verification)**: Verify FFmpeg, `gcloud` ADC credentials, and `.env` configuration (`GOOGLE_CLOUD_PROJECT`, `VIDEO_TRIMMER_BUCKET`) from `<PLUGIN_ROOT>`.
     - **Step 2 (Pipeline Execution)**: Run `python3 skills/video-trimmer/scripts/video_trimmer.py` (with `Cwd` set to `<PLUGIN_ROOT>`) directly via `run_command`. Pass `-s <SCRIPT_FILE>` whenever the user provides a shooting script or outline (Mode A: Monotonic Script-Anchored Alignment); omit `-s` for unscripted recordings (Mode B: Unscripted Intent-Window Arbitration). Run in default Static Multimodal mode (`MEDIA_RESOLUTION_LOW`) unless `--agentic` is explicitly requested. By default, all deliverables are automatically isolated in `<input_dir>/output/` (`./output/` for Google Drive inputs).
     - **Step 3 (Deliverable & `agent_verdict` Quality Gate Verification)**: Verify that all required outputs (`_trimmed.mp4`, `_edl.fcpxml`, `_edl.xml`, `_edl.json`, `_edl.csv`, `_edl_report.md`, `_edl_report.json`, and `_whisper_raw.json`) exist in `<OUTPUT_DIR>` (`<input_dir>/output/` by default) and are non-empty (`> 0 bytes`). Inspect the top-level `agent_verdict` in `_edl_report.json`.
3. **Fail-Fast on Cloud Errors & One-Shot Self-Healing Protocol**:
   - **Fail-Fast on Cloud/Auth Errors (Exit Code `1`)**: If any script exits with code `1` (e.g., missing ADC credentials, 403/401 GCS/Vertex AI permission error, or missing FFmpeg), stop immediately, report the exact error and exit status, and instruct the user to run `./setup.sh --project YOUR_PROJECT_ID` or `gcloud auth application-default login`.
   - **One-Shot Self-Healing on Quality Gate Failure (Max 1 Retry)**: If `_edl_report.json` reports `agent_verdict.pass_quality_gate == false` (`suggested_action == "ONE_SHOT_REMEDIATE"`, or exit code `2` with `--strict`), execute **at most ONE** remediation re-run (`remediation_cmd`). If the second run still fails the quality gate, stop immediately and report `[Degraded]` along with `fatal_violations`.
4. **Dynamic Language Mirroring & Strict Zero-Emoji Policy**:
   - Respond to the user in their prompt language (Traditional Chinese `zh-TW` when prompted in Traditional Chinese, English when prompted in English, Japanese when prompted in Japanese, etc.).
   - Do NOT use decorative emojis or icons in section headings, tables, or generated EDL reports.

---

## Part II: Repository Development & Engineering Standards (When Developing This Project)

When modifying code, prompts, infrastructure scripts, or documentation in this repository, you MUST adhere to the following engineering standards:

### 1. Single Source of Truth (SSOT) Directory Architecture (Agent Plugins 1.0 Specification)
- **Canonical Code Location**: All core Python scripts (`scripts/*.py`) and prompt templates (`prompts/*.md`) physically reside inside `skills/video-trimmer/scripts/` and `skills/video-trimmer/prompts/` in compliance with the [Agent Plugins 1.0 Specification](https://agent-plugins.org/specification) (§4.2 & §7.1).
- **Rule**: Always edit files under `skills/video-trimmer/scripts/` and `skills/video-trimmer/prompts/`. Do not create root-level symlinks or duplicate physical directories at the repository root.

### 2. 5-Layer Unified Architecture, Surgical Micro-Window Repair, & Zero Semantic String-Matching Invariant
- **Zero Python Semantic Retake Guessing**: Never use Python string-similarity heuristics (`difflib`, character overlap ratios, or regex keyword matching) in [transcribe.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/transcribe.py) to classify semantic retakes or filter sentences. Semantic take arbitration belongs exclusively to the LLM in [video_cut_prompt.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/prompts/video_cut_prompt.md).
- **Layer 1 — Pure Acoustic & Punctuation Clause Segmentation**:
  - `merge_whisper_segments_to_sentences` and `_split_sentence_on_paused_restarts` in [transcribe.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/transcribe.py) split `Sentence ID` units purely on physical boundaries: breath pauses (`gap >= 0.20s`), stretched word onsets (`word_dur >= 1.20s` and `>= 0.45s/char`), punctuation closure, and speaker turns, while preserving `CONJUNCTIONS` attachment across micro-pauses (`gap < 0.25s`).
- **Layer 2 — Dual-Mode LLM Take Arbitration & Surgical Micro-Window Repair**:
  - [gemini_client.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/gemini_client.py) and [video_cut_prompt.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/prompts/video_cut_prompt.md) support **Mode A** (Monotonic Script-Anchored Alignment via `[Script Block NN]`, max one winning take per block) and **Mode B** (Unscripted Intent-Window Arbitration, pruning abandoned fragments while preserving intentional rhetorical repetition).
  - Before video rendering, `repair_edl_micro_windows` in [edl_auditor.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/edl_auditor.py) detects any missed script blocks, unanchored clips, or adjacent retake candidates and re-scans only the affected `15s–90s` video micro-window via `VideoMetadata(start_offset=..., end_offset=...)`, avoiding full-video gateway timeouts.
- **Layer 3 — Sub-Unit Expansion & Word-Boundary Trimming**:
  - `resolve_clip_sub_units` and `_trim_matched_words_by_transcript` in [transcribe.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/transcribe.py) expand multi-sentence spans and trim word boundaries (`t_first`, `t_last`) to match `clip_data["transcript"]`.
- **Layer 4 — Global Cross-Clip Coalescing**:
  - `coalesce_adjacent_sub_units` in [transcribe.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/transcribe.py) merges consecutive `Sentence ID`s across adjacent clips when the physical inter-word gap is `< 0.40s` and no `Sentence ID` is skipped, eliminating artificial jump-cuts and redundant micro-fades inside continuous sentences.
- **Layer 5 — Deterministic Timeline Sanitization & 8-Dimension Quality Audit**:
  - `sanitize_refined_edl` and `audit_edl_quality` in [edl_auditor.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/edl_auditor.py) enforce `source_out >= t_last`, resolve boundary overlaps, coalesce `< 0.45s` flash-frame micro-clips, and generate `<base>_<tag>_edl_report.json` and `<base>_<tag>_edl_report.md` with a top-level `agent_verdict`.

### 3. Multi-NLE Timeline Interoperability & Acoustic Integrity
- **Universal NLE Support**: [exporters.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/exporters.py) produces frame-accurate timelines for Apple Final Cut Pro (`.fcpxml`), Adobe Premiere Pro (`.xml`), DaVinci Resolve (`.xml` / `.fcpxml`), CMX 3600 (`.edl`), and CSV (`.csv`) across standard broadcast and cinema frame rates (`23.976` to `60` fps).
- **Audio Pop Protection (`15ms` Equal-Power Micro-Crossfade)**: Every rendered cut boundary in [render.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/render.py) must apply a `15ms` equal-power micro-fade (`afade=t=in:d=0.015:curve=iqsin` and `afade=t=out:d=0.015:curve=qsin`).
- **Speech Boundary Tightening**: [acoustic.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/acoustic.py) dynamically calculates lead-in and lead-out margins from presenter CPS without truncating spoken phonemes.

### 4. 100% Google Cloud Vertex AI (ADC) + GCS Architecture
- **Designated Model & Zero API Key Policy**: All Gemini invocations in [gemini_client.py](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/scripts/gemini_client.py) use `genai.Client(vertexai=True, project=..., location=...)` via Application Default Credentials (ADC), defaulting to `MODEL_NAME=gemini-3.8-flash` and `GOOGLE_CLOUD_LOCATION=global`. Never introduce non-designated model IDs or legacy AI Studio API keys.
- **GCS Infrastructure & Ephemeral Staging (`setup.sh`)**: Raw videos staged under `gs://${BUCKET}/raw/` are deleted in `finally` blocks after inference and backed by a 2-day bucket lifecycle deletion policy provisioned via [setup.sh](file:///Users/sylph/Documents/Antigravity/video-trimmer/setup.sh).

### 5. ASD-STE100 English, Strict Generality, & Unit Testing Gate
- **ASD-STE100 & Zero Entity Hardcoding**: Write all Python code, docstrings, comments, and prompt templates in concise ASD-STE100 English. Use only generic placeholders (`raw_footage.mp4`, `shooting_script.md`, `rough_cut.fcpxml`) and never hardcode test-specific names or video titles.
- **Mandatory Unit Test Gate**: Run the full unit test suite and verify 100% pass rate before committing any change:
  ```bash
  python3 -m unittest discover -s tests -v
  ```

### 6. Antigravity Plugin Architecture, 5-Language Parity, & Commit Policy
- **Plugin & README Synchronization**: Keep [plugin.json](file:///Users/sylph/Documents/Antigravity/video-trimmer/plugin.json), [rules/AGENTS.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/rules/AGENTS.md), [skills/video-trimmer/SKILL.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/skills/video-trimmer/SKILL.md), and all 5 language READMEs ([README.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/README.md), [README.zh-TW.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/README.zh-TW.md), [README.zh-CN.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/README.zh-CN.md), [README.ja.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/README.ja.md), [README.ko.md](file:///Users/sylph/Documents/Antigravity/video-trimmer/README.ko.md)) synchronized at all times.
- **Human Authorship Only**: Author all commits as `sylphlin <sylph.lin@gmail.com>` with zero AI assistant branding or `Co-Authored-By` trailers.

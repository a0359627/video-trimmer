# Video Trimmer - Operational Invariants for AI Clients

When you execute tasks or skills from this plugin, you MUST follow these operational rules:

## 1. Strict Read-Only Execution & Direct CLI Invocation (Do Not Modify Plugin Code)
- All Python scripts (`skills/video-trimmer/scripts/*.py`), prompt specifications (`skills/video-trimmer/prompts/*.md`), and configuration files are read-only tools.
- Do NOT edit, patch, or rewrite any files in this plugin with `replace_file_content`, `write_to_file`, or shell commands.
- Do NOT write ad-hoc temporary Python scripts, custom regex deduplication, or one-off audio/video trimming scripts.
- Resolve `<PLUGIN_ROOT>` as two directory levels above `skills/video-trimmer/SKILL.md` (`../../`, e.g., `/Users/sylph/.gemini/config/plugins/video-trimmer`).
- Set `Cwd` to `<PLUGIN_ROOT>` and run `python3 skills/video-trimmer/scripts/video_trimmer.py` directly with `run_command` after verifying the environment (`Step 1`). Do NOT search for global `video-trimmer` CLI aliases with `find_by_name` or `list_dir`.

## 2. Fail-Fast on Errors & Exit Gate Verification (Do Not Debug or Rewrite Code)
- If a script fails (exit code is not 0) or an external error occurs (such as 401 Unauthorized, 403 Forbidden, Quota Exceeded, missing Application Default Credentials, or missing FFmpeg):
  - Stop immediately.
  - Show the exact error message and exit status to the user.
  - Give a clear, actionable solution to the user (for example, run `./setup.sh --project YOUR_PROJECT_ID`, run `gcloud auth application-default login`, or install FFmpeg).
  - Do NOT try to modify the script, probe different code paths, or rewrite logic.
- Verify all required deliverable files (`.mp4`, `.fcpxml`, `.xml`, `.edl`, `.csv`, `_edl_report.md`) exist on disk and are non-empty (`> 0 bytes`) before declaring task completion.

## 3. Strict Zero-Emoji Policy & Dynamic Language Mirroring
- Do NOT use emojis or decorative icons in headings or tables in generated Markdown reports or EDL notes.
- Keep all documentation and reports in plain, professional technical text.
- Always respond to the user in their prompt language (Traditional Chinese `zh-TW` when prompted in Traditional Chinese, English when prompted in English, Japanese when prompted in Japanese, etc.).

## 4. Acoustic Ground Truth & Dual-Mode Take Arbitration Integrity
- Execute the workflow defined in `skills/video-trimmer/SKILL.md`.
- Pass `-s <SCRIPT_FILE>` whenever the user provides a reference shooting script or outline to activate Mode A (Monotonic Script-Anchored Alignment); omit `-s` for unscripted recordings to activate Mode B (Unscripted Intent-Window Arbitration).
- All cut points and keep-ranges must respect the Whisper word-level acoustic ground truth (`word_timestamps=True`), onset snapping, and `Sentence ID` boundaries.
- Do NOT truncate speech phonemes or spoken words. Maintain dynamic speech padding and 15ms equal-power audio micro-crossfades (`iqsin`/`oqsin`) to eliminate audio pop artifacts.

## 5. Default Static Multimodal Execution
- Run `video_trimmer.py` in `<PLUGIN_ROOT>` in default Static Multimodal mode (`MEDIA_RESOLUTION_LOW`) for fast, timeout-free inference.
- Include `--agentic` only when the user explicitly requests Agentic Video Understanding mode.

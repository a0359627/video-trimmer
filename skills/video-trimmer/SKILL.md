---
name: video-trimmer
description: AI-Powered Smart Video Trimmer and rough-cut engine adhering to Agent Skills Specification. Features Gemini 3.8 Flash Multimodal Video Understanding (native video, no hallucinated cuts, last-take-wins), Whisper Word-Level Acoustic Ground Truth (Apple Silicon Metal GPU via mlx-whisper), Acoustic Onset Snapping (Smart Gap Shortening), 15ms Audio Equal-Power Micro-Crossfade, and multi-format NLE Project Export (Premiere Pro XML, DaVinci Resolve XML, Final Cut Pro FCPXML).
metadata:
  version: "1.0.0"
  author: "sylphlin"
  repository: "https://github.com/sylphlin/video-trimmer"
  category: "video-editing"
  specification: "https://agentskills.io/specification"
---

# Video Trimmer (`video-trimmer`)

AI-Powered Smart Video Trimmer and rough-cut engine adhering to the open [Agent Skills Specification](https://agentskills.io/specification).

Tailored for talking-head videos, tech explainers, video podcasts, tutorials, and multi-take raw footage. Combines **Gemini 3.8 Flash Multimodal Video Understanding** with local **Whisper Word-Level Acoustic Ground Truth** (with Apple Silicon Metal GPU acceleration via `mlx-whisper`), **Acoustic Onset Snapping (Smart Gap Shortening)**, and **15ms Audio Equal-Power Micro-Crossfades**.

---

## Directory Structure

This project complies with the [Agent Plugins 1.0 Specification](https://agent-plugins.org/) and the [Agent Skills Specification](https://agentskills.io/specification):

```text
video-trimmer/
├── plugin.json                       # Agent Plugins 1.0 specification manifest
├── rules/
│   └── AGENTS.md                     # Strict read-only & fail-fast operational invariants for AI clients
├── skills/
│   └── video-trimmer/
│       ├── SKILL.md                  # Skill definition and agent reference manual
│       ├── scripts/                  # Canonical core implementation modules (SSOT)
│       │   ├── __init__.py
│       │   ├── video_trimmer.py      # Master rough-cut orchestrator & acoustic engine
│       │   ├── constants.py          # Centrally managed named constants
│       │   ├── exceptions.py         # Custom exception hierarchy
│       │   ├── acoustic.py           # CPS calculation & text-locked acoustic bounds
│       │   ├── transcribe.py         # Whisper transcription, clause segmentation, cross-clip coalescing
│       │   ├── gemini_client.py      # Vertex AI (ADC) client & multimodal inference
│       │   ├── gcs_utils.py          # Google Cloud Storage upload & ephemeral cleanup
│       │   ├── exporters.py          # FCP7 XML / FCPXML / CSV generation
│       │   └── render.py             # ffprobe inspection & ffmpeg final render
│       └── prompts/                  # Canonical multimodal prompting specifications (SSOT)
│           └── video_cut_prompt.md   # Dual-mode take arbitration & 5-rule subtraction prompt
├── AGENTS.md                         # Workspace & engineering development rules (ASD-STE100 English)
├── README.md                         # Public GitHub README documentation
├── LICENSE                           # MIT License
├── .env.example                      # Environment variables template for Vertex AI & GCS
├── setup.sh                          # 100% Native gcloud GCP provisioning script (Zero Terraform)
├── pyproject.toml                    # Standard Python packaging & CLI console scripts
├── requirements.txt                  # Python runtime dependencies
└── tests/                            # Offline unit tests
```

---

## Key Capabilities & Architectural Pillars

1. **Multimodal Native Understanding (Vertex AI Gemini 3.8 Flash)**:
   - Eliminates fragile text-only editing.
   - Stages raw footage directly to Google Cloud Storage (GCS) with automatic ephemeral lifecycle cleanup.
   - Evaluates presenter eye contact, facial expressions, stuttering, and retakes simultaneously via Vertex AI.
2. **Dual-Mode Take Arbitration (Mode A Script-Anchored & Mode B Unscripted Intent-Window)**:
   - **Mode A (`--script`)**: Monotonic `[Script Block NN]` alignment keeping at most one winning take per script block.
   - **Mode B (No `--script`)**: Unscripted Intent-Window arbitration pruning abandoned fragments while preserving intentional rhetorical repetition.
3. **Multimodal Active Speaker Diarization (Gemini 3.8 Flash)**:
   - Evaluates on-camera visual cues (camera gaze, mouth articulatory sync, body language) and audio acoustics (close lavalier mic vs distant room echo).
   - Accurately differentiates the on-screen target host from off-screen crew shouting section cues (e.g. "Action", "CTA-S2", "CDA84") and casual blooper chatter between takes, eliminating cumbersome local diarization models while maintaining flawless role separation.
4. **Word-Level Acoustic Ground Truth & Global Cross-Clip Coalescing**:
   - Whisper (`mlx-whisper` on Apple Silicon Metal or `faster-whisper` on CPU/CUDA) extracts phoneme-aligned word timestamps segmented purely on physical breath pauses and punctuation closure.
   - Global Cross-Clip Coalescing (`coalesce_adjacent_sub_units`) automatically merges consecutive `Sentence ID`s across adjacent clips when `gap < 0.40s`, eliminating artificial internal jump-cuts.
5. **Acoustic Onset Snapping (Smart Gap Shortening)**:
   - Scans the pre-speech dead air to snap cut-ins precisely **80ms before vocal cord vibration**, eliminating awkward pre-speech dead air and post-slate pauses.
6. **Word Ground Truth Tail & Continuous Phonation Protection**:
   - Strictly enforces `true_speech_end >= t_last` with forward-only tracking down to ambient room noise.
   - Continuous Phonation Invariant: When `tail_gap < 0.18s` (speaker in continuous speech flow), trailing words are never trimmed even if omitted from the prompt transcript, completely preserving trailing phonemes and words. Only explicit blooper tokens from the managed whitelist are pruned.
   - Accommodates voiceless consonant plosive closures (e.g. `/t/`, `/p/`, `/k/` in words like「台」) and soft trailing nasal vowels without clipping word endings.
7. **15ms Audio Equal-Power Micro-Crossfade & Hardware-Accelerated Keyframe Rendering**:
   - FFmpeg rendering uses per-clip fast keyframe input seeking (`-ss`/`-to` before `-i`), Apple Silicon `VideoToolbox` hardware decoding/encoding (`-hwaccel videotoolbox` + `h264_videotoolbox` with `libx264` fallback), a 1-second GOP (`-g 30`), and 15ms `afade` equal-power micro-fades (`iqsin`/`oqsin`) across all cut boundaries.
8. **Production Script Injection (`--script`)**:
   - Ingests production shooting scripts (`.md` / `.txt`) to guide section-by-section matching and prevent skipping intended talking points.
9. **Universal NLE Project Export & DaVinci Resolve Conform**:
   - Generates industry-standard **FCP 7 XML** (`xmeml v5` with direct `<sequence>` nesting for Adobe Premiere Pro & DaVinci Resolve), **FCPXML** (Final Cut Pro X), **EDL CSV**, and Markdown validation report (`_edl_report.md`).
   - Prevents DaVinci Resolve media disconnection by strictly enforcing collision-free XML ID namespaces (`file_id = f"{video_name}_master"`, `clipitem id = f"{video_name}_v{idx}"`, `f"{video_name}_a{idx}"`) and passing embedded camera timecode without `<frame>` tags to prevent freeze-frames.

---

## Standard Agent Workflow (Autonomous Pipeline Execution)

When an AI agent is instructed to rough-cut or trim a raw video, follow this protocol directly.
Resolve `<PLUGIN_ROOT>` as the repository or plugin root located two levels above `skills/video-trimmer/SKILL.md` (`../../`, for example `/Users/sylph/.gemini/config/plugins/video-trimmer`).
Set the command working directory (`Cwd`) to `<PLUGIN_ROOT>` and invoke `python3 skills/video-trimmer/scripts/video_trimmer.py` directly. Do not search the filesystem with `find_by_name` or `list_dir` to locate the CLI entrypoint.

### Step 1: Environment Verification & GCP Native Setup
Run a single pre-flight check in `<PLUGIN_ROOT>` to verify that FFmpeg and `.env` configuration exist:
```bash
# 1. Pre-flight verification (FFmpeg + .env configuration)
ffmpeg -version >/dev/null && test -f .env && echo "Environment Ready"
```
If `.env` is missing or Google Cloud credentials are not configured, run or instruct the user to run the one-click native `gcloud` setup:
```bash
# 2. Authenticate once with Google Cloud ADC
gcloud auth application-default login

# 3. One-click automated GCP provisioning (Creates bucket with 2-day auto-cleanup, SA, and .env):
./setup.sh
```

### Step 2: Execute Primary Video Trimmer Pipeline

Default execution uses fast Static Multimodal mode (`MEDIA_RESOLUTION_LOW`). Pass `--agentic` only when the user explicitly requests Agentic Video Understanding.

```bash
# Standard automatic rough-cut (Static Multimodal by default, Cwd = <PLUGIN_ROOT>):
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4"

# If production shooting script is provided (Mode A - Monotonic Script-Anchored Alignment):
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --script "/path/to/shooting_script.md"

# Fast-paced explainer pacing:
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --pacing compact

# Explicit output directory:
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --script "/path/to/shooting_script.md" -o "/path/to/output_dir"
```

### Step 3: Fast Local Iteration (Cached EDL Workflow)
To adjust pacing, fine-tune margins, or re-render without re-incurring cloud API inference:
```bash
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --cached-json "/path/to/raw_footage_static_edl.json" --suffix "fine_tuned"
```

---

## CLI Options Reference

| Option | Flag | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input` | `-i` | *(Required)* | Path to input raw video file (`.mp4`, `.mov`). |
| `--output-dir` | `-o` | Same as video | Directory to save all generated output files. |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini model name (defaults to `$MODEL_NAME` or `gemini-3.8-flash`). |
| `--project` | | `None` | Google Cloud Project ID (defaults to `$GOOGLE_CLOUD_PROJECT` or ADC). |
| `--region` | | `None` | Vertex AI location/region (defaults to `$GOOGLE_CLOUD_LOCATION` or `global`). |
| `--bucket` | | `None` | GCS bucket for video staging (defaults to `$VIDEO_TRIMMER_BUCKET`). |
| `--keep-gcs-upload` | | `False` | Retain ephemeral staged video in GCS instead of deleting after inference. |
| `--script` | `-s` | `None` | Path to production shooting script (`.md` / `.txt`). |
| `--agentic` | | `False` | Enable Agentic Video Understanding dynamic exploration mode. |
| `--pacing` | `-p` | `auto` | Pacing style: `auto` (adaptive CPS), `compact`, `breathing`. |
| `--cached-json`| | `None` | Path to existing EDL JSON to bypass cloud Gemini inference. |
| `--suffix` | | `None` | Custom tag suffix for generated filenames. |
| `--crf` | | `18` | FFmpeg H.264 rendering CRF parameter (18 = visually lossless). |
| `--skip-whisper`| | `False` | Skip local Whisper transcription (use pure energy fallback). |
| `--verbose` | | `False` | Verbose DEBUG-level logging (default is INFO). |

---

## Output Deliverables

For an input file `raw_footage.mp4`, the skill generates:
1. `raw_footage_<suffix>_trimmed.mp4`: High-bitrate assembled video cut with 15ms micro-fades.
2. `raw_footage_<suffix>_edl.xml`: Final Cut Pro 7 XML timeline for **Premiere Pro** & **DaVinci Resolve**.
3. `raw_footage_<suffix>_edl.fcpxml`: FCPXML timeline for **Final Cut Pro X**.
4. `raw_footage_<suffix>_edl.json`: Structured decision metadata with per-clip CPS and margins.
5. `raw_footage_<suffix>_edl.csv`: Spreadsheet table with visual/audio validation notes.
6. `raw_footage_whisper_sentences.json`: Word-level semantic sentence transcript cache.

---

## License

[MIT License](LICENSE) © 2026 sylphlin

---

## Google Drive Direct Link & GCS Smart Caching (ADC)

`video-trimmer` natively supports passing Google Drive file links (`https://drive.google.com/file/d/.../view` or `gdrive://...`) directly to `-i / --input`:
- Authenticated 100% via Application Default Credentials (`gcloud auth application-default login`).
- Automatically checks remote MD5 (`md5Checksum`) to cache locally in `<output_dir>/gdrive_inputs/` and checks `sha256` / `gdrive_md5` metadata on `gs://${VIDEO_TRIMMER_BUCKET}/raw/` to skip redundant GCS uploads.
- Example:
  ```bash
  python3 skills/video-trimmer/scripts/video_trimmer.py -i "https://drive.google.com/file/d/YOUR_VIDEO_FILE_ID/view?usp=sharing" -o output/
  ```

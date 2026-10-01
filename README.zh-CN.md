# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 项目概览 (Overview)

**Video Trimmer** 是面向口播视频、教程与演讲录像的 AI 自动粗剪引擎。系统结合 **Google Vertex AI Gemini 3.8 Flash** 多模态视频推理、**Whisper 逐字声学时间戳**（`mlx-whisper`）、**四层统一剪辑架构（4-Layer Unified Architecture）** 与 **声学起音锁定（Acoustic Onset Snapping）**。每次运行会自动剔除 NG 重录、口误与静音停顿，同时保护连贯长句不被切碎，并导出专业 NLE 时间线（`.xml`, `.fcpxml`, `.edl`, `.csv`）及渲染完成的 MP4 视频。

---

## 四层统一剪辑架构与核心技术能力

1. **第一层：纯声学与标点子句切分 (`transcribe.py`)**：
   - 仅依据物理边界切分 `Sentence ID`：换气停顿（`gap >= 0.20s`）、吃螺丝拉长音起音（`word_dur >= 1.20s`）、句尾标点与说话人轮替，同时保留微停顿（`gap < 0.25s`）下的连词黏合（`CONJUNCTIONS`）。绝不在 Python 中使用字符串相似度猜测 NG 重录。
2. **第二层：双模式 LLM 语义择优 (`gemini-3.8-flash`)**：
   - **Mode A：有讲稿单调锚定模式（传入 `--script`）**：将讲稿格式化为 `[Script Block 01] .. [Script Block NN]`，严格依序单调对齐，每个讲稿段落最多保留最后一次完整成功的 Take。
   - **Mode B：无讲稿意图视窗仲裁模式（未传 `--script`）**：剔除未完成残句重录（Abandoned Fragment），同时保护刻意修辞排比强调（如三遍重复强调）。
   - **局部小视窗精准重扫（Surgical Micro-Window Repair，`edl_auditor.py`）**：在进入 FFmpeg 渲染前，自动检测遗漏的讲稿段落（Mode A）、未绑定 `Sentence ID` 的片段或相邻疑似重录句，并仅针对该 `15s–90s` 局部视频视窗通过 `VideoMetadata` 发起单次重扫修补，避免全片 Timeout。
3. **第三层：子句展开与逐字稿边界精修 (`resolve_clip_sub_units`)**：
   - 将多句跨度展开为独立的 `Sentence ID` 子单元，并依据 `transcript` 自动精修首尾词边界（`_trim_matched_words_by_transcript`）。
4. **第四层：跨片段连贯小句无缝合一 (`coalesce_adjacent_sub_units`)**：
   - 当相邻片段为连续 `Sentence ID` 且物理字间距 `< 0.40s` 时，自动合并为单一连续片段，消除长句内部的跳接（Jump-Cut）。
5. **第五层：物理时间轴确定性自愈与 8 维度质量审计 (`edl_auditor.py`)**：
   - 自动消除相邻边界微重叠、强制保底 `source_out >= t_last`、合并 `< 0.45s` 闪帧微碎切，并生成 `<base>_<tag>_edl_report.md` 与含顶层 `agent_verdict` 质量门禁的 `<base>_<tag>_edl_report.json`。
6. **声学起音锁定、15 ms 等功率微交叉淡化与关键帧硬件加速渲染 (`acoustic.py` / `render.py`)**：
   - 将剪辑入点锁定在声带振动前 80 ms；成片渲染采用每片段前置 `-ss` / `-to` 关键帧快速定位（跳过废片解码）、Apple Silicon `VideoToolbox` 硬件编解码（`-hwaccel videotoolbox` + `h264_videotoolbox`，支持 `libx264` 自动降级）、1 秒 GOP（`-g 30`）与 15 ms 等功率淡入淡出（`afade=t=in:d=0.015:curve=iqsin` 与 `afade=t=out:d=0.015:curve=qsin`）。
7. **Agent Plugins 1.0 标准架构与多平台 NLE 时间线导出**：
   - 核心代码与提示词位于 `skills/video-trimmer/scripts/` 与 `skills/video-trimmer/prompts/`（SSOT），无根目录冗余软链接，并提供双层 `AGENTS.md` / `rules/AGENTS.md` 规范；支持导出 **FCP7 XML**、**FCPXML**、**CMX 3600 EDL** 与 **CSV**。

---

## 安装与 Google Cloud 环境配置

```bash
# 1. 安装 FFmpeg 与克隆 Agent Plugin（推荐）
brew install ffmpeg
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer

# （可选）旧版独立 Skill 目录安装（~/.gemini/config/skills/ 兼容方式）
ln -s ~/.gemini/config/plugins/video-trimmer/skills/video-trimmer ~/.gemini/config/skills/video-trimmer

pip install -r requirements.txt
pip install mlx-whisper

# 2. 授权 ADC 并运行 setup.sh
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## 命令行使用说明 (CLI Usage)

默认情况下，所有生成文件均会自动隔离保存至输入视频所在目录下的 `output/` 子目录（Google Drive 链接则为 `./output/`），也可通过 `-o` 自定义输出目录：

```bash
# Mode B：无讲稿自动粗剪（默认采用 Static Multimodal 快速模式，输出自动保存在 <input_dir>/output/）
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4"

# Mode A：配合拍摄脚本单调锚定对齐
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md"

# 明确启用 Agentic 视频理解模式
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# 直接从 Google Drive 分享链接进行粗剪（默认输出至 ./output/）
python3 skills/video-trimmer/scripts/video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
```

---

## Google Drive 直连与 GCS 双层生命周期规则

| GCS 路径前缀 (`matchesPrefix`) | 存储对象 | 保留天数 (`age`) | 说明 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 暂存原始视频 (`raw/<filename>.mp4`) | **2 天 (`age: 2`)** | 推理后立即删除，并以 2 天自动清理规则作为安全兜底。 |
| **`output/`**、**`deliverables/`**、**`trimmed/`** | 粗剪视频、XML/FCPXML 时间线与报告 | **15 天 (`age: 15`)** | 保留 15 天供团队审阅，到期自动清理。 |

---

## 许可证 (License)

本项目采用 [MIT License](LICENSE) 授权。

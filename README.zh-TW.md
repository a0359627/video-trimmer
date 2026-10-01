# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 專案總覽 (Overview)

**Video Trimmer** 是專為單機位口播、教學影片與演講錄影設計的 AI 自動粗剪與去蕪存菁引擎。系統結合 **Google Vertex AI Gemini 3.8 Flash** 原生多模態影片理解、**Whisper 毫秒級逐字聲學時間戳**（Apple Silicon Metal `mlx-whisper` 加速）、**四層統一剪輯架構（4-Layer Unified Architecture）** 以及 **聲學起音鎖定（Acoustic Onset Snapping）**。每次執行會自動剔除 NG 重錄、吃螺絲與無效停頓，同時保護連貫長句不被切碎，並匯出多格式專業剪輯時間軸（`.xml`, `.fcpxml`, `.edl`, `.csv`）與完成粗剪的 MP4 影片。

---

## 四層統一剪輯架構與核心技術特點

1. **第一層：純聲學與標點子句切分 (`transcribe.py`)**：
   - 透過 Whisper 提取逐字時間戳，僅依據物理邊界切分 `Sentence ID`：換氣停頓（`gap >= 0.20s`）、吃螺絲拉長音起音（`word_dur >= 1.20s` 且單字元 `>= 0.45s`）、句尾標點與說話者輪替，同時保留微停頓（`gap < 0.25s`）下的連詞黏合（`CONJUNCTIONS`）。
   - **零 Python 語意猜測原則**：絕不在 Python 端使用字串相似度猜測 NG 重講，語意判斷 100% 交由 LLM 處理。
2. **第二層：雙模式 LLM 語意擇優 (`gemini-3.8-flash`)**：
   - **Mode A：有講稿單調錨定模式（傳入 `--script`）**：自動將講稿切分為 `[Script Block 01] .. [Script Block NN]`，嚴格依序單調對齊，每個講稿段落最多僅保留最後一次完整成功的 Take，徹底杜絕講稿句子重複出現。
   - **Mode B：無講稿意圖視窗仲裁模式（未傳 `--script`）**：以局部語意視窗識別「未完成殘句重講（Abandoned Fragment）」並予以剔除，同時保護「刻意修辭排比強調（如：請訂閱、請訂閱、請訂閱）」不被誤刪。
   - **跨片段首尾防重疊檢查（Tail-to-Head Overlap Check）**：防止相鄰輸出片段出現首尾重複子句。
3. **第三層：子句展開與逐字稿邊界精修 (`resolve_clip_sub_units`)**：
   - 將多句跨度自動展開為獨立的 `Sentence ID` 子單元，自動略過時間區間內未被選中的 NG 句。
   - 當 LLM 在 `transcript` 欄位修剪了句首或句尾贅字時，透過 `_trim_matched_words_by_transcript` 自動將 `t_first` 與 `t_last` 對齊至實際保留字詞的 Whisper 邊界。
4. **第四層：跨片段連貫小句無縫合一 (`coalesce_adjacent_sub_units`)**：
   - 當相鄰片段為連續 `Sentence ID`（中間未跳過任何 NG 句）且物理字間距 `< 0.40s` 時，自動合併為單一連續片段，消除長句內部的無謂跳接（Jump-Cut）與多餘微淡化。
5. **聲學起音鎖定與字尾塞音保護 (`acoustic.py`)**：
   - 將剪輯入點鎖定於聲帶發聲前 80 ms，並依據講者語速（CPS）動態計算緩衝邊界，強制 `true_speech_end >= t_last` 以保護字尾無聲除阻音與鼻音。
6. **15 ms 等功率音訊微淡化與關鍵幀硬體加速渲染 (`render.py`)**：
   - 每個保留片段採用前置 `-ss` / `-to` 關鍵幀快速定位（Fast Input Seeking，免除廢片區段解碼），結合 Apple Silicon `VideoToolbox` 硬體編解碼（`-hwaccel videotoolbox` + `h264_videotoolbox`，具備 `libx264` 自動降級備援）、1 秒關鍵幀間距（`-g 30`）與 15 ms 等功率淡入淡出（`afade=t=in:d=0.015:curve=iqsin` 與 `afade=t=out:d=0.015:curve=oqsin`），消除跳接爆音並大幅提升成片渲染與快轉速度。
7. **多平台 NLE 時間軸匯出 (`exporters.py`)**：
   - 支援匯出 **Final Cut Pro 7 XML**（`.xml`，適用於 Adobe Premiere Pro 與 DaVinci Resolve）、**Apple Final Cut Pro FCPXML**（`.fcpxml`）、**CMX 3600 EDL**（`.edl`）與 **CSV** 剪輯表。

---

## 專案目錄結構（Agent Plugins 1.0 標準規範）

```text
video-trimmer/
├── plugin.json                              # Agent Plugins 1.0 宣告清單
├── rules/
│   └── AGENTS.md                            # 打包於 Plugin 內的客戶端執行期守則（唯讀與 Fail-Fast）
├── skills/
│   └── video-trimmer/                       # 標準技能套件主幹（Single Source of Truth）
│       ├── SKILL.md                         # 技能規範與自動化執行手冊
│       ├── scripts/                         # 核心引擎模組實體目錄 (SSOT)
│       │   ├── __init__.py
│       │   ├── video_trimmer.py             # CLI 解析與四層管線協調器
│       │   ├── acoustic.py                  # CPS 語速計算、聲學起音鎖定與字尾保護
│       │   ├── transcribe.py                # 聲學切句、逐字邊界精修與跨片段無縫合一
│       │   ├── gemini_client.py             # Vertex AI (ADC) 客戶端與雙模式提示詞建構
│       │   ├── gcs_utils.py                 # GCS 暫存、Google Drive 快取與中文檔名修復
│       │   ├── exporters.py                 # FCP7 XML、FCPXML、EDL 與 CSV 時間軸匯出
│       │   └── render.py                    # ffprobe 檢測與 FFmpeg 15ms 微淡化渲染
│       └── prompts/                         # 提示詞規範實體目錄 (SSOT)
│           └── video_cut_prompt.md          # 雙模式語意仲裁與五律減法剪輯規範
├── AGENTS.md                                # 工作區與開發工程規範（Part I 執行守則 & Part II 開發規範）
├── setup.sh                                 # 原生 gcloud 雲端環境一鍵配置腳本
└── tests/                                   # 離線單元測試套件（83 項測試）
```

---

## 安裝與 Google Cloud 環境設定

### 1. 安裝 FFmpeg

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 安裝為 Antigravity Plugin 或本地 CLI

```bash
# 全域 Antigravity Plugin（建議）
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer

# 舊版獨立 Skill 目錄安裝（~/.gemini/config/skills/ 相容方式）
ln -s ~/.gemini/config/plugins/video-trimmer/skills/video-trimmer ~/.gemini/config/skills/video-trimmer

# 安裝 Python 相依套件與 Apple Silicon Metal 加速
pip install -r requirements.txt
pip install mlx-whisper
```

### 3. 一鍵配置 Google Cloud 資源 (`./setup.sh`)

```bash
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## 命令列使用說明 (CLI Usage)

```bash
# Mode B：無講稿自動粗剪（預設採用 Static Multimodal 快速模式）
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4"

# Mode A：搭配拍攝腳本單調錨定對齊（推薦有講稿拍攝使用）
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md"

# 採用緊湊節奏模式（適用於快節奏教學影片）
python3 skills/video-trimmer/scripts/video_trimmer.py -i "sample_take.mp4" --pacing compact --suffix "fast"

# 明確啟用 Agentic 動態影格探索模式
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# 使用已快取的 EDL JSON 本地快速重算與渲染
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --cached-json "raw_footage_static_edl.json"
```

---

## Google Drive 分享連結與 GCS 雙層生命週期規則

| GCS 路徑前綴 (`matchesPrefix`) | 儲存內容 | 保留天數 (`age`) | 規則說明 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 暫存原始影片 (`raw/<filename>.mp4`) | **2 天 (`age: 2`)** | 推論完成後於 `finally` 區塊立即刪除，並以 2 天自動刪除規則作為安全備援。 |
| **`output/`**、**`deliverables/`**、**`trimmed/`** | 粗剪影片、XML/FCPXML 時間軸與 EDL 報告 | **15 天 (`age: 15`)** | 保留 15 天供團隊審閱與下載，期滿自動清理。 |

---

## 單元測試 (Unit Testing)

```bash
python3 -m unittest discover -s tests -v
```

---

## 授權條款 (License)

本專案採用 [MIT License](LICENSE) 授權。

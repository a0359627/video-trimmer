# 單機粗剪調優 Milestone 紀錄與新 Session 交接報告 (國威 V4 成果與工頭堅測試規劃)

**紀錄時間**：2026-10-06  
**當前工作目錄**：`/Users/shengwei/video_tim/single-cam-trimmer`  
**測試素材路徑**：`/Volumes/Crucial X10/singlecam_test_file/`  
**下一階段素材**：`/Volumes/Crucial X10/singlecam_test_file/test_2_kenwork/`

---

## 一、本次 Milestone 核心成果與現狀評估

### 1. 使用者評估反饋
> 「有比較好，但沒有到合格，看起來還是有很多重複的片段藏再中間很難辨識，目前先做個記錄，我要開一個新session 測試工頭堅的影片，先做個 milestone 記錄一下，我接下來會指定 video-timmer 原始資料夾工作」

### 2. 本次 (V4) 明顯改善項目
相較於 V3 版，V4 成功清除了大部分宏觀層面的棄用殘句與開頭口誤：
- **消除棄用假開頭 (Abandoned False Starts)**：
  - V3 Clip 20 (`1738.98s ~ 1740.91s`，「雖然研究強烈支持」) -> **已徹底剔除**。
  - V3 Clip 26 (`1874.52s ~ 1876.32s`，「健康相關他也曾和口腔菌相」) -> **已徹底剔除**。
  - V3 Clip 28 (`1888.77s ~ 1889.76s`，「所以說人類」) -> **已由 Seam Prefix Duplicate 檢測自動剔除**。
- **切除句首吃螺絲 (Opening Stutters)**：
  - Clip 16 原先選入句首贅詞「這些高拷貝變異啊」，V4 成功自動往前修剪，乾淨起音於「但這不是有人被馬鈴薯咬到...」(`1676.74s`)。
- **接縫重複詞消除 (Seam Overlaps)**：
  - Clip 17 尾端重複詞「這就叫」被自動裁切，俐落結束於「讓冷門外掛變成主流配備」，由 Clip 18 乾淨接續「這個就叫軟性選擇掃蕩」。
- **整體時長與片段數縮減**：
  - 從 V3 的 30 個片段（199.88 秒）濃縮為 27 個片段（195.39 秒）。

### 3. 尚未達標（仍有重複藏在中間）的根本原因分析
雖然開頭吃螺絲與孤立殘句被剔除，但「長 Take 內部夾雜之微小重錄或主語殘留」仍未完全過濾：
1. **主講人漏字導致的語意孤立 (如 Clip 23 & 24)**：
   - 劇本台詞：「例如同樣擁有高拷貝數的阿基梅爾·奧德姆人，在面對現代飲食時依然飽受極高的代謝疾病盛行率之苦。」
   - 現場錄影：國威無法順利唸出「阿基梅爾·奧德姆人」，實際說了「例如同樣擁有高拷貝數的」（停頓 0.96s），接著直接跳講「在面對現代飲食時依然飽受極高的代謝疾病盛行率之苦」。
   - 人類剪接師判斷：因主語缺失，直接將「例如同樣擁有高拷貝數的」整句丟棄，只留後半句。
   - AI (Mode A) 判斷：大模型為了滿足劇本前面「例如同樣擁有高拷貝數」，選入了 Clip 23，導致聽感上出現缺乏主語的突兀斷句。
2. **長句中段語氣詞或同義句重覆 (Intra-sentence Hidden Retakes)**：
   - 當講者在一段說話中，未出現大於 1.0 秒之實質物理停頓，而是快速吸氣後換個說法重新開展時，Layer 1 (Whisper) 將其連成同一個長 Segment，若 Layer 2 (LLM) 未精確標記出該微小重複的確切字元區間，下游字級修剪便無法介入。

---

## 二、目前代碼改動與分支狀態

所有調優代碼均在 `/Users/shengwei/video_tim/single-cam-trimmer` 倉庫下維護：
- **`skills/video-trimmer/prompts/video_cut_prompt.md`**：
  - 強化 Last Take Wins 與禁止拼湊假開頭之負向約束規則。
- **`skills/video-trimmer/scripts/transcribe.py`**：
  - `_trim_matched_words_by_transcript`：解鎖字級微調能力，支援首字吃螺絲與句尾 blooper 剔除。
  - `resolve_clip_sub_units` / `coalesce_adjacent_sub_units`：換氣停頓門檻放寬至 0.65s / 0.70s。
  - `trim_cross_clip_seam_overlaps`：
    - 新增獨立的第一階段「Intra-clip Opening Stutter」滑動窗重複開頭裁切。
    - 新增「Seam Prefix Duplicate」剔除機制。
- **`tests/test_transcribe.py`**：
  - 新增 `test_trim_prefix_duplicate_dropped` 與 `test_trim_intra_clip_opening_stutter` 等測試，現有 **92 個單元測試 100% 通過**。

---

## 三、現有輸出檔案清單 (國威 C7232)

所有驗證檔均位於 `/Volumes/Crucial X10/singlecam_test_file/output/`：
- **V4 成果 (供 DaVinci / Premiere 比對)**：
  - `C7232_compatible_v4_edl.xml` (FCP7 XML，23.976 fps)
  - `C7232_compatible_v4_edl.fcpxml` (Final Cut Pro X)
  - `C7232_compatible_v4_edl.json` (結構化 EDL)
  - `C7232_compatible_v4_edl.csv` (表格剪輯表)
- **人工精剪真值 (Human Ground Truth)**：
  - `C7232_compatible_TrimmerCut_humanfix.xml` (人類剪接師精剪版，28 段，183.22 秒)
- **快取檔案 (避免重複轉錄)**：
  - `C7232_compatible_whisper_raw.json`
  - `C7232_compatible_whisper_sentences.json`

---

## 四、新 Session 測試目標：工頭堅影片 (`test_2_kenwork`)

### 1. 素材規格
- **素材目錄**：`/Volumes/Crucial X10/singlecam_test_file/test_2_kenwork/`
- **影片檔案**：`C6619.MP4`
- **原始腳本**：`京都新飯店2026-27_腳本_v1.docx`
- **分鏡剪本**：`(剪本用)2026 京都奢華新旅宿_分鏡版.docx`

### 2. 跨講者泛化驗證重點
- 工頭堅的語速、節奏、換氣長度與國威不同（工頭堅語速相對沉穩、停頓結構不同）。
- 驗證目前調優之規則是否在「不同講者、不同語速」下依然具有泛化能力，避免針對單一講者過度擬合（Overfitting）。
- 檢驗能否在不同語音風格中，更乾淨地過濾長句內部的隱藏重複段落。

# AI 智慧粗剪系統 (Single-Cam Video Trimmer) 接手進度與架構交接報告

**文件版本**：v1.0  
**產生時間**：2026-10-08  
**當前工作目錄**：`/Users/shengwei/video_tim/single-cam-trimmer`  
**主要遠端分支**：`origin/feat/davinci-conform-phoneme-protection` (Fork: `a0359627/video-trimmer`)  
**上游基準倉庫**：`upstream/main` (`sylphlin/video-trimmer`)  
**測試通過率**：98 / 98 測試項 100% 通過 (`python3 -m unittest discover -s tests -v`)

---

## 一、 專案現況與已交付之里程碑 (Current Milestones)

### 1. 達標項目：工頭堅測試素材 (C6619) 已達 A-copy 生產標準
- **素材路徑**：`/Volumes/Crucial X10/singlecam_test_file/test_2_kenwork/C6619.MP4`
- **成果狀態**：V4 版本經人工剪接師實際檢驗，**確認達到無須手動微調、可直接作為 A-copy 交付之水準**。
- **凍結規範**：工頭堅的成功驗證了「單調腳本錨定（Mode A）」與「連續發音保護」的有效性，現有核心邏輯已作為基準線，後續調整不得劣化工頭堅之表現。

### 2. 已修復並標準化之核心技術難題
- **DaVinci Resolve 脫機與循環參照徹底解決**：
  - 實施全域 ID 命名空間隔離（Master File: `{name}_master`，ClipItem: `{name}_v{idx}` 與 `{name}_a{idx}`），徹底消除剪輯軟體 DOM 衝突。
  - 機身時間碼直通（Timecode Pass-through）：XML 僅保留 `<timecode><string>`，移除易引起定格之 `<frame>` 標籤。
- **連續發音保護（Continuous Phonation Invariant）**：
  - 實施 `tail_gap < 0.18s` 連續語流保護，防止專有名詞尾字被削切，並搭配 `BLOOPER_TAIL_WORDS` 剔除真正雜語。
- **全格式交付與自動化技術審計**：
  - 單次執行輸出 `.xml` (FCP7), `.fcpxml` (FCPX), `.edl` (CMX 3600), `.csv`, `.json` 與純文字版 `_edl_report.md`。

---

## 二、 國威測試素材 (C7232) 現存問題與深度診斷

- **測試素材**：`/Volumes/Crucial X10/singlecam_test_file/C7232_compatible.mp4` (語速 5.0 ~ 6.5 CPS)
- **對標真值**：`/Volumes/Crucial X10/singlecam_test_file/output/C7232_compatible_TrimmerCut_humanfix.xml` (人工手剪版)
- **最新輸出**：`/Volumes/Crucial X10/singlecam_test_file/output/C7232_compatible_new_ai_cut_edl.xml`

### 4 大已知問題與技術根因剖析：
1. **句尾現場脫口碎念未剔除 (Clip 2 尾端)**：
   - 語音現況：國威在唸完「高碳水化合物飲食」後，0 毫秒無縫脫口「幹嘛呢」(`1408.54s ~ 1409.58s`)。
   - 根因：大模型將「幹嘛呢」直接寫入 transcript，下游字級對齊將其視為有效台詞，且連續發音保護判定 `tail_gap == 0.0s`，導致保留。
2. **同句內極速連唸口吃重複 (Clip 10 首端)**：
   - 語音現況：國威在 1 秒內同口氣連唸三次「為了追查這誇張的...」(`1554.14s ~ 1555.48s`)。
   - 根因：Whisper 產生零長度重複 Token，字級對齊演算法錨點落在重複詞中間，導致入點前緣夾帶殘留口吃。
3. **句中修辭重複未整理 (Clip 16)**：
   - 語音現況：前半句「這些高拷貝變異啊... 但這不是...」，後半句「這些高拷貝變異啊，早在農業出現以前就已經存在了」。
   - 根因：人類剪接師手剪直接丟棄前半句口吃，只取後半句主幹；AI 則全選，導致出現兩次「這些高拷貝變異啊」。
4. **專有名詞丟失與斷頭句 (Clip 23 & 24)**：
   - 語音現況：劇本要求「例如同樣擁有高拷貝數的阿基梅爾·奧德姆人...」。Take A (`1789s`) 完整唸出；Take B (`1837s`，全片最後嘗試) 國威漏唸部落名，直接跳唸後半句。
   - 根因：現行規則恪守「Last Take Wins」，AI 挑選了最後但語意殘缺的 Take B；人類剪接師則依據語意完整性往前挑選 Take A。

---

## 三、 重大架構決策與已否決方案 (Technical Invariants & Denied Ideas)

### 否決方案：實體視訊放慢後剪輯再加速還原 (Time-Stretching Retiming)
- **評估結論**：經第一性原理與訊號分析，**判定完全無測試價值，嚴禁採用**。
- **否決依據**：
  1. 數學等價性：物理放慢視訊在數學上 100% 等價於直接將程式內的換氣浮點數常數縮小（如 `0.20s * 0.75 = 0.15s`），無需耗費 15~20 分鐘重編碼數十 GB 視訊。
  2. 大模型不受語速影響：Gemini 處理的是離線提取的文字 Token，平行推論，不存在人類耳朵「反應不及」之生理限制。
  3. 無法修復核心病灶：放慢後，漏唸的專有名詞依然漏唸，重複連唸依然重複。
  4. 引入訊號與時碼失真：時域拉伸演算法破壞暫態與相位，造成 Whisper 精度下滑，且時間碼浮點逆運算易引發 1 格捨入誤差（導致爆音或跳格）。

---

## 四、 未來系統架構與功能藍圖 (Architecture Blueprint for Next Agent)

公司內部有 6 位以上主持人，涵蓋「逐字稿照唸（如工頭堅）」與「大綱即興隨講（如國威及其他講者）」。下一階段需落實以下模組：

### 1. 自適應講者畫像架構 (Adaptive Presenter Registry)
- **自動偵測模式 (`presenter: "auto"`)**：
  - 在 Layer 0 轉錄完成時，依據全片統計：
    - `median_cps = total_chars / total_speech_time`
    - `median_pause_gap = percentile(pauses, 50)`
  - 若 `CPS > 4.8` 且換氣短，自動啟動 **`fast_explainer`** 策略。
  - 若 `CPS <= 4.2` 且換氣深，自動啟動 **`scripted_narrative`** 策略。
  - 若無逐字稿只有大綱，啟動 **`outline_improv`** 意圖語義窗聚類。
- **大模型 Prompt 核心演進**：
  - 增加約束：**「實體完整度優先於最後錄製（Completeness Trumps Recency）」**。若後續 Take 丟失關鍵主語或名詞，判定為殘缺，必須回退選取資訊完整之正式 Take。
- **文字層物理去重**：
  - 實施同句內 1.5 秒內相同 N-gram（>=3 字）重複片段自動濾除。

### 2. 爭議與疑慮清單系統 (Human-in-the-Loop Ambiguity System)
剪接師最大痛點為「AI 剪錯後需去 40 分鐘母片大海撈針」。新功能規劃如下：
- **時間軸 V2 備選軌道 (Alternative Track)**：
  - **V1 軌道**：AI 推薦之 A-copy。
  - **V2 軌道 (預設 Disabled / 靜音)**：放置次佳或語意有爭議之備選 Take，時間點與 V1 精確對齊。剪接師在 DaVinci / Premiere 選中片段按 `D` 鍵即可在 1 秒內試聽切換。
- **時間軸彩色標記 (Timeline Markers)**：
  - 紅色：`[語意疑慮] 疑似漏唸專有名詞`。
  - 橘色：`[備選鏡頭] V2 附上流暢版備選`。
  - 黃色：`[即興發揮] 偏離大綱但語意完整`。
- **報告交付**：同步產出 `<basename>_ambiguity_report.md` 爭議審查對照表。

### 3. 部署方案：Google Cloud Run MCP 伺服器 (邊緣-雲端混合模式)
針對剪接師在 Claude Code / Codex / Antigravity 中調用，且無法上傳數十 GB 原始視訊之物理限制：
- **本地輕量 Worker (Local Edge)**：
  - 本機 FFmpeg 瞬時抽取 48kbps Mono MP3 (~15MB)。
  - 本機 Apple Silicon Metal GPU 運行 `mlx-whisper`（免費、2 分鐘完成）。
  - 將 15MB 音檔與 Whisper JSON 上傳至 GCS。
- **雲端 Cloud Run (MCP Brain)**：
  - 接收純文字與輕量音訊，調用 Vertex AI Gemini 3.8 Flash 進行取鏡仲裁與爭議評估。
  - 回傳結構化 EDL JSON。
- **本地 Exporter 完成收尾**：
  - 本地模組將 EDL JSON 自動綁定回剪接師硬碟上的「原始相機大檔實體路徑」與「機身相機時間碼」，生成最終 XML。

### 4. 整合 `sylphlin/subtitle-craft` 字幕系統
- **資料共用**：
  - Video Trimmer 產出的 Whisper 字級快取 (`_words.json`) 直通 Subtitle Craft，字幕不再需要重跑轉錄。
  - Subtitle Craft 產出的專有名詞庫 (`_glossary.md`) 反哺 Video Trimmer，作為關鍵字詞保護白名單。
- **雲端生命週期**：共用 GCS 儲存桶與 2 天自動生命週期過期策略。

---

## 五、 接手 Agent 作業規範與約束 (Operational Rules of Engagement)

接手本專案之 Agent 必須嚴格遵守以下操作紀律：
1. **繁體中文原則**：所有對話、回覆、報告一律使用繁體中文 (`zh-TW`)。
2. **零表情符號原則 (Zero-Emoji Policy)**：技術報告、標題、表格內嚴格禁止使用裝飾性表情符號。
3. **單元測試守門**：任何代碼或規則修改前與修改後，必須執行單元測試並確保 100% 通過（若系統預設 Python 未安裝相關套件，請使用 Anaconda 環境）：
   ```bash
   /opt/anaconda3/bin/python3 -m unittest discover -s tests -v
   ```
4. **Git 作者規範**：提交時必須使用標準人類作者身分，嚴禁任何 AI 助手標籤：
   ```bash
   git commit --author="sylphlin <sylph.lin@gmail.com>" -m "..."
   ```
5. **工頭堅基準保護**：任何針對國威或新主持人的調優，不得破壞工頭堅素材 (`test_2_kenwork/C6619.MP4`) 的現有 A-copy 水準。

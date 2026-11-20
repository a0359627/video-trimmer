"""集中管理的具名常數（取代原本散落於程式碼中的字面值）。"""

# --- 音訊等級判定：改用絕對 dBFS 門檻，取代原本的「底噪倍數」相對邏輯 ---
# 依據：EBU/ITU-R BS.1770 loudness gating 概念與業界工具（如 FireCut）常用範圍，
# 專業配音噪音底噪基準約 -60dBFS 以下，語音峰值約 -20dBFS，
# 取兩者中間值作為語音/靜音判定的絕對門檻（可由 CLI 覆寫）。
SILENCE_THRESHOLD_DBFS = -40.0

# --- 音訊 crossfade 時長 ---
# 依據：業界共識 <5ms 幾乎無效，SoundOnSound 建議 20-50ms 可穩妥避免 click，
# 取建議區間下限，維持原本「緊湊剪輯」風格但比原本 15ms 更保守。
CROSSFADE_DURATION_SEC = 0.020  # 原為 0.015

# --- 以下為經驗值，查證後確認無公開業界標準，僅做具名化管理，數值不變 ---
HANGOVER_STEPS = 7                  # 70ms 靜音平息期（10ms/step）
# 註：原本的 CLAP_* 拍手偵測相關常數已移除，改用 Whisper 句子邊界做結構性限制（見 acoustic.py）

# --- 允許的輸入影片副檔名（不分大小寫） ---
ALLOWED_INPUT_EXTENSIONS = (".mp4", ".mov")

# --- Gemini API 網路呼叫重試參數 ---
# 最多重試 3 次（即最多 4 次嘗試：1 次原始呼叫 + 3 次重試），
# 重試間隔採指數退避 1s -> 2s -> 4s，第 3 次重試（第 4 次嘗試）仍失敗則拋出 GeminiAPIError。
GEMINI_RETRY_ATTEMPTS = 4
GEMINI_RETRY_WAIT_MIN_SEC = 1
GEMINI_RETRY_WAIT_MAX_SEC = 4

# --- Gemini output token limit ---
# Set max_output_tokens to 65536 while preserving native dynamic thinking.
GEMINI_MAX_OUTPUT_TOKENS = 65536



# --- Gemini video input size limit ---
# Vertex AI rejects very large Cloud Storage video files with 400 INVALID_ARGUMENT.
# A 22 GB 4K camera original failed; a 258 MB stream-copy clip of the same file passed.
# Above this size, upload a small analysis proxy with the same timeline instead.
GEMINI_MAX_VIDEO_BYTES = 2 * 1024 ** 3
ANALYSIS_PROXY_HEIGHT = 720
# Maximum allowed duration difference between source and proxy (truncated read guard).
ANALYSIS_PROXY_MAX_DURATION_DIFF_SEC = 1.0

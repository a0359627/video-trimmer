"""聲學分析：語速計算、動態留白、文字剪輯時間鎖定與邊界收放。"""

import logging
import re

import numpy as np

from .constants import HANGOVER_STEPS, SILENCE_THRESHOLD_DBFS

logger = logging.getLogger(__name__)

# 由 SILENCE_THRESHOLD_DBFS 換算而來的線性振幅絕對門檻（-40dBFS ≈ 0.01）。
_SILENCE_THRESHOLD_LINEAR = 10 ** (SILENCE_THRESHOLD_DBFS / 20)

# speech_thresh 與 silence_thresh 維持原本相對邏輯下的比例關係（3.5 : 1.8），
# 僅將基準由「局部底噪倍數」改為「絕對 dBFS 門檻」，以保留原本的遲滯（hysteresis）
# 判定行為（speech_thresh 較高、silence_thresh 較低，避免邊界抖動）。
_SPEECH_SILENCE_RATIO = 3.5 / 1.8


def calculate_clip_cps(transcript, duration):
    """計算片段口播語速 CPS (Characters/Syllables Per Second)"""
    if not transcript or duration <= 0:
        return 3.0, 0
    zh = len(re.findall(r'[一-鿿]', transcript))
    en_chars = len(re.findall(r'[a-zA-Z0-9]', transcript))
    syllables = zh + int(en_chars * 0.6)
    if syllables == 0:
        syllables = len(re.sub(r'\s+', '', transcript))
    dur = max(0.5, duration)
    return round(syllables / dur, 2), syllables


def refine_speech_bounds_locked(
    audio,
    sr,
    t_first,
    t_last,
    total_dur,
    transcript="",
    pacing="auto",
    prev_sentence_end=None,
    next_sentence_start=None,
):
    """
    文字剪輯時間鎖定與自適應聲學包絡標準 (Text-Based Locked Time & Acoustic Envelope Standard):
    1. 雙向聲學追蹤 (Bidirectional Acoustic Envelope Tracking)：
       - 以 Whisper 發音時間為錨點，由局部環境底噪動態判定語音與靜音閾值。
       - 句尾向前追蹤：若講者仍在發音（如外語弱音或鼻音），自動順延至能量真正落入底噪（徹底杜絕腰斬斷句）。
       - 句尾向後回縮：若講者已提前閉口，自動回縮多餘的無聲死寂（徹底杜絕突兀空白）。
       - 句首微向後檢查：保護起首輔音/爆破音之微觀氣息。
    2. 雙向 Whisper 句子邊界上限（結構性防護，取代拍手偵測）：
       - 句尾向後追蹤 true_speech_end 時，搜尋範圍不可跨越下一個 Whisper 句子的起點 (next_sentence_start)。
       - 句首向前追蹤 true_speech_start 時，搜尋範圍不可跨越上一個 Whisper 句子的結尾 (prev_sentence_end)。
       拍手、打板等雜音天然落在句子之間的空白區，把聲學搜尋範圍鎖在相鄰句子邊界之內，
       等於自動避開這些雜音，且不需要判斷「這是什麼聲音」。
    3. 自適應動態定格留白 (Dynamic Settle Buffer)。
    """
    # 1. 計算片段語速 (CPS)
    raw_dur = max(0.5, t_last - t_first)
    cps, _ = calculate_clip_cps(transcript, raw_dur)

    # 2. 局部環境底噪測繪 (Local Noise Floor Estimation)
    hop = int(0.01 * sr)  # 10ms 採樣
    win = int(0.03 * sr)  # 30ms 視窗

    idx_center = int(t_last * sr)
    idx_start = max(0, idx_center - int(1.0 * sr))
    idx_end = min(len(audio), idx_center + int(1.2 * sr))

    all_rms = [np.sqrt(np.mean(audio[i : i + win] ** 2)) for i in range(idx_start, idx_end - win, hop)]
    noise_floor = np.percentile(all_rms, 10) if all_rms else _SILENCE_THRESHOLD_LINEAR * 0.15

    # 絕對 dBFS 門檻優先；局部底噪估算僅作為輔助校正 —
    # 當環境噪音本身已高於絕對門檻時（例如吵雜環境錄音），改以底噪倍數 fallback，
    # 避免固定的絕對門檻在吵雜環境下失效。
    silence_thresh = max(_SILENCE_THRESHOLD_LINEAR, noise_floor * 1.8)
    speech_thresh = max(_SILENCE_THRESHOLD_LINEAR * _SPEECH_SILENCE_RATIO, noise_floor * 3.5)

    # 3. 句尾聲學邊界前向追蹤 (Acoustic Out-point Forward Tracking Only)
    # 核心防護準則：Whisper 字級標記點 t_last 已經是該字結束點，嚴禁向後倒退切入字詞內部！
    # 一律以 t_last 為絕對下限向後前向追蹤聲帶殘響自然衰減至底噪，徹底杜絕句末字 (如清塞音、輕聲詞) 被截斷。
    silent_count = 0
    true_speech_end = t_last

    search_end_limit = min(total_dur - 0.05, t_last + 1.2)
    if next_sentence_start is not None:
        search_end_limit = min(search_end_limit, next_sentence_start)

    f_steps = np.arange(t_last, max(t_last, search_end_limit), 0.01)
    for t in f_steps:
        idx = int(t * sr)
        if idx + win >= len(audio):
            break
        r = np.sqrt(np.mean(audio[idx : idx + win] ** 2))
        peak = np.max(np.abs(audio[idx : idx + win]))

        # 防干擾：先出現靜音間隔，隨後又突然爆發外人說話/拍手脈衝/導演喊卡
        if silent_count >= 5 and (r > speech_thresh * 2 or peak > 0.30):
            true_speech_end = max(t_last, t - (silent_count * 0.01) + 0.02)
            break

        if r < silence_thresh:
            silent_count += 1
            if silent_count >= HANGOVER_STEPS:
                true_speech_end = max(t_last, t - (HANGOVER_STEPS * 0.01) + 0.03)
                break
        else:
            silent_count = 0
            true_speech_end = t

    if next_sentence_start is not None:
        true_speech_end = min(true_speech_end, next_sentence_start)

    # 4. 依據 CPS 計算自然呼吸氣息與定格留白
    if pacing == "compact" or cps >= 5.5:
        settle_out = 0.06
        breath_in = 0.06
    elif pacing == "breathing" or cps <= 3.0:
        settle_out = 0.16
        breath_in = 0.14
    else:
        alpha = (cps - 3.0) / (5.5 - 3.0)
        settle_out = 0.16 - alpha * (0.16 - 0.06)
        breath_in = 0.14 - alpha * (0.14 - 0.06)

    final_out = min(total_dur, round(true_speech_end + settle_out, 2))
    if next_sentence_start is not None:
        final_out = min(final_out, next_sentence_start)

    # 5. 句首起音聲學追蹤 (Acoustic In-point Snapping & Dead Air Trimming)
    in_idx = int(t_first * sr)
    in_rms = np.sqrt(np.mean(audio[in_idx : in_idx + win] ** 2)) if in_idx + win < len(audio) else 0
    true_speech_start = t_first

    if in_rms < speech_thresh:
        # Whisper 標記點落在開拍前的死寂發呆空白中 -> 向前掃描尋找真正聲帶起音點 (徹底消除前置空白)
        search_forward_e = min(t_last - 0.2, t_first + 2.5)
        for t in np.arange(t_first, search_forward_e, 0.01):
            idx = int(t * sr)
            if idx + win >= len(audio):
                break
            w30 = int(0.03 * sr)
            r = np.sqrt(np.mean(audio[idx : idx + w30] ** 2))
            if r >= speech_thresh:
                true_speech_start = t
                break
    else:
        # 講者在 t_first 處已在發音 -> 向前尋找微觀起音輔音 (最多回溯 150ms，保護微弱輔音氣息)，
        # 但搜尋範圍不可跨越上一個 Whisper 句子的結尾 (prev_sentence_end)。
        search_back_limit = max(0.0, t_first - 0.15)
        if prev_sentence_end is not None:
            search_back_limit = max(search_back_limit, prev_sentence_end)
        for t in np.arange(t_first, search_back_limit, -0.01):
            idx = int(t * sr)
            if idx < 0:
                break
            r = np.sqrt(np.mean(audio[idx : idx + win] ** 2))
            if r < silence_thresh:
                true_speech_start = t + 0.01
                break
            true_speech_start = t


    # 6. 前置呼吸氣息邊界約束：若與前一句結束點間距極短（如急促重講），動態收縮預留邊距，嚴防踩穿前句尾音
    if prev_sentence_end is not None and true_speech_start > prev_sentence_end:
        avail_silence = true_speech_start - prev_sentence_end
        effective_breath_in = min(breath_in, max(0.01, avail_silence * 0.45))
        target_in = true_speech_start - effective_breath_in
    else:
        target_in = max(0.0, true_speech_start - breath_in)
        if prev_sentence_end is not None:
            target_in = max(target_in, prev_sentence_end)
    final_in = round(target_in, 2)

    in_m = round(t_first - final_in, 2)
    out_m = round(final_out - t_last, 2)

    return final_in, final_out, cps, in_m, out_m

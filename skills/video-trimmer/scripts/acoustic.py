"""Acoustic analysis: CPS speech rate calculation, onset snapping, and boundary refinement."""

import logging
import re

import numpy as np

from .constants import HANGOVER_STEPS, SILENCE_THRESHOLD_DBFS

logger = logging.getLogger(__name__)

# Linear amplitude threshold converted from SILENCE_THRESHOLD_DBFS (-40 dBFS ~= 0.01).
_SILENCE_THRESHOLD_LINEAR = 10 ** (SILENCE_THRESHOLD_DBFS / 20)

# Maintain hysteresis ratio (3.5 : 1.8) between speech onset and silence decay thresholds.
_SPEECH_SILENCE_RATIO = 3.5 / 1.8


def calculate_clip_cps(transcript, duration):
    """Calculate presenter speaking rate in Characters/Syllables Per Second (CPS)."""
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
    Refine clip cut-in and cut-out points around Whisper word timestamps.

    ASD-STE100:
    1. Track the local noise floor and forward speech decay from t_last without truncating final phonemes.
    2. Clamp forward and backward acoustic searches within adjacent Whisper sentence boundaries
       (prev_sentence_end and next_sentence_start) to avoid capturing inter-sentence noise.
    3. Compute dynamic lead-in and lead-out margins from presenter CPS and pacing mode.
    """
    # 1. Calculate presenter CPS
    raw_dur = max(0.5, t_last - t_first)
    cps, _ = calculate_clip_cps(transcript, raw_dur)

    # 2. Estimate local ambient noise floor (10ms hop, 30ms window)
    hop = int(0.01 * sr)
    win = int(0.03 * sr)

    idx_center = int(t_last * sr)
    idx_start = max(0, idx_center - int(1.0 * sr))
    idx_end = min(len(audio), idx_center + int(1.2 * sr))

    all_rms = [np.sqrt(np.mean(audio[i : i + win] ** 2)) for i in range(idx_start, idx_end - win, hop)]
    noise_floor = np.percentile(all_rms, 10) if all_rms else _SILENCE_THRESHOLD_LINEAR * 0.15

    # Combine absolute dBFS threshold with local noise floor fallback for noisy environments
    silence_thresh = max(_SILENCE_THRESHOLD_LINEAR, noise_floor * 1.8)
    speech_thresh = max(_SILENCE_THRESHOLD_LINEAR * _SPEECH_SILENCE_RATIO, noise_floor * 3.5)

    # 3. Track out-point forward from t_last (never cut earlier than t_last)
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

        # Stop before sudden off-screen impulse or crew voice after a short silence gap
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

    # 4. Calculate dynamic lead-in and settle-out buffers from presenter CPS
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

    # 5. Snap cut-in point to vocal onset and trim pre-speech dead air
    in_idx = int(t_first * sr)
    in_rms = np.sqrt(np.mean(audio[in_idx : in_idx + win] ** 2)) if in_idx + win < len(audio) else 0
    true_speech_start = t_first

    if in_rms < speech_thresh:
        # Whisper start fell inside pre-speech silence; scan forward (up to 2.5s) to true vocal onset
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
        # Speaker is already phonating at t_first; scan backward up to 150ms without crossing prev_sentence_end
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

    # 6. Constrain the lead-in breath to the silence after the previous sentence (rapid restarts):
    #    shrink the margin to 45% of the available gap so the cut never enters the previous tail
    if prev_sentence_end is not None and true_speech_start > prev_sentence_end:
        avail_silence = true_speech_start - prev_sentence_end
        effective_breath_in = min(breath_in, max(0.01, avail_silence * 0.45))
        target_in = true_speech_start - effective_breath_in
    else:
        target_in = max(0.0, true_speech_start - breath_in)
        if prev_sentence_end is not None:
            target_in = max(target_in, prev_sentence_end)
    # 7. Protect the first word: if the onset search moved the cut-in more than 0.30s past the Whisper
    #    first-word start (soft first syllable or high noise-floor estimate), fall back to just before the
    #    word start. Extra breath is better than a lost first word. Never cross the previous sentence end.
    if target_in > t_first + 0.30:
        target_in = max(0.0, t_first - 0.06)
        if prev_sentence_end is not None:
            target_in = max(target_in, prev_sentence_end)
    final_in = round(target_in, 2)

    in_m = round(t_first - final_in, 2)
    out_m = round(final_out - t_last, 2)

    return final_in, final_out, cps, in_m, out_m

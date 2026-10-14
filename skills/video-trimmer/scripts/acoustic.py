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

# Background-relative threshold for "connected sound": a frame more than 8 dB above the local
# background (10th-percentile RMS) is sound; quieter frames are background (room noise, light breath).
_GAP_OVER_BACKGROUND = 10 ** (8.0 / 20)
# Lowest gap threshold (-60 dBFS), so digital silence does not make room noise count as sound.
_GAP_THRESHOLD_FLOOR_LINEAR = 10 ** (-60.0 / 20)
# A silence gap is at least this long (shorter dips are stop closures inside speech).
MIN_SILENCE_GAP_SEC = 0.08

# Whisper word timestamps can run early (about 0.2 s, up to 0.3 s; the next sentence often starts exactly
# at t_last and takes this word's tail). The out-point tracking can pass the Whisper next-sentence start by
# up to this value (tail plus the 70 ms silence hangover) when the speech then decays into silence.
NEXT_START_SLACK_SEC = 0.40
# Time kept between a cut-out and the onset of the next sound (longer than the 20 ms micro-fade).
NEXT_ONSET_GUARD_SEC = 0.03

# Residual check at t_first: the sound at t_first started before the previous Whisper sentence ended,
# stops within this time after t_first, and a silence of at least RESIDUAL_GAP_SEC follows before the onset.
RESIDUAL_MAX_AFTER_T_FIRST_SEC = 0.15
RESIDUAL_GAP_SEC = 0.15


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


def _rms_at(audio, sr, t, win):
    """RMS of the window that starts at t, or None outside the audio."""
    idx = int(t * sr)
    if idx < 0 or idx + win >= len(audio):
        return None
    return float(np.sqrt(np.mean(audio[idx : idx + win] ** 2)))


def _percentile_rms(audio, sr, t0, t1, pct=10):
    """Percentile RMS (10 ms hop, 30 ms window) of [t0, t1], or None for an empty span."""
    hop = int(0.01 * sr)
    win = int(0.03 * sr)
    i0 = max(0, int(t0 * sr))
    i1 = min(len(audio), int(t1 * sr))
    vals = [np.sqrt(np.mean(audio[i : i + win] ** 2)) for i in range(i0, i1 - win, hop)]
    return float(np.percentile(vals, pct)) if vals else None


def _boundary_thresholds(audio, sr, t_center):
    """Return (noise_floor, silence_thresh, speech_thresh) from the noise floor around t_center (-1.0 s / +1.2 s)."""
    noise_floor = _percentile_rms(audio, sr, t_center - 1.0, t_center + 1.2)
    if noise_floor is None:
        noise_floor = _SILENCE_THRESHOLD_LINEAR * 0.15
    # Combine absolute dBFS threshold with local noise floor fallback for noisy environments
    silence_thresh = max(_SILENCE_THRESHOLD_LINEAR, noise_floor * 1.8)
    speech_thresh = max(_SILENCE_THRESHOLD_LINEAR * _SPEECH_SILENCE_RATIO, noise_floor * 3.5)
    return noise_floor, silence_thresh, speech_thresh


def _gap_threshold(background, silence_thresh):
    """Threshold between background and connected sound: 8 dB over the background, never above silence_thresh."""
    return min(silence_thresh, max(background * _GAP_OVER_BACKGROUND, _GAP_THRESHOLD_FLOOR_LINEAR))


def _connected_sound_start(audio, sr, t_onset, t_stop, gap_thresh, min_gap=MIN_SILENCE_GAP_SEC):
    """
    Walk back from t_onset and return (start, gap_found) for the sound connected to the onset.

    ASD-STE100:
    The sound run continues while frames are above gap_thresh. Dips shorter than min_gap do not stop it.
    The run stops at the first silence gap of min_gap or more: start = first sound frame after that gap,
    gap_found = True. When no such gap occurs before t_stop, gap_found = False (the sound before the onset
    is connected to it down to t_stop).
    """
    win = int(0.03 * sr)
    gap_frames = int(round(min_gap / 0.01))
    start = t_onset
    quiet = 0
    t = t_onset - 0.01
    while t >= t_stop - 1e-9:
        r = _rms_at(audio, sr, t, win)
        if r is None:
            break
        if r < gap_thresh:
            quiet += 1
            if quiet >= gap_frames:
                return round(start, 2), True
        else:
            quiet = 0
            start = t
        t -= 0.01
    return round(start, 2), False


def find_next_sound_onset(audio, sr, t_from, t_until, silence_thresh=None, speech_thresh=None, gap_thresh=None,
                          detect_thresh=None):
    """
    Return the start of the next sound after t_from, or None when no sound starts before t_until.

    ASD-STE100:
    1. Scan forward from t_from to the first frame at or above detect_thresh (default: speech_thresh).
    2. Walk back from that frame through the sound connected to it (soft consonant, inhale) and
       return the start of that sound. The result is never earlier than t_from.
    Thresholds default to the local noise floor around t_from. Use the result minus
    NEXT_ONSET_GUARD_SEC as the latest cut-out before the next sound.
    """
    if silence_thresh is None or speech_thresh is None:
        noise_floor, silence_thresh, speech_thresh = _boundary_thresholds(audio, sr, t_from)
        if gap_thresh is None:
            gap_thresh = _gap_threshold(noise_floor, silence_thresh)
    if gap_thresh is None:
        gap_thresh = silence_thresh
    if detect_thresh is None:
        detect_thresh = speech_thresh
    win = int(0.03 * sr)
    for t in np.arange(t_from, max(t_from, t_until), 0.01):
        r = _rms_at(audio, sr, t, win)
        if r is None:
            return None
        if r >= detect_thresh:
            start, _ = _connected_sound_start(audio, sr, float(t), t_from, gap_thresh)
            return max(t_from, start)
    return None


def _other_mic_speech_before(other_audio, sr, t0, t1, min_speech=0.20):
    """True when another microphone has at least min_speech seconds of voiced frames (above its silence threshold) in [t0, t1]."""
    win = int(0.03 * sr)
    for other in other_audio:
        _, other_silence, _ = _boundary_thresholds(other, sr, t0)
        frames = 0
        for t in np.arange(t0, t1, 0.01):
            r = _rms_at(other, sr, t, win)
            if r is not None and r >= other_silence:
                frames += 1
        if frames * 0.01 >= min_speech - 1e-9:
            return True
    return False


def _onset_after_residual(audio, sr, t_first, prev_sentence_end, search_end, silence_thresh, speech_thresh,
                          gap_thresh):
    """
    Detect a residual of earlier audio at t_first and return (residual_end, onset), or None.

    ASD-STE100:
    Whisper often starts the first word early: at the last syllable of the previous take, with the word
    stretched over the following silence. The sound at t_first is then a residual, not the first word.
    The sound at t_first is a residual when all of these are true:
    1. It started at or before the end of the previous Whisper sentence (it continues earlier audio).
    2. It stops within RESIDUAL_MAX_AFTER_T_FIRST_SEC after t_first.
    3. A silence of at least RESIDUAL_GAP_SEC follows, and then speech starts before search_end.
    """
    if prev_sentence_end is None:
        return None
    win = int(0.03 * sr)

    # 1. Start of the sound run that contains t_first (look back up to 0.5 s).
    run_start = t_first
    for t in np.arange(t_first - 0.01, t_first - 0.5, -0.01):
        r = _rms_at(audio, sr, t, win)
        if r is None or r < silence_thresh:
            break
        run_start = t
    if run_start > prev_sentence_end + 0.02:
        return None

    # 2. End of that sound run after t_first.
    run_end = None
    for t in np.arange(t_first, t_first + RESIDUAL_MAX_AFTER_T_FIRST_SEC + 0.01, 0.01):
        r = _rms_at(audio, sr, t, win)
        if r is None:
            return None
        if r < silence_thresh:
            run_end = float(t)
            break
    if run_end is None:
        return None

    # 3. Next speech onset, with a long enough silence before it.
    onset = None
    longest_quiet = 0
    quiet = 0
    for t in np.arange(run_end, search_end, 0.01):
        r = _rms_at(audio, sr, t, win)
        if r is None:
            break
        if r >= speech_thresh:
            onset = float(t)
            break
        if r < gap_thresh:
            quiet += 1
            longest_quiet = max(longest_quiet, quiet)
        else:
            quiet = 0
    if onset is None or longest_quiet * 0.01 < RESIDUAL_GAP_SEC - 1e-9:
        return None
    return round(run_end, 2), onset


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
    tail_pad=0.0,
    other_audio=None,
):
    """
    Refine clip cut-in and cut-out points around Whisper word timestamps.

    ASD-STE100:
    1. Track the local noise floor and forward speech decay from t_last without truncating final phonemes.
    2. The cut-out never enters the next sound: the limit is the acoustic onset of the next sound minus
       NEXT_ONSET_GUARD_SEC. The Whisper next_sentence_start is a reference only. The tracking can pass it
       by NEXT_START_SLACK_SEC (Whisper times run early). When the speech continues past that point with no
       silence, the cut-out falls back to next_sentence_start.
    3. The cut-in snaps to the vocal onset of the first word. It falls back to just before t_first only when
       sound connected to the onset reaches back to t_first (soft first syllable). When the sound at t_first
       is a residual of the previous take (separated from the onset by silence), the cut-in skips it.
       The cut-in never crosses prev_sentence_end.
    4. Compute dynamic lead-in and lead-out margins from presenter CPS and pacing mode.
    5. tail_pad (Show Mode) adds up to tail_pad seconds after the cut-out, with the same next-sound limit.
       The returned out-margin does not include the pad.
    6. other_audio (Show Mode): signals of the other microphones. The next-sound limit also checks them,
       so the cut-out does not enter the other speaker's next line.
    """
    # 1. Calculate presenter CPS
    raw_dur = max(0.5, t_last - t_first)
    cps, _ = calculate_clip_cps(transcript, raw_dur)

    # 2. Estimate local ambient noise floor (10ms hop, 30ms window) around t_last
    win = int(0.03 * sr)
    noise_floor, silence_thresh, speech_thresh = _boundary_thresholds(audio, sr, t_last)
    gap_thresh_out = _gap_threshold(noise_floor, silence_thresh)

    # The acoustic limits need thresholds that separate this speaker from the background. When the
    # speech level near t_last is not above silence_thresh (noise floor estimated from dense speech),
    # keep the Whisper next_sentence_start as the hard out-point limit.
    speech_level = _percentile_rms(audio, sr, max(t_first, t_last - 1.0), t_last, pct=75)
    acoustic_limits = speech_level is not None and speech_level > silence_thresh

    def next_sound_onset(t_from, t_until):
        # Any sound above the silence threshold on this microphone, or speech on another microphone
        onsets = [find_next_sound_onset(audio, sr, t_from, t_until, silence_thresh=silence_thresh,
                                        speech_thresh=speech_thresh, gap_thresh=gap_thresh_out,
                                        detect_thresh=silence_thresh)]
        for other in other_audio or []:
            onsets.append(find_next_sound_onset(other, sr, t_from, t_until))
        onsets = [o for o in onsets if o is not None]
        return min(onsets) if onsets else None

    # 3. Track out-point forward from t_last (never cut earlier than t_last)
    silent_count = 0
    true_speech_end = t_last
    boundary_found = False  # True when silence (or a new impulse after a short silence) ends the tracking
    impulse_onset = None

    search_end_limit = min(total_dur - 0.05, t_last + 1.2)
    if next_sentence_start is not None:
        slack = NEXT_START_SLACK_SEC if acoustic_limits else 0.0
        search_end_limit = min(search_end_limit, max(t_last, float(next_sentence_start)) + slack)

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
            boundary_found = True
            impulse_onset = float(t)
            break

        if r < silence_thresh:
            silent_count += 1
            if silent_count >= HANGOVER_STEPS:
                true_speech_end = max(t_last, t - (HANGOVER_STEPS * 0.01) + 0.03)
                boundary_found = True
                break
        else:
            silent_count = 0
            true_speech_end = t

    # 4. Limit of the cut-out: the next sound onset (acoustic); Whisper next_sentence_start only when
    #    no silence separates this speech from the next sentence (continuous phonation).
    out_limit = None
    if not acoustic_limits:
        if next_sentence_start is not None:
            true_speech_end = min(true_speech_end, next_sentence_start)
            out_limit = float(next_sentence_start)
    elif boundary_found:
        if impulse_onset is not None:
            out_limit = impulse_onset - NEXT_ONSET_GUARD_SEC
        else:
            nxt = next_sound_onset(true_speech_end, min(total_dur, true_speech_end + 0.6))
            if nxt is not None:
                out_limit = nxt - NEXT_ONSET_GUARD_SEC
    elif next_sentence_start is not None:
        true_speech_end = min(true_speech_end, next_sentence_start)
        out_limit = float(next_sentence_start)

    # 5. Calculate dynamic lead-in and settle-out buffers from presenter CPS
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
    if out_limit is not None:
        # Never cut before the tracked end of speech (or t_last) to make room for the guard
        final_out = min(final_out, max(t_last, true_speech_end, round(out_limit, 2)))
        if not acoustic_limits:
            final_out = min(final_out, out_limit)

    # 6. Snap cut-in point to vocal onset and trim pre-speech dead air
    in_rms = _rms_at(audio, sr, t_first, win) or 0.0
    true_speech_start = t_first
    prev_bound = prev_sentence_end
    search_forward_e = min(t_last - 0.2, t_first + 2.5)
    # Background near the first word (the noise floor above comes from the clip end)
    local_bg = _percentile_rms(audio, sr, t_first - 1.0, min(t_last, t_first + 1.5))
    background = noise_floor if local_bg is None else min(noise_floor, local_bg)
    gap_thresh_in = _gap_threshold(background, silence_thresh)

    forward_onset = None
    if in_rms < speech_thresh:
        # Whisper start fell inside pre-speech silence (or a soft syllable); scan forward (up to 2.5s) to the onset
        for t in np.arange(t_first, search_forward_e, 0.01):
            r = _rms_at(audio, sr, t, win)
            if r is None:
                break
            if r >= speech_thresh:
                forward_onset = float(t)
                break
    else:
        residual = _onset_after_residual(audio, sr, t_first, prev_sentence_end, search_forward_e,
                                         silence_thresh, speech_thresh, gap_thresh_in)
        if residual is not None:
            # The sound at t_first is the tail of the previous take: skip it and the silence after it
            residual_end, forward_onset = residual
            prev_bound = residual_end if prev_bound is None else max(prev_bound, residual_end)
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

    if forward_onset is not None:
        true_speech_start = forward_onset

    # 7. Constrain the lead-in breath to the silence after the previous sentence (rapid restarts):
    #    shrink the margin to 45% of the available gap so the cut never enters the previous tail
    if prev_bound is not None and true_speech_start > prev_bound:
        avail_silence = true_speech_start - prev_bound
        effective_breath_in = min(breath_in, max(0.01, avail_silence * 0.45))
        target_in = true_speech_start - effective_breath_in
    else:
        target_in = max(0.0, true_speech_start - breath_in)
        if prev_bound is not None:
            target_in = max(target_in, prev_bound)
    # 8. Protect the first word: the onset search moved the cut-in more than 0.30s past the Whisper first-word
    #    start. Fall back to just before the word start only when sound connected to the onset reaches back to
    #    t_first (soft first syllable, or a noise floor estimate at the clip end that is too high for the first
    #    word). The connection test uses the background near t_first. When a silence gap separates the onset
    #    from t_first (Whisper set the first word early, or the interval holds only a residual of the previous
    #    take), keep the acoustic onset.
    #    Show Mode: speech on another microphone that lasts past the Whisper bias window (0.15s after t_first)
    #    means the first words of this sentence were spoken there (the transcript keeps them); keep them too.
    if forward_onset is not None and target_in > t_first + 0.30:
        _, gap_found = _connected_sound_start(audio, sr, forward_onset, t_first, gap_thresh_in)
        if gap_found and other_audio:
            gap_found = not _other_mic_speech_before(other_audio, sr, t_first + RESIDUAL_MAX_AFTER_T_FIRST_SEC,
                                                      forward_onset)
        if not gap_found:
            target_in = max(0.0, t_first - 0.06)
            if prev_bound is not None:
                target_in = max(target_in, prev_bound)
    final_in = round(target_in, 2)

    in_m = round(t_first - final_in, 2)
    out_m = round(final_out - t_last, 2)

    # 9. Show Mode tail pad: let the thought land, but never enter the next sound
    if tail_pad and tail_pad > 0.0:
        pad_limit = total_dur
        if acoustic_limits:
            nxt = next_sound_onset(final_out, min(total_dur, final_out + tail_pad + 0.5))
            if nxt is not None:
                pad_limit = min(pad_limit, nxt - NEXT_ONSET_GUARD_SEC)
        elif next_sentence_start is not None:
            pad_limit = min(pad_limit, float(next_sentence_start) - 0.02)
        final_out = round(max(final_out, min(pad_limit, final_out + tail_pad)), 2)

    return final_in, final_out, cps, in_m, out_m

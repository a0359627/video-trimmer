"""Mic-side speaker attribution for two-microphone shoots (Show Mode).

ASD-STE100:
Many studio shoots record the host on one channel (lavalier) and the director or
off-screen voice role on the other channel (camera or boom microphone).
This module measures the level of each Whisper word on both channels and assigns
one of three roles:

- HOST:      the on-camera host (host channel is louder).
- OFFSCREEN: the director or voice-role performer (off-screen channel is louder).
- CUE:       countdown digits spoken before a take (never part of the program).

The roles are written to each word (`speaker_id`, `is_target_speaker`, `mic_role`,
`mic_diff_db`) and raw segments are split at role changes so that each merged
sentence contains one speaker only. Only CUE words set `is_target_speaker=False`,
because the rest of the pipeline treats that flag as "never use this audio".
OFFSCREEN words keep `is_target_speaker=True`; the Gemini prompt decides whether
each off-screen sentence is part of the show or a direction to drop.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROLE_HOST = "HOST"
ROLE_OFFSCREEN = "OFFSCREEN"
ROLE_CUE = "CUE"
ROLE_VALUES = (ROLE_HOST, ROLE_OFFSCREEN, ROLE_CUE)

VALID_OFFSCREEN_CHANNELS = ("none", "auto", "left", "right")

# Countdown tokens that may appear as separate Whisper words ("5", "4", "五", "四", ...).
_COUNTDOWN_TOKENS = {
    "5", "4", "3", "2", "1", "0",
    "５", "４", "３", "２", "１", "０",
    "五", "四", "三", "二", "一", "零",
}
_CUE_WORDS = {"action", "rolling", "cut", "standby", "卡", "開始", "好開始"}
_NON_WORD_RE = re.compile(r"[^\w一-鿿]")


def _norm_token(word: str) -> str:
    return _NON_WORD_RE.sub("", word or "").lower()


def extract_stereo_wav(video_path: Path | str, out_wav: Path | str, sample_rate: int = 16000) -> Path:
    """Extract a two-channel PCM WAV from the video with FFmpeg."""
    out_wav = Path(out_wav)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "2", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(out_wav)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"無法從 {video_path} 抽出立體聲音軌（需要 FFmpeg 與雙聲道音訊）")
    return out_wav


def load_stereo_audio(wav_path: Path | str):
    """Load a WAV file as float32 with shape (N, channels)."""
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    return audio, sr


def _dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-9)))


def detect_offscreen_channel(audio: np.ndarray) -> str:
    """Return "left" or "right": the quieter channel over the whole recording is the off-screen mic."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        raise ValueError("需要雙聲道音訊才能自動判定場外麥克風聲道")
    rms_l = _dbfs(np.sqrt(np.mean(audio[:, 0] ** 2)))
    rms_r = _dbfs(np.sqrt(np.mean(audio[:, 1] ** 2)))
    return "left" if rms_l < rms_r else "right"


def word_channel_levels(audio: np.ndarray, sr: int, start: float, end: float) -> tuple[float, float]:
    """RMS level in dBFS of (left, right) for one time span; spans shorter than 50 ms are padded."""
    n = audio.shape[0]
    i0 = max(0, min(n - 1, int(round(start * sr))))
    i1 = max(i0 + int(0.05 * sr), min(n, int(round(end * sr))))
    seg = audio[i0:i1]
    if seg.shape[0] == 0:
        return -120.0, -120.0
    return _dbfs(np.sqrt(np.mean(seg[:, 0] ** 2))), _dbfs(np.sqrt(np.mean(seg[:, 1] ** 2)))


def attribute_word_roles(
    segments: list[dict],
    audio: np.ndarray,
    sr: int,
    offscreen_channel: str = "left",
    margin_db: float = -3.0,
) -> list[dict]:
    """Assign HOST / OFFSCREEN to every word by comparing the two microphone channels.

    A word is OFFSCREEN when (offscreen_level - host_level) >= margin_db. The host
    channel is normally 10 to 20 dB louder for the host, so a margin of -3 dB keeps
    quiet host words on the host side while catching the off-screen voice.
    Words without usable timestamps inherit the role of the previous word.
    """
    if offscreen_channel not in ("left", "right"):
        raise ValueError(f"offscreen_channel 必須是 left 或 right，收到 {offscreen_channel!r}")
    off_idx = 0 if offscreen_channel == "left" else 1
    host_idx = 1 - off_idx

    out = []
    for seg in segments:
        seg_copy = dict(seg)
        words = []
        prev_role = ROLE_HOST
        for w in seg.get("words", []) or []:
            w2 = dict(w)
            try:
                ws, we = float(w2.get("start", 0.0)), float(w2.get("end", 0.0))
            except (TypeError, ValueError):
                ws, we = 0.0, 0.0
            if we - ws >= 0.02:
                levels = word_channel_levels(audio, sr, ws, we)
                diff = levels[off_idx] - levels[host_idx]
                role = ROLE_OFFSCREEN if diff >= margin_db else ROLE_HOST
                w2["mic_diff_db"] = round(diff, 1)
            else:
                role = prev_role
                w2["mic_diff_db"] = None
            w2["mic_role"] = role
            w2["speaker_id"] = role
            w2["is_target_speaker"] = True
            words.append(w2)
            prev_role = role
        seg_copy["words"] = words
        out.append(seg_copy)
    return out


def _is_countdown_word(word: str) -> bool:
    tok = _norm_token(word)
    if not tok:
        return False
    if tok in _COUNTDOWN_TOKENS or tok in _CUE_WORDS:
        return True
    # "5,4,3,2" may arrive as one token after normalisation ("5432") or as "五四三二"
    return len(tok) >= 2 and all(ch in _COUNTDOWN_TOKENS for ch in tok)


def _mark_countdown_cues(words: list[dict], min_tokens: int = 2) -> None:
    """Re-label a leading or trailing run of countdown tokens (>= min_tokens) as CUE in place."""
    if not words:
        return
    # Leading run
    i = 0
    while i < len(words) and _is_countdown_word(words[i].get("word", "")):
        i += 1
    if i >= min_tokens or (i == len(words) and i >= 1):
        for w in words[:i]:
            w["mic_role"] = ROLE_CUE
            w["speaker_id"] = ROLE_CUE
            w["is_target_speaker"] = False
    # Trailing run
    j = len(words)
    while j > i and _is_countdown_word(words[j - 1].get("word", "")):
        j -= 1
    if len(words) - j >= min_tokens:
        for w in words[j:]:
            w["mic_role"] = ROLE_CUE
            w["speaker_id"] = ROLE_CUE
            w["is_target_speaker"] = False


def _group_runs(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in words:
        role = w.get("mic_role", ROLE_HOST)
        if groups and groups[-1][0].get("mic_role", ROLE_HOST) == role:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def smooth_word_roles(
    words: list[dict],
    max_run_words: int = 1,
    max_run_chars: int = 2,
    max_gap: float = 0.25,
) -> None:
    """Flip tiny HOST/OFFSCREEN runs that are glued to a neighbouring run of the other role.

    A single short word at the edge of a phrase (for example the first word of the host's
    sentence) is often picked up on the wrong microphone. When such a run is at most
    max_run_words / max_run_chars long and the gap to a neighbouring run of the other role
    is below max_gap, it takes the neighbour's role. CUE runs are never changed.
    Operates in place; repeated until stable.
    """
    changed = True
    while changed:
        changed = False
        groups = _group_runs(words)
        for gi, g in enumerate(groups):
            role = g[0].get("mic_role", ROLE_HOST)
            if role == ROLE_CUE:
                continue
            n_chars = sum(len(_norm_token(w.get("word", ""))) for w in g)
            if len(g) > max_run_words or n_chars > max_run_chars:
                continue
            other = ROLE_OFFSCREEN if role == ROLE_HOST else ROLE_HOST
            best_gap = None
            for ni in (gi - 1, gi + 1):
                if ni < 0 or ni >= len(groups):
                    continue
                ng = groups[ni]
                if ng[0].get("mic_role", ROLE_HOST) != other:
                    continue
                if ni < gi:
                    gap = float(g[0].get("start", 0.0)) - float(ng[-1].get("end", 0.0))
                else:
                    gap = float(ng[0].get("start", 0.0)) - float(g[-1].get("end", 0.0))
                if gap < max_gap and (best_gap is None or gap < best_gap):
                    best_gap = gap
            if best_gap is not None:
                for w in g:
                    w["mic_role"] = other
                    w["speaker_id"] = other
                changed = True
                break


def split_segments_by_role(segments: list[dict]) -> list[dict]:
    """Split each raw segment into runs of words that share one role; renumber ids."""
    out: list[dict] = []
    next_id = 1
    for seg in segments:
        words = [dict(w) for w in (seg.get("words") or [])]
        if not words:
            seg_copy = dict(seg)
            seg_copy["id"] = next_id
            seg_copy.setdefault("speaker_id", ROLE_HOST)
            seg_copy.setdefault("is_target_speaker", True)
            seg_copy["words"] = []
            out.append(seg_copy)
            next_id += 1
            continue

        _mark_countdown_cues(words)
        smooth_word_roles(words)

        groups = _group_runs(words)

        for g in groups:
            role = g[0].get("mic_role", ROLE_HOST)
            text = "".join(w.get("word", "") for w in g).strip()
            starts = [float(w.get("start", 0.0)) for w in g]
            ends = [float(w.get("end", 0.0)) for w in g]
            out.append({
                "id": next_id,
                "start": round(min(starts), 2) if starts else float(seg.get("start", 0.0)),
                "end": round(max(ends), 2) if ends else float(seg.get("end", 0.0)),
                "text": text,
                "words": g,
                "speaker_id": role,
                "is_target_speaker": role != ROLE_CUE,
                "source_segment_id": seg.get("id"),
            })
            next_id += 1
    return out


def apply_mic_roles(
    segments: list[dict],
    video_path: Path | str,
    temp_dir: Path | str,
    offscreen_channel: str = "auto",
    margin_db: float = -3.0,
) -> tuple[list[dict], dict]:
    """Full pipeline: stereo WAV -> word roles -> role-split segments. Returns (segments, info)."""
    video_path = Path(video_path)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav = temp_dir / f"temp_{video_path.stem}_roles_stereo.wav"
    try:
        extract_stereo_wav(video_path, wav)
        audio, sr = load_stereo_audio(wav)
        if audio.shape[1] < 2:
            raise RuntimeError("音訊不是雙聲道，無法做麥克風聲道歸屬")
        channel = detect_offscreen_channel(audio) if offscreen_channel == "auto" else offscreen_channel
        attributed = attribute_word_roles(segments, audio, sr, offscreen_channel=channel, margin_db=margin_db)
        split = split_segments_by_role(attributed)
    finally:
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass

    counts = {r: 0 for r in ROLE_VALUES}
    for s in split:
        counts[s.get("speaker_id", ROLE_HOST)] = counts.get(s.get("speaker_id", ROLE_HOST), 0) + 1
    info = {"offscreen_channel": channel, "margin_db": margin_db, "segment_counts": counts}
    logger.info(
        "    [麥克風角色歸屬] 場外聲道=%s | 片段：HOST=%d, OFFSCREEN=%d, CUE=%d",
        channel, counts[ROLE_HOST], counts[ROLE_OFFSCREEN], counts[ROLE_CUE],
    )
    return split, info


def role_of_sentence(sentence: dict) -> str | None:
    """Return the sentence role when the sentence came from Show Mode, else None."""
    spk = sentence.get("speaker_id")
    return spk if spk in ROLE_VALUES else None


def has_role_tags(sentences: list[dict]) -> bool:
    return any(role_of_sentence(s) is not None for s in sentences or [])

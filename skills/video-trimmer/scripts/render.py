"""Video stream inspection and hardware-accelerated FFmpeg rough-cut rendering."""

import json
import logging
import subprocess

from .constants import CROSSFADE_DURATION_SEC
from .exceptions import FFmpegError

logger = logging.getLogger(__name__)

# Maximum concurrent VideoToolbox hardware decode sessions per FFmpeg command.
MAX_HWACCEL_INPUTS = 16
DEFAULT_RENDER_GOP = 30


def probe_video(video_path):
    """Probe video duration, frame dimensions, and frame rate via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate",
        "-of", "json", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe error: {res.stderr}")
    info = json.loads(res.stdout)
    duration = float(info["format"]["duration"])
    v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(v_stream.get("width", 1920))
    height = int(v_stream.get("height", 1080))
    fps_parts = v_stream.get("r_frame_rate", "24000/1001").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 23.976
    return duration, width, height, fps


def _build_render_cmd(
    edl,
    video_path,
    out_mp4_path,
    crf=18,
    encoder="h264_videotoolbox",
    use_hwaccel=True,
    gop=DEFAULT_RENDER_GOP,
):
    """
    Build the FFmpeg command for multi-input keyframe-accelerated cut rendering.

    Each retained clip uses fast input seeking (-ss and -to before -i) so FFmpeg
    seeks directly to the nearest preceding keyframe and skips discarded footage.
    """
    n = len(edl)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    enable_input_hwaccel = bool(use_hwaccel and n <= MAX_HWACCEL_INPUTS)
    filter_complex = []

    for i, c in enumerate(edl):
        start_sec = float(c["source_in"])
        end_sec = float(c["source_out"])
        dur = max(0.01, float(c.get("duration", end_sec - start_sec)))
        fade_d = min(CROSSFADE_DURATION_SEC, max(0.005, dur / 4.0))
        fade_out_st = max(0.0, dur - fade_d)

        if enable_input_hwaccel:
            cmd.extend(["-hwaccel", "videotoolbox"])
        cmd.extend([
            "-ss", f"{start_sec:.3f}",
            "-to", f"{end_sec:.3f}",
            "-i", str(video_path),
        ])

        filter_complex.append(
            f"[{i}:v]setpts=PTS-STARTPTS[v{i}]; "
            f"[{i}:a]asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade_d:.3f}:curve=iqsin,"
            f"afade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}:curve=oqsin[a{i}];"
        )

    concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(n)])
    filter_complex.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")

    cmd.extend([
        "-filter_complex", "".join(filter_complex),
        "-map", "[outv]", "-map", "[outa]",
    ])

    if encoder == "h264_videotoolbox":
        cmd.extend([
            "-c:v", "h264_videotoolbox",
            "-b:v", "12M",
            "-g", str(gop),
            "-pix_fmt", "yuv420p",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(crf),
            "-g", str(gop),
            "-pix_fmt", "yuv420p",
        ])

    cmd.extend([
        "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart",
        str(out_mp4_path),
    ])
    return cmd


def render_cut_video(edl, video_path, out_mp4_path, crf=18):
    """Render the trimmed MP4 with keyframe input seeking, 1s GOP, and VideoToolbox acceleration."""
    n = len(edl)
    if n == 0:
        logger.warning("No selected clips available for rendering.")
        return

    # 1. Try Apple Silicon VideoToolbox hardware encoding + hardware decoding
    cmd = _build_render_cmd(
        edl, video_path, out_mp4_path, crf=crf,
        encoder="h264_videotoolbox", use_hwaccel=True,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # 2. Retry VideoToolbox encoder without per-input hwaccel if input hwaccel failed
    if proc.returncode != 0 and n <= MAX_HWACCEL_INPUTS:
        cmd = _build_render_cmd(
            edl, video_path, out_mp4_path, crf=crf,
            encoder="h264_videotoolbox", use_hwaccel=False,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)

    # 3. Fallback to software libx264 encoder if VideoToolbox is unavailable
    if proc.returncode != 0:
        cmd = _build_render_cmd(
            edl, video_path, out_mp4_path, crf=crf,
            encoder="libx264", use_hwaccel=False,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise FFmpegError(f"FFmpeg render failed: {proc.stderr}")


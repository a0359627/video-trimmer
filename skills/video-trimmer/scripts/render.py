"""Video stream inspection and hardware-accelerated FFmpeg rough-cut rendering."""

import json
import logging
import os
import subprocess
import sys
import tempfile

from .constants import CROSSFADE_DURATION_SEC
from .exceptions import FFmpegError

logger = logging.getLogger(__name__)

# Maximum concurrent VideoToolbox hardware decode sessions per FFmpeg command.
MAX_HWACCEL_INPUTS = 16
DEFAULT_RENDER_GOP = 30

# Hardware H.264 encoders that use a fixed bitrate instead of CRF.
HW_BITRATE_ENCODERS = ("h264_videotoolbox", "h264_qsv", "h264_nvenc", "h264_amf")

# Windows CreateProcess limits the command line to 32767 characters.
# Above this length, the filtergraph moves to a file (-/filter_complex).
WINDOWS_CMDLINE_LIMIT = 30000


def _run(cmd):
    """Run a command and decode its output as UTF-8 on all platforms."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _encoder_attempts():
    """
    Return the ordered (encoder, use_hwaccel) attempts for this platform.

    macOS: VideoToolbox with hardware decode, then without, then libx264.
    Windows: Intel QSV, NVIDIA NVENC, AMD AMF, then libx264. An unavailable
    encoder fails at initialization, so the fallback cost is small.
    """
    if sys.platform == "darwin":
        return [("h264_videotoolbox", True), ("h264_videotoolbox", False), ("libx264", False)]
    if sys.platform == "win32":
        return [("h264_qsv", False), ("h264_nvenc", False), ("h264_amf", False), ("libx264", False)]
    return [("libx264", False)]


def probe_video(video_path):
    """Probe video duration, frame dimensions, and frame rate via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate",
        "-of", "json", str(video_path)
    ]
    res = _run(cmd)
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
            f"afade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}:curve=qsin[a{i}];"
        )

    concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(n)])
    filter_complex.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")

    cmd.extend([
        "-filter_complex", "".join(filter_complex),
        "-map", "[outv]", "-map", "[outa]",
    ])

    if encoder in HW_BITRATE_ENCODERS:
        cmd.extend([
            "-c:v", encoder,
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


def _run_ffmpeg_render(cmd):
    """
    Run an FFmpeg render command.

    On Windows, move the filtergraph to a temporary file when the command line
    is longer than WINDOWS_CMDLINE_LIMIT, to prevent WinError 206.
    """
    if sys.platform != "win32" or len(subprocess.list2cmdline(cmd)) <= WINDOWS_CMDLINE_LIMIT:
        return _run(cmd)

    idx = cmd.index("-filter_complex")
    fd, graph_path = tempfile.mkstemp(suffix="_filtergraph.txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(cmd[idx + 1])
        script_cmd = cmd[:idx] + ["-/filter_complex", graph_path] + cmd[idx + 2:]
        return _run(script_cmd)
    finally:
        os.remove(graph_path)


def render_cut_video(edl, video_path, out_mp4_path, crf=18):
    """Render the trimmed MP4 with keyframe input seeking, 1s GOP, and platform hardware encoding."""
    n = len(edl)
    if n == 0:
        logger.warning("No selected clips available for rendering.")
        return

    proc = None
    tried_cmds = []
    for encoder, use_hwaccel in _encoder_attempts():
        cmd = _build_render_cmd(
            edl, video_path, out_mp4_path, crf=crf,
            encoder=encoder, use_hwaccel=use_hwaccel,
        )
        # Above MAX_HWACCEL_INPUTS, hwaccel is off, so two attempts can build the same command.
        if cmd in tried_cmds:
            continue
        tried_cmds.append(cmd)
        proc = _run_ffmpeg_render(cmd)
        if proc.returncode == 0:
            logger.info("FFmpeg render encoder: %s", encoder)
            return
        logger.debug("FFmpeg encoder %s failed, trying next fallback.", encoder)

    raise FFmpegError(f"FFmpeg render failed: {proc.stderr}")

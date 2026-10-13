"""Custom exception hierarchy for Video Trimmer.

ASD-STE100:
Raise these exceptions inside internal modules instead of calling sys.exit().
Catch them at the top-level CLI entrypoint in scripts/video_trimmer.py.
"""


class VideoTrimmerError(Exception):
    """Base exception class for all Video Trimmer errors."""


class FFmpegError(VideoTrimmerError):
    """Raised when an ffmpeg or ffprobe command fails."""


class GeminiAPIError(VideoTrimmerError):
    """Raised when a Vertex AI Gemini API call or JSON response fails."""


class InvalidInputError(VideoTrimmerError):
    """Raised when a user-supplied file path, URL, or extension is invalid."""

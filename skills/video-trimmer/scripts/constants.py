"""Centrally managed named constants for the Video Trimmer pipeline."""

# Absolute dBFS silence floor based on EBU/ITU-R BS.1770 loudness gating (-40 dBFS).
SILENCE_THRESHOLD_DBFS = -40.0

# Equal-power audio micro-crossfade duration (20ms, lower bound of the 20-50ms click-free range).
CROSSFADE_DURATION_SEC = 0.020

# Number of 10ms steps (70ms total) required to confirm trailing speech decay into silence.
HANGOVER_STEPS = 7

# Supported input video file extensions (case-insensitive).
ALLOWED_INPUT_EXTENSIONS = (".mp4", ".mov")

# Vertex AI network retry configuration (1 initial attempt + up to 3 exponential-backoff retries).
GEMINI_RETRY_ATTEMPTS = 4
GEMINI_RETRY_WAIT_MIN_SEC = 1
GEMINI_RETRY_WAIT_MAX_SEC = 4

# Maximum output token limit for Gemini multimodal JSON responses.
GEMINI_MAX_OUTPUT_TOKENS = 65536

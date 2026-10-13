"""scripts/gcs_utils.py - Google Cloud Storage & Google Drive (ADC) Integration.

Provides:
- GCS client initialization via Application Default Credentials (ADC)
- SHA-256 / Google Drive MD5 smart caching for GCS uploads (`raw/` 2-day lifecycle)
- Google Drive API v3 file/folder link resolution and streaming download via ADC (`drive.readonly`)
- Zero-transfer GCS fast path when `gdrive_md5` already matches on the remote GCS blob
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

logger = logging.getLogger(__name__)

GDRIVE_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Extension to MIME type mapping for audio & video
_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".m4v": "video/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
}


def guess_mime_type(path: Path | str) -> str:
    """Best-effort MIME type inference for media files, defaulting to video/mp4."""
    suffix = Path(path).suffix.lower()
    return _MIME_TYPES.get(suffix, "video/mp4")


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """
    Parse a Google Cloud Storage URI (gs://bucket-name/path/to/blob).
    Returns (bucket_name, blob_name).
    """
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Invalid GCS URI (must start with gs://): {gcs_uri}")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    return bucket_name, blob_name


def get_gcs_client(project_id: str = None) -> storage.Client:
    """Return an authenticated Google Cloud Storage client via Application Default Credentials (ADC)."""
    return storage.Client(project=project_id)


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_file_md5(filepath: Path | str) -> str:
    """Compute MD5 hex digest of a local file (matches Google Drive md5Checksum)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_gdrive_source(val: str | Path | None) -> bool:
    """Return True if val is a Google Drive URL or gdrive:// URI."""
    if not val:
        return False
    s = str(val).strip()
    return (
        "drive.google.com" in s
        or "docs.google.com" in s
        or s.startswith("gdrive://")
    )


def parse_gdrive_url(url_or_id: str) -> dict:
    """
    Parse a Google Drive file/folder URL or ID into {'id': <id>, 'type': 'file'|'folder'|'unknown'}.
    """
    s = str(url_or_id).strip()
    if s.startswith("gdrive://"):
        rest = s[len("gdrive://"):].strip("/")
        if rest.startswith("folder/"):
            return {"id": rest.split("/", 1)[1].split("?")[0], "type": "folder"}
        if rest.startswith("file/"):
            return {"id": rest.split("/", 1)[1].split("?")[0], "type": "file"}
        return {"id": rest.split("?")[0], "type": "unknown"}

    m_folder = re.search(r"/folders/([a-zA-Z0-9_-]{10,})", s)
    if m_folder:
        return {"id": m_folder.group(1), "type": "folder"}

    m_file = re.search(r"/file/d/([a-zA-Z0-9_-]{10,})", s)
    if m_file:
        return {"id": m_file.group(1), "type": "file"}

    m_id = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", s)
    if m_id:
        return {"id": m_id.group(1), "type": "unknown"}

    if re.match(r"^[a-zA-Z0-9_-]{15,}$", s) and not Path(s).exists():
        return {"id": s, "type": "unknown"}

    raise ValueError(f"Invalid Google Drive URL or ID: {url_or_id}")


def _load_gcloud_user_drive_credentials():
    """
    Attempt to load personal user Google Drive credentials (user@gmail.com) from
    ~/.config/gcloud/legacy_credentials/<account>/adc.json (populated when user runs
    `gcloud auth login --enable-gdrive-access`, which uses Google Cloud SDK's
    whitelisted first-party CLOUDSDK_CLIENT_ID and never triggers 'This app is blocked').
    Returns refreshed Credentials if valid for Drive access, or None otherwise.
    """
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest

    gcloud_dir = os.path.expanduser("~/.config/gcloud")
    legacy_dir = os.path.join(gcloud_dir, "legacy_credentials")
    if not os.path.isdir(legacy_dir):
        return None

    active_account = None
    try:
        active_cfg_name = "default"
        active_cfg_file = os.path.join(gcloud_dir, "active_config")
        if os.path.isfile(active_cfg_file):
            with open(active_cfg_file, "r", encoding="utf-8") as f:
                active_cfg_name = f.read().strip() or "default"
        cfg_path = os.path.join(gcloud_dir, "configurations", f"config_{active_cfg_name}")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line and line.strip().startswith("account"):
                        active_account = line.split("=", 1)[1].strip()
                        break
    except Exception:
        pass

    candidate_accounts = []
    if active_account:
        candidate_accounts.append(active_account)
    try:
        for entry in sorted(os.listdir(legacy_dir)):
            if entry not in candidate_accounts:
                candidate_accounts.append(entry)
    except Exception:
        pass

    drive_scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    for acct in candidate_accounts:
        adc_file = os.path.join(legacy_dir, acct, "adc.json")
        if os.path.isfile(adc_file):
            try:
                creds, _ = google.auth.load_credentials_from_file(adc_file, scopes=drive_scopes)
                creds.refresh(GoogleAuthRequest())
                if creds.valid:
                    return creds
            except Exception:
                continue
    return None


def get_gdrive_session(project_id: str = None):
    """
    Return an HTTP session using Service Account Impersonation (via standard cloud-platform ADC)
    or public requests.Session() fallback so personal gcloud logins NEVER hit 'This app is blocked'.
    """
    import requests
    import google.auth
    from google.auth.transport.requests import AuthorizedSession, Request as GoogleAuthRequest

    quota_proj = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
    )
    source_creds = None
    default_proj = None
    try:
        source_creds, default_proj = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not quota_proj:
            quota_proj = default_proj
    except Exception:
        pass

    # Tier 1A: Personal Google Account credentials (`gcloud auth login --enable-gdrive-access`)
    user_drive_creds = _load_gcloud_user_drive_credentials()
    if user_drive_creds is not None:
        sess = AuthorizedSession(user_drive_creds)
        if quota_proj:
            sess.headers["X-Goog-User-Project"] = quota_proj
        return sess

    # Tier 1B: Native Service Account credentials
    if source_creds and hasattr(source_creds, "service_account_email"):
        sa_creds, _ = google.auth.default(scopes=GDRIVE_SCOPES)
        sess = AuthorizedSession(sa_creds)
        if quota_proj:
            sess.headers["X-Goog-User-Project"] = quota_proj
        return sess

    if source_creds and quota_proj:
        try:
            from google.auth import impersonated_credentials
            for prefix in ("video-trimmer-sa", "meeting-transcribe-sa", "multicam-video-sa"):
                sa_email = f"{prefix}@{quota_proj}.iam.gserviceaccount.com"
                try:
                    imp_creds = impersonated_credentials.Credentials(
                        source_credentials=source_creds,
                        target_principal=sa_email,
                        target_scopes=GDRIVE_SCOPES,
                        lifetime=3600,
                    )
                    imp_creds.refresh(GoogleAuthRequest())
                    sess = AuthorizedSession(imp_creds)
                    sess.headers["X-Goog-User-Project"] = quota_proj
                    return sess
                except Exception:
                    continue
        except Exception:
            pass

    return requests.Session()


def get_gdrive_file_metadata(url_or_id: str, project_id: str = None, session=None) -> dict:
    """Query Google Drive API v3 for file metadata (id, name, mimeType, size, md5Checksum)."""
    parsed = parse_gdrive_url(url_or_id)
    file_id = parsed["id"]
    sess = session or get_gdrive_session(project_id=project_id)
    api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {
        "fields": "id,name,mimeType,size,md5Checksum",
        "supportsAllDrives": "true",
    }
    resp = sess.get(api_url, params=params, timeout=30)
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"Google Drive API Permission Error ({resp.status_code}): {resp.text}\n"
            f"Run the following command to authorize ADC for Google Drive:\n"
            f"  gcloud auth application-default login --scopes=\"https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.readonly\""
        )
    resp.raise_for_status()
    return resp.json()


def fix_mojibake_filename(name: str) -> str:
    """
    Recover UTF-8 filenames that were decoded as ISO-8859-1 (latin-1) by HTTP headers.
    Leaves valid UTF-8 and ASCII filenames untouched.
    """
    if not name:
        return ""
    s = str(name)
    if any(0x80 <= ord(c) <= 0xFF for c in s) and all(ord(c) <= 0xFF for c in s):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def _decode_run(m: "re.Match[str]") -> str:
        chunk = m.group(0)
        try:
            return chunk.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return chunk

    if any(0x80 <= ord(c) <= 0xFF for c in s):
        s = re.sub(r"[\x80-\xff]{2,}", _decode_run, s)
    return s


def extract_filename_from_content_disposition(cd: str, fallback_name: str) -> str:
    """
    Extract and decode filename from an HTTP Content-Disposition header.
    Handles RFC 5987 filename*=UTF-8''... (case-insensitive) and ISO-8859-1 mojibake
    inside filename="..." headers.
    """
    from urllib.parse import unquote
    if not cd:
        return fix_mojibake_filename(fallback_name)
    m_utf8 = re.search(r"filename\*\s*=\s*(?:UTF-8|utf-8)''([^;\r\n]+)", cd, re.IGNORECASE)
    if m_utf8:
        raw_val = m_utf8.group(1).strip().strip("\"'")
        return fix_mojibake_filename(unquote(raw_val, encoding="utf-8", errors="replace"))
    m_quoted = re.search(r'filename\s*=\s*"([^"]+)"', cd, re.IGNORECASE)
    if m_quoted:
        raw_val = m_quoted.group(1).strip()
        return fix_mojibake_filename(unquote(raw_val, encoding="utf-8", errors="replace"))
    m_plain = re.search(r'filename\s*=\s*([^;\r\n]+)', cd, re.IGNORECASE)
    if m_plain:
        raw_val = m_plain.group(1).strip().strip("\"'")
        return fix_mojibake_filename(unquote(raw_val, encoding="utf-8", errors="replace"))
    return fix_mojibake_filename(fallback_name)


def download_gdrive_file_with_cache(
    url_or_id: str,
    target_dir: Path | str = None,
    project_id: str = None,
    force_download: bool = False,
) -> Path:
    """
    Download a Google Drive media file via ADC into `target_dir`, verifying remote MD5
    to skip re-downloading when a matching cached file already exists on disk.
    """
    parsed = parse_gdrive_url(url_or_id)
    file_id = parsed["id"]
    sess = get_gdrive_session(project_id=project_id)
    try:
        meta = get_gdrive_file_metadata(url_or_id, project_id=project_id, session=sess)
        filename = fix_mojibake_filename(meta.get("name") or f"gdrive_{file_id}.mp4")
        remote_md5 = meta.get("md5Checksum")
        remote_size = int(meta.get("size", 0) or 0)
    except Exception:
        import requests
        dest_dir = Path(target_dir or (Path.cwd() / "gdrive_inputs")).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dl_url = "https://drive.usercontent.google.com/download"
        with requests.get(dl_url, params={"id": file_id, "export": "download", "confirm": "t"}, stream=True, timeout=600) as r:
            r.raise_for_status()
            cd = r.headers.get("Content-Disposition", "")
            detected_name = extract_filename_from_content_disposition(cd, f"gdrive_{file_id}.mp4")
            local_path = dest_dir / detected_name
            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(local_path)
            return local_path

    dest_dir = Path(target_dir or (Path.cwd() / "gdrive_inputs")).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / filename

    if not force_download and local_path.is_file():
        if remote_md5 and local_path.stat().st_size == remote_size:
            if compute_file_md5(local_path) == remote_md5:
                logger.info("[GDrive Cache Hit] Local MD5 matches Google Drive (%s). Skipping download.", local_path.name)
                return local_path

    size_mb = remote_size / (1024 * 1024)
    logger.info("[GDrive Download] Downloading '%s' (%.1f MB) from Google Drive via ADC...", filename, size_mb)
    dl_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true"
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with sess.get(dl_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(local_path)
    logger.info("[Done] Google Drive download complete: %s", local_path)
    return local_path


def upload_file_to_gcs(
    local_path: Path | str,
    bucket_name: str,
    destination_blob_name: str,
    content_type: str = None,
    client: storage.Client = None,
    extra_metadata: dict = None,
    force_upload: bool = False,
) -> str:
    """
    Upload a local file to a GCS bucket with SHA-256 / gdrive_md5 metadata caching.
    Returns the gs:// URI of the uploaded blob.
    """
    local_p = Path(local_path).resolve()
    if not local_p.is_file():
        raise FileNotFoundError(f"Local file not found: {local_p}")

    gcs_client = client or get_gcs_client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        file_sha256 = compute_file_sha256(local_p)
        file_md5 = compute_file_md5(local_p)
        file_size = local_p.stat().st_size
    except OSError:
        file_sha256 = None
        file_md5 = None
        file_size = None

    if not force_upload and file_sha256 and file_size is not None:
        try:
            if blob.exists():
                blob.reload()
                remote_meta = blob.metadata or {}
                if (
                    blob.size == file_size
                    and (
                        remote_meta.get("sha256") == file_sha256
                        or remote_meta.get("gdrive_md5") == file_md5
                    )
                ):
                    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
                    logger.info("[GCS Cache Hit] Remote blob hash matches (%s). Skipping upload.", gcs_uri)
                    return gcs_uri
        except Exception:
            pass

    if content_type:
        blob.content_type = content_type

    if file_sha256 or extra_metadata:
        meta = {}
        if file_sha256:
            meta["sha256"] = file_sha256
        if file_md5:
            meta["gdrive_md5"] = file_md5
        if extra_metadata:
            meta.update(extra_metadata)
        blob.metadata = meta

    logger.info("[*] Uploading %s to gs://%s/%s...", local_p.name, bucket_name, destination_blob_name)
    blob.upload_from_filename(str(local_p))
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    logger.info("[Done] Upload complete: %s", gcs_uri)
    return gcs_uri


def delete_gcs_blob(gcs_uri: str, client: storage.Client = None) -> None:
    """
    Delete a blob by its gs:// URI. Used to clean up ephemeral raw/ uploads
    when requested. Missing blobs are treated as already cleaned up.
    """
    try:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        gcs_client = client or get_gcs_client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info("[Done] Deleted temporary GCS blob: %s", gcs_uri)
    except Exception as e:
        logger.debug("Ignored error while deleting GCS blob (%s): %s", gcs_uri, e)

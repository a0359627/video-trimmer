"""母帶不上雲：沒有指定 --analysis-video、旁邊也沒有 _ai_proxy.mp4 時，一律先轉 AI proxy 再上傳，不上傳原檔。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import gemini_client  # noqa: E402


class NoMasterUploadTest(unittest.TestCase):
    def _run(self, name="C0001.MP4", already_proxy=False, make_proxy=False):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / name
            src.write_bytes(b"x" * 1024)                     # 很小的檔案（以前 1.8 GB 以下會直接上傳原檔）
            if make_proxy:
                (Path(d) / f"{src.stem}_ai_proxy.mp4").write_bytes(b"p" * 10)
            uploaded = {}

            def fake_ffmpeg(cmd, check):
                Path(cmd[-1]).write_bytes(b"proxy")

            def fake_upload(local_path, **kw):
                uploaded["path"] = Path(local_path)
                return "gs://bucket/raw/x.mp4"

            with mock.patch.object(gemini_client.subprocess, "run", side_effect=fake_ffmpeg), \
                    mock.patch.object(gemini_client, "upload_file_to_gcs", side_effect=fake_upload):
                gemini_client.stage_video_to_gcs(src, "bucket", gcs_client=object(), already_proxy=already_proxy)
            return src, uploaded["path"]

    def test_small_master_is_never_uploaded(self):
        src, up = self._run()
        self.assertNotEqual(up.name, src.name)
        self.assertTrue(up.name.endswith("_ai_proxy.mp4"))

    def test_existing_proxy_is_reused(self):
        src, up = self._run(make_proxy=True)
        self.assertEqual(up.name, f"{src.stem}_ai_proxy.mp4")

    def test_analysis_video_is_uploaded_as_is(self):
        src, up = self._run(name="proxy 1080p.mp4", already_proxy=True)
        self.assertEqual(up.name, src.name)


if __name__ == "__main__":
    unittest.main()

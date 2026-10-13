"""exporters.py 的離線單元測試：假造 EDL list，驗證輸出 XML 為合法格式且時間碼換算正確。"""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.exporters import XML_PROFILES, generate_edl_csv, generate_edl_report, generate_fcp7_xml, generate_fcpxml


def _sample_edl():
    return [
        {
            "clip_id": 1, "topic": "開場", "source_in": 10.0, "source_out": 15.0, "duration": 5.0,
            "cps": 4.0, "in_margin": 0.1, "out_margin": 0.15,
            "transcript": "大家好", "visual_check": "眼神就緒", "audio_check": "收音完整",
        },
        {
            "clip_id": 2, "topic": "重點段落說明", "source_in": 20.0, "source_out": 27.5, "duration": 7.5,
            "cps": 5.0, "in_margin": 0.08, "out_margin": 0.12,
            "transcript": "這是重點", "visual_check": "眼神就緒", "audio_check": "收音完整",
        },
    ]


class TestGenerateFcp7Xml(unittest.TestCase):
    def test_output_is_well_formed_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "out.xml"

            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                               width=1920, height=1080, fps=24.0)

            root = ET.parse(xml_path).getroot()
            assert root.tag == "xmeml"
            # one video clipitem per cut plus one audio clipitem per cut per channel (stereo default)
            assert len(root.findall(".//video//clipitem")) == len(_sample_edl())
            assert len(root.findall(".//audio/track")) == 2
            assert len(root.findall(".//audio//clipitem")) == len(_sample_edl()) * 2
            seq = root.find("sequence")
            assert int(seq.findtext("duration")) == sum(round((c["source_out"] - c["source_in"]) * 24.0) for c in _sample_edl())
            pathurl = root.find(".//file/pathurl").text
            assert pathurl.startswith("file://localhost/") and "file:///" not in pathurl
            assert root.find(".//file/media/audio/channelcount").text == "2"
            audio_clip = root.find(".//audio//clipitem")
            assert audio_clip.find("sourcetrack/trackindex").text == "1"
            assert audio_clip.find("link/linkclipref").text == "clipitem-v1"

    def test_sequence_markers_are_written_and_escaped(self):
        for profile in XML_PROFILES:
            with self.subTest(xml_profile=profile), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                video_path = tmp_path / "take1.mp4"
                video_path.write_bytes(b"")
                xml_path = tmp_path / "out.xml"
                markers = [{"name": "確認 <A&B>", "comment": "原片 00:01:00:00 | 說明", "frame": 120}]
                generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                                   width=1920, height=1080, fps=24.0, audio_channels=2, markers=markers,
                                   xml_profile=profile)
                root = ET.parse(xml_path).getroot()
                m = root.findall("sequence/marker")
                assert len(m) == 1
                assert m[0].findtext("name") == "確認 <A&B>"
                assert m[0].findtext("in") == "120" and m[0].findtext("out") == "-1"

    def test_timecode_conversion_is_frame_accurate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "out.xml"
            fps = 24.0

            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                               width=1920, height=1080, fps=fps)

            root = ET.parse(xml_path).getroot()
            first_clip = root.find(".//video//clipitem")
            assert int(first_clip.find("in").text) == round(10.0 * fps)
            assert int(first_clip.find("out").text) == round(15.0 * fps)

            second_clip = root.findall(".//video//clipitem")[1]
            assert int(second_clip.find("start").text) == round(15.0 * fps) - round(10.0 * fps)

    def test_ntsc_flag_reflects_fractional_vs_integer_fps(self):
        for profile in XML_PROFILES:
            with self.subTest(xml_profile=profile), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                video_path = tmp_path / "take1.mp4"
                video_path.write_bytes(b"")
                xml_ntsc = tmp_path / "ntsc.xml"
                xml_pal = tmp_path / "pal.xml"

                generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_ntsc,
                                   width=1920, height=1080, fps=29.97, xml_profile=profile)
                generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_pal,
                                   width=1920, height=1080, fps=25.0, xml_profile=profile)

                root_ntsc = ET.parse(xml_ntsc).getroot()
                root_pal = ET.parse(xml_pal).getroot()
                assert root_ntsc.find(".//rate/timebase").text == "30"
                assert root_ntsc.find(".//rate/ntsc").text == "TRUE"
                assert root_pal.find(".//rate/timebase").text == "25"
                assert root_pal.find(".//rate/ntsc").text == "FALSE"

    def test_camera_timecode_offsets_frames_properly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "out.xml"
            fps = 24.0
            # 01:00:00:00 at 24fps = 86400 frames
            start_tc = "01:00:00:00"
            expected_offset = 86400

            # 相機時間碼直通與移除 <frame> 是 DaVinci Resolve 的修法，屬於 davinci 口味
            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                               width=1920, height=1080, fps=fps, start_tc=start_tc, xml_profile="davinci")

            root = ET.parse(xml_path).getroot()
            assert root.get("version") == "5"
            assert root[0].tag == "sequence"
            first_clip = root.find(".//video//clipitem")
            assert int(first_clip.find("in").text) == round(10.0 * fps)
            assert int(first_clip.find("out").text) == round(15.0 * fps)

            tc_node = root.find(".//file//timecode")
            assert tc_node.findtext("string") == "01:00:00:00"
            # In FCP7 XML for DaVinci Resolve, frame must not be present in file timecode to prevent freeze-frame
            assert tc_node.find("frame") is None

    def test_clipitem_ids_and_file_ids_are_globally_unique(self):
        """Verify all clipitem IDs and file IDs are globally unique to prevent Resolve media disconnect."""
        # 兩種口味 x 單聲道 / 雙聲道 / 預設（ffprobe 失敗時為 2）都要成立；第二個 EDL 夠長，能撞出 "C6619.MP4 1" 類型的碰撞
        long_edl = [
            {"clip_id": i + 1, "topic": f"T{i}", "source_in": i * 3.0, "source_out": i * 3.0 + 2.0, "duration": 2.0}
            for i in range(12)
        ]
        for profile in XML_PROFILES:
            for channels in (None, 1, 2):
                for edl in (_sample_edl(), long_edl):
                    with self.subTest(xml_profile=profile, audio_channels=channels, clips=len(edl)), \
                            tempfile.TemporaryDirectory() as tmp:
                        tmp_path = Path(tmp)
                        video_path = tmp_path / "take1.mp4"
                        video_path.write_bytes(b"")
                        xml_path = tmp_path / "out.xml"

                        generate_fcp7_xml(edl, video_path, total_source_dur=40.0, output_xml_path=xml_path,
                                           width=1920, height=1080, fps=24.0, audio_channels=channels,
                                           xml_profile=profile)

                        root = ET.parse(xml_path).getroot()
                        clip_ids = [elem.get("id") for elem in root.findall(".//clipitem")]
                        assert len(clip_ids) == len(set(clip_ids)), f"Duplicate clipitem IDs detected: {clip_ids}"
                        file_ids = {elem.get("id") for elem in root.findall(".//file")}
                        assert len(file_ids) == 1
                        assert file_ids.isdisjoint(clip_ids), f"File IDs and Clip IDs must never collide: {file_ids & set(clip_ids)}"
                        # 每個 link 都指向真實存在的 clipitem
                        link_refs = {e.text for e in root.iter("linkclipref")}
                        assert link_refs <= set(clip_ids), link_refs - set(clip_ids)


    def test_premiere_profile_keeps_zero_file_timecode_and_xmeml_v4(self):
        """預設 premiere 口味維持 show-mode 的結構：xmeml v4、<file><timecode> 為 00:00:00:00 並帶 <frame>0。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "out.xml"
            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                               width=1920, height=1080, fps=24.0, start_tc="06:58:34:00")
            root = ET.parse(xml_path).getroot()
            assert root.get("version") == "4"
            assert root.find(".//file").get("id") == "file-1"
            tc_node = root.find(".//file/timecode")
            assert tc_node.findtext("string") == "00:00:00:00"
            assert tc_node.findtext("frame") == "0"

    def test_davinci_profile_uses_namespaced_ids_and_autodetects_camera_timecode(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "C6619.MP4"
            video_path.write_bytes(b"\x00" * 16)
            xml_path = tmp_path / "out.xml"
            with patch("scripts.exporters.get_video_timecode", return_value="06:58:34:00") as mock_tc:
                generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                                   width=3840, height=2160, fps=24000 / 1001, audio_channels=2,
                                   xml_profile="davinci")
                mock_tc.assert_called_once()
            root = ET.parse(xml_path).getroot()
            assert root.get("version") == "5"
            assert root.find(".//file").get("id") == "C6619.MP4_master"
            video_ids = [c.get("id") for c in root.findall(".//video//clipitem")]
            assert video_ids == ["C6619.MP4_v0", "C6619.MP4_v1"]
            audio_tracks = root.findall(".//audio/track")
            assert [c.get("id") for c in audio_tracks[0].findall("clipitem")] == ["C6619.MP4_a0", "C6619.MP4_a1"]
            assert [c.get("id") for c in audio_tracks[1].findall("clipitem")] == ["C6619.MP4_a0_ch2", "C6619.MP4_a1_ch2"]
            tc_node = root.find(".//file/timecode")
            assert tc_node.findtext("string") == "06:58:34:00"
            assert tc_node.find("frame") is None
            assert tc_node.findtext("reel/name") == "C6619"

    def test_profiles_share_identical_timeline_placement(self):
        """兩種口味只差 ID / timecode / 版本號，in/out/start/end 必須逐一相同。"""
        roots = {}
        for profile in XML_PROFILES:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                video_path = tmp_path / "take1.mp4"
                video_path.write_bytes(b"")
                xml_path = tmp_path / "out.xml"
                generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                                   width=1920, height=1080, fps=24000 / 1001, audio_channels=2,
                                   start_tc="01:00:00:00", xml_profile=profile)
                roots[profile] = ET.parse(xml_path).getroot()

        def placement(root):
            return [tuple(c.findtext(k) for k in ("in", "out", "start", "end")) for c in root.iter("clipitem")]

        assert placement(roots["premiere"]) == placement(roots["davinci"])
        assert roots["premiere"].find("sequence/duration").text == roots["davinci"].find("sequence/duration").text

    def test_unknown_xml_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            with self.assertRaises(ValueError):
                generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0,
                                   output_xml_path=tmp_path / "out.xml", fps=24.0, xml_profile="avid")


class TestGenerateFcpxml(unittest.TestCase):
    def test_output_is_well_formed_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            fcpxml_path = tmp_path / "out.fcpxml"
            edl = _sample_edl()
            total_out_dur = sum(c["duration"] for c in edl)

            generate_fcpxml(edl, video_path, total_source_dur=30.0, total_out_dur=total_out_dur,
                             output_fcpxml_path=fcpxml_path, fps=23.976)

            root = ET.parse(fcpxml_path).getroot()
            assert root.tag == "fcpxml"
            assert len(root.findall(".//asset-clip")) == len(edl)

    def test_duration_fraction_matches_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            fcpxml_path = tmp_path / "out.fcpxml"
            edl = [{"clip_id": 1, "topic": "A", "source_in": 0.0, "source_out": 2.0, "duration": 2.0}]

            generate_fcpxml(edl, video_path, total_source_dur=2.0, total_out_dur=2.0,
                             output_fcpxml_path=fcpxml_path, fps=23.976)

            root = ET.parse(fcpxml_path).getroot()
            clip = root.find(".//asset-clip")
            num, den = clip.get("duration").rstrip("s").split("/")
            seconds = float(num) / float(den)
            frame_duration = 1001 / 24000
            self.assertAlmostEqual(seconds, 2.0, delta=frame_duration)


    def test_supports_multiple_frame_rates_and_cumulative_spine_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            fcpxml_60 = tmp_path / "out_60.fcpxml"
            edl = _sample_edl()

            generate_fcpxml(edl, video_path, total_source_dur=30.0, total_out_dur=12.5,
                             output_fcpxml_path=fcpxml_60, fps=60.0, width=3840, height=2160)

            root = ET.parse(fcpxml_60).getroot()
            fmt = root.find(".//format")
            assert fmt.get("frameDuration") == "100/6000s"
            assert fmt.get("width") == "3840"
            assert fmt.get("height") == "2160"
            clips = root.findall(".//asset-clip")
            assert clips[0].get("offset") == "0s"
            assert clips[1].get("offset") == "30000/6000s"

class TestGenerateEdlCsv(unittest.TestCase):
    def test_csv_contains_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "out.csv"
            generate_edl_csv(_sample_edl(), csv_path)

            content = csv_path.read_text(encoding="utf-8-sig")
            lines = content.strip().splitlines()
            assert lines[0].startswith("Clip_ID,Topic,Source_In")
            assert len(lines) == 1 + len(_sample_edl())
            assert "開場" in lines[1]

    def test_quotes_in_fields_are_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "out.csv"
            edl = [{
                "clip_id": 1, "topic": "A", "source_in": 0.0, "source_out": 1.0, "duration": 1.0,
                "cps": 3.0, "in_margin": 0.1, "out_margin": 0.1,
                "transcript": '他說"你好"', "visual_check": "ok", "audio_check": "ok",
            }]
            generate_edl_csv(edl, csv_path)
            content = csv_path.read_text(encoding="utf-8-sig")
            assert '""你好""' in content


class TestGenerateEdlReport(unittest.TestCase):
    def test_report_contains_metrics_and_markdown_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "out_report.md"
            generate_edl_report(_sample_edl(), total_source_dur=30.0, report_path=report_path, base_name="test_video")

            content = report_path.read_text(encoding="utf-8")
            assert "# test_video EDL 剪輯驗證報告" in content
            assert "原始素材時長: 30.00 秒" in content
            assert "| Clip ID | 題旨/段落 |" in content
            assert "| 1 | 開場 | 10.00 | 15.00 | 5.00 | 4.00 | 大家好 |" in content
            # Ensure zero emojis in headings or tables
            for forbidden_emoji in ["🎬", "📊", "✅", "❌", "⏱️"]:
                assert forbidden_emoji not in content


class TestRenderCutVideo(unittest.TestCase):
    def test_build_render_cmd_uses_per_clip_fast_seeking_and_gop_30(self):
        from scripts.render import _build_render_cmd

        edl = _sample_edl()
        cmd = _build_render_cmd(
            edl,
            video_path="raw_footage.mp4",
            out_mp4_path="out_trimmed.mp4",
            crf=18,
            encoder="h264_videotoolbox",
            use_hwaccel=True,
        )
        cmd_str = " ".join(cmd)
        self.assertIn("-hwaccel videotoolbox -ss 10.000 -to 15.000 -i raw_footage.mp4", cmd_str)
        self.assertIn("-hwaccel videotoolbox -ss 20.000 -to 27.500 -i raw_footage.mp4", cmd_str)
        self.assertIn("afade=t=in:st=0:d=0.020:curve=iqsin", cmd_str)
        self.assertIn("d=0.020:curve=qsin", cmd_str)
        self.assertNotIn("curve=oqsin", cmd_str)
        self.assertIn("-c:v h264_videotoolbox -b:v 12M -g 30", cmd_str)
        self.assertIn("-movflags +faststart", cmd_str)

    def test_render_cut_video_falls_back_to_libx264_when_videotoolbox_fails(self):
        from unittest.mock import MagicMock, patch
        from scripts.render import render_cut_video

        fail_proc = MagicMock(returncode=1, stderr="vt failed")
        ok_proc = MagicMock(returncode=0, stderr="")
        with patch("scripts.render.subprocess.run", side_effect=[fail_proc, fail_proc, ok_proc]) as mock_run:
            render_cut_video(_sample_edl(), "raw_footage.mp4", "out_trimmed.mp4", crf=20)
            self.assertEqual(mock_run.call_count, 3)
            final_cmd = " ".join(mock_run.call_args_list[2][0][0])
            self.assertIn("-c:v libx264 -preset fast -crf 20 -g 30", final_cmd)

    def test_real_ffmpeg_filtergraph_execution(self):
        import shutil
        import subprocess
        from scripts.render import render_cut_video

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_mp4 = tmp_path / "synthetic_input.mp4"
            out_mp4 = tmp_path / "synthetic_trimmed.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=1.5",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.5",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "15",
                    "-c:a", "aac",
                    str(src_mp4),
                ],
                check=True,
                capture_output=True,
            )
            edl = [
                {"clip_id": 1, "topic": "A", "source_in": 0.1, "source_out": 0.5, "duration": 0.4},
                {"clip_id": 2, "topic": "B", "source_in": 0.8, "source_out": 1.3, "duration": 0.5},
            ]
            render_cut_video(edl, src_mp4, out_mp4, crf=24)
            self.assertTrue(out_mp4.exists())
            self.assertGreater(out_mp4.stat().st_size, 0)


class TestResolveOutputDir(unittest.TestCase):
    def test_defaults_to_output_subdir_under_parent(self):
        from scripts.video_trimmer import _resolve_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp)
            out_dir = _resolve_output_dir(None, parent_dir)
            self.assertEqual(out_dir, parent_dir.resolve() / "output")
            self.assertTrue(out_dir.is_dir())

    def test_explicit_output_dir_overrides_default(self):
        from scripts.video_trimmer import _resolve_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp)
            custom_dir = parent_dir / "custom_deliverables"
            out_dir = _resolve_output_dir(str(custom_dir), parent_dir)
            self.assertEqual(out_dir, custom_dir.resolve())
            self.assertTrue(out_dir.is_dir())



import json
import os
import subprocess
import urllib.parse
from pathlib import Path


def file_path_to_url(path_str):
    """Convert absolute path to URL-encoded file:/// URI format for XML."""
    abs_path = os.path.abspath(str(path_str))
    encoded = urllib.parse.quote(abs_path, safe="/:")
    return f"file://{encoded}"


def get_video_timecode(video_path):
    """Extract embedded camera timecode from video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags=timecode:stream_tags=timecode",
            "-of", "json",
            str(video_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        tc = data.get("format", {}).get("tags", {}).get("timecode")
        if tc:
            return tc
        for s in data.get("streams", []):
            stc = s.get("tags", {}).get("timecode")
            if stc:
                return stc
    except Exception:
        pass
    return "00:00:00:00"


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976, start_tc="00:00:00:00"):
    """產生 Premiere Pro / DaVinci Resolve 相容的 FCP 7 XML (xmeml v5)"""
    timebase = int(round(fps))
    ntsc_str = "TRUE" if (abs(fps - 23.976) < 0.01 or abs(fps - 29.97) < 0.01 or abs(fps - 59.94) < 0.01) else "FALSE"
    file_url = file_path_to_url(video_path)
    video_name = Path(video_path).name
    reel_name = Path(video_path).stem
    total_src_frames = int(round(total_source_dur * fps))

    # Auto-detect camera timecode if start_tc is default 00:00:00:00
    if (not start_tc or start_tc == "00:00:00:00") and Path(video_path).exists() and Path(video_path).is_file() and Path(video_path).stat().st_size > 0:
        detected_tc = get_video_timecode(video_path)
        if detected_tc and detected_tc != "00:00:00:00":
            start_tc = detected_tc

    if not start_tc:
        start_tc = "00:00:00:00"

    def s2f(sec):
        return int(round(sec * fps))

    total_seq_frames = 0
    for clip in edl:
        in_f = s2f(clip["source_in"])
        out_f = s2f(clip["source_out"])
        dur_f = max(1, out_f - in_f)
        total_seq_frames += dur_f

    file_id = f"{video_name}_master"

    master_file_node = f"""                <file id="{file_id}">
                  <duration>{total_src_frames}</duration>
                  <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>
                  <name>{video_name}</name>
                  <pathurl>{file_url}</pathurl>
                  <timecode>
                    <string>{start_tc}</string>
                    <displayformat>NDF</displayformat>
                    <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>
                    <reel><name>{reel_name}</name></reel>
                  </timecode>
                  <media>
                    <video>
                      <duration>{total_src_frames}</duration>
                      <samplecharacteristics>
                        <width>{width}</width>
                        <height>{height}</height>
                      </samplecharacteristics>
                    </video>
                    <audio>
                      <channelcount>2</channelcount>
                    </audio>
                  </media>
                </file>"""

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="5">',
        '  <sequence>',
        f'    <name>{reel_name}_TrimmerCut</name>',
        f'    <duration>{total_seq_frames}</duration>',
        f'    <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>',
        '    <in>-1</in>',
        '    <out>-1</out>',
        '    <timecode>',
        '      <string>00:00:00:00</string>',
        '      <frame>0</frame>',
        '      <displayformat>NDF</displayformat>',
        f'      <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>',
        '    </timecode>',
        '    <media>',
        '      <video>',
        '        <format>',
        '          <samplecharacteristics>',
        f'            <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>',
        f'            <width>{width}</width>',
        f'            <height>{height}</height>',
        '            <pixelaspectratio>square</pixelaspectratio>',
        '          </samplecharacteristics>',
        '        </format>',
        '        <track>'
    ]

    timeline_cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        start_frame = timeline_cursor
        end_frame = timeline_cursor + dur_frame
        timeline_cursor = end_frame

        clip_file_node = master_file_node if idx == 0 else f'                <file id="{file_id}" />'
        clip_comment = f'Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}'

        xml_lines.extend([
            f'          <clipitem id="{video_name}_v{idx}">',
            f'            <name>{video_name}</name>',
            f'            <duration>{total_src_frames}</duration>',
            f'            <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>',
            f'            <start>{start_frame}</start>',
            f'            <end>{end_frame}</end>',
            f'            <enabled>TRUE</enabled>',
            f'            <in>{in_frame}</in>',
            f'            <out>{out_frame}</out>',
            clip_file_node,
            '            <sourcetrack>',
            '              <mediatype>video</mediatype>',
            '              <trackindex>1</trackindex>',
            '            </sourcetrack>',
            f'            <comments>{clip_comment}</comments>',
            '          </clipitem>'
        ])

    xml_lines.extend([
        '        </track>',
        '      </video>',
        '      <audio>',
        '        <track>'
    ])

    timeline_cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        start_frame = timeline_cursor
        end_frame = timeline_cursor + dur_frame
        timeline_cursor = end_frame

        clip_comment = f'Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}'
        audio_clip_id = f"{video_name}_a{idx}"

        xml_lines.extend([
            f'          <clipitem id="{audio_clip_id}">',
            f'            <name>{video_name}</name>',
            f'            <duration>{total_src_frames}</duration>',
            f'            <rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>',
            f'            <start>{start_frame}</start>',
            f'            <end>{end_frame}</end>',
            f'            <enabled>TRUE</enabled>',
            f'            <in>{in_frame}</in>',
            f'            <out>{out_frame}</out>',
            f'            <file id="{file_id}" />',
            '            <sourcetrack>',
            '              <mediatype>audio</mediatype>',
            '              <trackindex>1</trackindex>',
            '            </sourcetrack>',
            f'            <comments>{clip_comment}</comments>',
            '          </clipitem>'
        ])

    xml_lines.extend([
        '        </track>',
        '      </audio>',
        '    </media>',
        '  </sequence>',
        '</xmeml>'
    ])

    Path(output_xml_path).write_text('\n'.join(xml_lines), encoding='utf-8')



def generate_fcpxml(edl, video_path, total_source_dur, total_out_dur, output_fcpxml_path, fps=23.976):
    """產生 Final Cut Pro X 相容的 FCPXML (v1.9)"""
    file_url = file_path_to_url(video_path)

    def sec_to_fraction(sec):
        frames = int(round(sec * 24000 / 1001))
        return f"{frames * 1001}/24000s"

    total_out_frac = sec_to_fraction(total_out_dur)
    total_src_frac = sec_to_fraction(total_source_dur)

    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE fcpxml>',
        '<fcpxml version="1.9">',
        '  <resources>',
        '    <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>',
        f'    <asset id="r2" name="{Path(video_path).name}" src="{file_url}" start="0s" duration="{total_src_frac}" hasVideo="1" hasAudio="1" format="r1"/>',
        '  </resources>',
        '  <library>',
        f'    <event name="VideoTrimmer_{Path(video_path).stem}">',
        f'      <project name="{Path(video_path).stem}_TrimmerCut">',
        f'        <sequence format="r1" duration="{total_out_frac}">',
        '          <spine>'
    ]

    for clip in edl:
        clip_dur = clip["duration"]
        start_frac = sec_to_fraction(clip["source_in"])
        dur_frac = sec_to_fraction(clip_dur)
        name = f"Clip_{clip['clip_id']:02d}_{clip['topic'][:15]}"
        xml.append(f'            <asset-clip name="{name}" ref="r2" offset="0s" start="{start_frac}" duration="{dur_frac}"/>')

    xml.extend([
        '          </spine>',
        '        </sequence>',
        '      </project>',
        '    </event>',
        '  </library>',
        '</fcpxml>'
    ])

    Path(output_fcpxml_path).write_text('\n'.join(xml), encoding='utf-8')


def generate_edl_csv(edl, csv_path):
    """輸出 EDL 表格清單 (CSV, UTF-8 BOM 供 Excel 開啟)"""
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("Clip_ID,Topic,Source_In,Source_Out,Duration,CPS,In_Margin,Out_Margin,Transcript,Visual_Check,Audio_Check\n")
        for c in edl:
            tr = c.get("transcript", "").replace('"', '""')
            vc = c.get("visual_check", "OK").replace('"', '""')
            ac = c.get("audio_check", "OK").replace('"', '""')
            cps = c.get("cps", 0.0)
            in_m = c.get("in_margin", 0.0)
            out_m = c.get("out_margin", 0.0)
            f.write(f'{c["clip_id"]},"{c.get("topic", "")}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},{cps:.2f},{in_m:.2f},{out_m:.2f},"{tr}","{vc}","{ac}"\n')


def generate_edl_report(edl, total_source_dur, report_path, base_name=""):
    """
    輸出 Markdown 格式的 EDL 剪輯驗證報告 (_edl_report.md)。
    遵守 Zero-Emoji Policy，提供純技術指標與剪輯表格。
    """
    total_out_dur = round(sum(c.get("duration", 0.0) for c in edl), 2)
    cut_ratio = (1.0 - total_out_dur / max(1.0, total_source_dur)) * 100
    avg_cps = round(sum(c.get("cps", 0.0) for c in edl) / max(1, len(edl)), 2)

    lines = [
        f"# {base_name} EDL 剪輯驗證報告",
        "",
        "## 專案指標概要",
        f"- 原始素材時長: {total_source_dur:.2f} 秒 ({total_source_dur / 60:.2f} 分鐘)",
        f"- 粗剪輸出時長: {total_out_dur:.2f} 秒 ({total_out_dur / 60:.2f} 分鐘)",
        f"- 濃縮精簡率: {cut_ratio:.1f}%",
        f"- 總鏡頭數: {len(edl)} 個",
        f"- 全片平均語速: {avg_cps:.2f} 字/秒",
        "",
        "## 鏡頭剪輯明細表",
        "| Clip ID | 題旨/段落 | 來源入點 (s) | 來源出點 (s) | 長度 (s) | 語速 (CPS) | 台詞摘要 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for c in edl:
        cid = c.get("clip_id", 0)
        top = c.get("topic", "")
        sin = c.get("source_in", 0.0)
        sout = c.get("source_out", 0.0)
        dur = c.get("duration", 0.0)
        cps = c.get("cps", 0.0)
        tr = c.get("transcript", "").replace("\n", " ").strip()
        tr_short = tr[:45] + ("..." if len(tr) > 45 else "")
        lines.append(f"| {cid} | {top} | {sin:.2f} | {sout:.2f} | {dur:.2f} | {cps:.2f} | {tr_short} |")

    lines.append("")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


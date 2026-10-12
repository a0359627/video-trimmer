"""輸出格式產生：FCP 7 XML、FCPXML、CSV。"""

from pathlib import Path


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976):
    """產生 Premiere Pro / DaVinci Resolve 相容的 FCP 7 XML (xmeml v4)"""
    timebase = int(round(fps))

    def s2f(sec):
        return int(round(sec * fps))

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '  <project>',
        f'    <name>VideoTrimmer_{video_path.stem}</name>',
        '    <children>',
        '      <sequence id="sequence-1">',
        f'        <name>{video_path.stem}_TrimmerCut</name>',
        f'        <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
        '        <media>',
        '          <video>',
        '            <format>',
        '              <samplecharacteristics>',
        f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
        f'                <width>{width}</width>',
        f'                <height>{height}</height>',
        '                <pixelaspectratio>square</pixelaspectratio>',
        '              </samplecharacteristics>',
        '            </format>',
        '            <track>'
    ]

    timeline_cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        start_frame = timeline_cursor
        end_frame = timeline_cursor + dur_frame
        timeline_cursor = end_frame

        xml_lines.extend([
            f'              <clipitem id="clipitem-v{idx+1}">',
            f'                <name>Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}</name>',
            f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                <in>{in_frame}</in>',
            f'                <out>{out_frame}</out>',
            f'                <start>{start_frame}</start>',
            f'                <end>{end_frame}</end>',
            f'                <file id="file-1">',
            f'                  <name>{video_path.name}</name>',
            f'                  <pathurl>{video_path.resolve().as_uri()}</pathurl>',
            f'                  <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                  <duration>{s2f(total_source_dur)}</duration>',
            '                </file>',
            '              </clipitem>'
        ])

    xml_lines.extend([
        '            </track>',
        '          </video>',
        '          <audio>',
        '            <track>'
    ])

    timeline_cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        start_frame = timeline_cursor
        end_frame = timeline_cursor + dur_frame
        timeline_cursor = end_frame

        xml_lines.extend([
            f'              <clipitem id="clipitem-a{idx+1}">',
            f'                <name>Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}</name>',
            f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                <in>{in_frame}</in>',
            f'                <out>{out_frame}</out>',
            f'                <start>{start_frame}</start>',
            f'                <end>{end_frame}</end>',
            f'                <file id="file-1"/>',
            '              </clipitem>'
        ])

    xml_lines.extend([
        '            </track>',
        '          </audio>',
        '        </media>',
        '      </sequence>',
        '    </children>',
        '  </project>',
        '</xmeml>'
    ])

    Path(output_xml_path).write_text('\n'.join(xml_lines), encoding='utf-8')


def generate_fcpxml(edl, video_path, total_source_dur, total_out_dur, output_fcpxml_path, fps=23.976):
    """產生 Final Cut Pro X 相容的 FCPXML (v1.9)"""
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
        f'    <asset id="r2" name="{video_path.name}" src="{video_path.resolve().as_uri()}" start="0s" duration="{total_src_frac}" hasVideo="1" hasAudio="1" format="r1"/>',
        '  </resources>',
        '  <library>',
        f'    <event name="VideoTrimmer_{video_path.stem}">',
        f'      <project name="{video_path.stem}_TrimmerCut">',
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
            tr = c["transcript"].replace('"', '""')
            vc = c["visual_check"].replace('"', '""')
            ac = c["audio_check"].replace('"', '""')
            f.write(f'{c["clip_id"]},"{c["topic"]}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},{c["cps"]:.2f},{c["in_margin"]:.2f},{c["out_margin"]:.2f},"{tr}","{vc}","{ac}"\n')

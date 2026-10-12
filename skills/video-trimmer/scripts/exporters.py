"""輸出格式產生：FCP 7 XML、FCPXML、CSV。"""

import json
import subprocess
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

AUDIO_DEPTH = 16
AUDIO_SAMPLE_RATE = 48000


def _probe_audio_channels(video_path, default=2):
    """Return the channel count of the first audio stream (ffprobe), or `default` when unknown."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
             "-of", "json", str(video_path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30, check=False,
        )
        streams = json.loads(res.stdout or "{}").get("streams") or []
        ch = int(streams[0].get("channels", 0)) if streams else 0
        return ch if ch > 0 else default
    except Exception:  # noqa: BLE001
        return default


def format_path_for_fcp7(system_path):
    """Absolute path -> `file://localhost/...` URI as Premiere Pro writes it.

    Premiere Pro and DaVinci Resolve both read `file://localhost/<percent-encoded path>`;
    on Windows the drive colon is encoded too (`C%3A/Users/...`), which is the form Premiere
    itself exports. The empty-authority form `file:///C:/...` is not accepted by Premiere.
    """
    path_str = Path(system_path).resolve().as_posix()
    if not path_str.startswith("/"):
        path_str = "/" + path_str
    return "file://localhost" + urllib.parse.quote(path_str, safe="/")


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976,
                      audio_channels=None, markers=None):
    """產生 Premiere Pro / DaVinci Resolve 相容的 FCP 7 XML (xmeml v4)。

    結構比照 Premiere Pro 自己匯出的 FCP7 XML：sequence 有 duration 與 timecode、file 有完整
    media 描述（video/audio samplecharacteristics、channelcount）、每個聲道一條音軌並帶
    sourcetrack/trackindex，video 與 audio clipitem 以 link 互相連結，名稱做 XML 跳脫。
    markers: 選用，[{"name": str, "comment": str, "frame": int}]，寫成 sequence marker，
    Premiere 匯入後顯示在時間軸上方，剪輯師可直接跳到要確認的位置。
    """
    video_path = Path(video_path)
    timebase = int(round(fps))
    ntsc = "TRUE" if abs(fps - timebase) > 1e-3 else "FALSE"
    channels = int(audio_channels) if audio_channels else _probe_audio_channels(video_path)
    channels = max(1, min(channels, 8))

    def s2f(sec):
        return int(round(sec * fps))

    def rate_xml(indent):
        return f'{indent}<rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>'

    file_name = _xml_escape(video_path.name)
    file_url = format_path_for_fcp7(video_path)
    file_frames = s2f(total_source_dur)
    reel_name = _xml_escape(video_path.stem)

    def file_node_full(indent):
        i = indent
        return [
            f'{i}<file id="file-1">',
            f'{i}  <name>{file_name}</name>',
            f'{i}  <pathurl>{file_url}</pathurl>',
            rate_xml(i + "  "),
            f'{i}  <duration>{file_frames}</duration>',
            f'{i}  <timecode>',
            rate_xml(i + "    "),
            f'{i}    <string>00:00:00:00</string>',
            f'{i}    <frame>0</frame>',
            f'{i}    <displayformat>NDF</displayformat>',
            f'{i}    <reel><name>{reel_name}</name></reel>',
            f'{i}  </timecode>',
            f'{i}  <media>',
            f'{i}    <video>',
            f'{i}      <samplecharacteristics>',
            rate_xml(i + "        "),
            f'{i}        <width>{width}</width>',
            f'{i}        <height>{height}</height>',
            f'{i}        <anamorphic>FALSE</anamorphic>',
            f'{i}        <pixelaspectratio>square</pixelaspectratio>',
            f'{i}        <fielddominance>none</fielddominance>',
            f'{i}      </samplecharacteristics>',
            f'{i}    </video>',
            f'{i}    <audio>',
            f'{i}      <samplecharacteristics>',
            f'{i}        <depth>{AUDIO_DEPTH}</depth>',
            f'{i}        <samplerate>{AUDIO_SAMPLE_RATE}</samplerate>',
            f'{i}      </samplecharacteristics>',
            f'{i}      <channelcount>{channels}</channelcount>',
            f'{i}    </audio>',
            f'{i}  </media>',
            f'{i}</file>',
        ]

    # Timeline placement (frames), shared by video and audio clipitems
    placements = []
    cursor = 0
    for clip in edl:
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        placements.append((in_frame, in_frame + dur_frame, cursor, cursor + dur_frame))
        cursor += dur_frame
    total_frames = cursor

    def clip_name(clip):
        return _xml_escape(f'Clip_{int(clip["clip_id"]):02d}_{str(clip.get("topic", ""))[:15]}')

    def link_xml(indent, clip_index):
        """Links between the video clipitem and every audio clipitem of the same cut."""
        lines = [
            f'{indent}<link>',
            f'{indent}  <linkclipref>clipitem-v{clip_index}</linkclipref>',
            f'{indent}  <mediatype>video</mediatype>',
            f'{indent}  <trackindex>1</trackindex>',
            f'{indent}  <clipindex>{clip_index}</clipindex>',
            f'{indent}</link>',
        ]
        for ch in range(1, channels + 1):
            lines += [
                f'{indent}<link>',
                f'{indent}  <linkclipref>clipitem-a{ch}-{clip_index}</linkclipref>',
                f'{indent}  <mediatype>audio</mediatype>',
                f'{indent}  <trackindex>{ch}</trackindex>',
                f'{indent}  <clipindex>{clip_index}</clipindex>',
                f'{indent}  <groupindex>1</groupindex>',
                f'{indent}</link>',
            ]
        return lines

    seq_name = _xml_escape(f"{video_path.stem}_TrimmerCut")
    x = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '  <sequence id="sequence-1">',
        f'    <name>{seq_name}</name>',
        f'    <duration>{total_frames}</duration>',
        rate_xml("    "),
        '    <timecode>',
        rate_xml("      "),
        '      <string>00:00:00:00</string>',
        '      <frame>0</frame>',
        '      <displayformat>NDF</displayformat>',
        '    </timecode>',
        '    <in>-1</in>',
        '    <out>-1</out>',
        '    <media>',
        '      <video>',
        '        <format>',
        '          <samplecharacteristics>',
        rate_xml("            "),
        f'            <width>{width}</width>',
        f'            <height>{height}</height>',
        '            <anamorphic>FALSE</anamorphic>',
        '            <pixelaspectratio>square</pixelaspectratio>',
        '            <fielddominance>none</fielddominance>',
        '          </samplecharacteristics>',
        '        </format>',
        '        <track>',
    ]

    for idx, (clip, (in_f, out_f, start_f, end_f)) in enumerate(zip(edl, placements), start=1):
        x += [
            f'          <clipitem id="clipitem-v{idx}">',
            '            <masterclipid>masterclip-1</masterclipid>',
            f'            <name>{clip_name(clip)}</name>',
            '            <enabled>TRUE</enabled>',
            f'            <duration>{file_frames}</duration>',
            rate_xml("            "),
            f'            <start>{start_f}</start>',
            f'            <end>{end_f}</end>',
            f'            <in>{in_f}</in>',
            f'            <out>{out_f}</out>',
        ]
        x += file_node_full("            ") if idx == 1 else ['            <file id="file-1"/>']
        x += ['            <sourcetrack><mediatype>video</mediatype></sourcetrack>']
        x += link_xml("            ", idx)
        x += ['          </clipitem>']

    x += [
        '          <enabled>TRUE</enabled>',
        '          <locked>FALSE</locked>',
        '        </track>',
        '      </video>',
        '      <audio>',
        '        <format>',
        '          <samplecharacteristics>',
        f'            <depth>{AUDIO_DEPTH}</depth>',
        f'            <samplerate>{AUDIO_SAMPLE_RATE}</samplerate>',
        '          </samplecharacteristics>',
        '        </format>',
    ]

    for ch in range(1, channels + 1):
        x += ['        <track>']
        for idx, (clip, (in_f, out_f, start_f, end_f)) in enumerate(zip(edl, placements), start=1):
            x += [
                f'          <clipitem id="clipitem-a{ch}-{idx}">',
                '            <masterclipid>masterclip-1</masterclipid>',
                f'            <name>{clip_name(clip)}</name>',
                '            <enabled>TRUE</enabled>',
                f'            <duration>{file_frames}</duration>',
                rate_xml("            "),
                f'            <start>{start_f}</start>',
                f'            <end>{end_f}</end>',
                f'            <in>{in_f}</in>',
                f'            <out>{out_f}</out>',
                '            <file id="file-1"/>',
                '            <sourcetrack>',
                '              <mediatype>audio</mediatype>',
                f'              <trackindex>{ch}</trackindex>',
                '            </sourcetrack>',
            ]
            x += link_xml("            ", idx)
            x += ['          </clipitem>']
        x += [
            '          <enabled>TRUE</enabled>',
            '          <locked>FALSE</locked>',
            f'          <outputchannelindex>{ch}</outputchannelindex>',
            '        </track>',
        ]

    x += [
        '      </audio>',
        '    </media>',
    ]

    for m in (markers or []):
        frame = max(0, int(m.get("frame", 0)))
        x += [
            '    <marker>',
            f'      <name>{_xml_escape(str(m.get("name", "")))}</name>',
            f'      <comment>{_xml_escape(str(m.get("comment", "")))}</comment>',
            f'      <in>{frame}</in>',
            f'      <out>{int(m["out"]) if m.get("out") is not None else -1}</out>',
            '    </marker>',
        ]

    x += [
        '  </sequence>',
        '</xmeml>',
    ]

    Path(output_xml_path).write_text('\n'.join(x) + '\n', encoding='utf-8')


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

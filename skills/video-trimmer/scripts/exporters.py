"""輸出格式產生：FCP 7 XML、FCPXML、CSV。"""

from pathlib import Path
from xml.sax.saxutils import escape


def media_file_url(video_path):
    """
    Return a file URL for NLE media links (file://localhost/<percent-encoded path>).

    Windows paths become file://localhost/C:/... with forward slashes. The old
    f"file://localhost{path}" form put a raw backslash path after "localhost"
    on Windows, which Premiere Pro and DaVinci Resolve cannot link.
    """
    uri = Path(video_path).resolve().as_uri()  # file:///C:/... or file:///Users/...
    return "file://localhost" + uri[len("file://"):]


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976,
                      audio_channels=2, audio_sample_rate=48000):
    """
    Write an FCP 7 XML (xmeml v4) timeline for Premiere Pro and DaVinci Resolve.

    The structure follows what Premiere Pro exports: the source <file> is defined
    once with its media characteristics, later clips reference it by id, audio
    clips name their source track, and each video clip links to its audio clip.
    """
    video_path = Path(video_path)
    timebase = int(round(fps))
    ntsc = "TRUE" if abs(fps - round(fps)) > 0.01 else "FALSE"
    rate = f"<rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>"
    src_dur_frames = int(round(total_source_dur * fps))

    def s2f(sec):
        return int(round(sec * fps))

    # Timeline positions are shared by the video and audio tracks.
    placements = []
    cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = max(in_frame + 1, s2f(clip["source_out"]))
        dur = out_frame - in_frame
        placements.append((idx + 1, clip, in_frame, out_frame, cursor, cursor + dur))
        cursor += dur
    seq_dur = cursor

    full_file = [
        '                <file id="file-1">',
        f'                  <name>{escape(video_path.name)}</name>',
        f'                  <pathurl>{escape(media_file_url(video_path))}</pathurl>',
        f'                  {rate}',
        f'                  <duration>{src_dur_frames}</duration>',
        '                  <media>',
        '                    <video>',
        '                      <samplecharacteristics>',
        f'                        {rate}',
        f'                        <width>{width}</width>',
        f'                        <height>{height}</height>',
        '                        <pixelaspectratio>square</pixelaspectratio>',
        '                      </samplecharacteristics>',
        '                    </video>',
        '                    <audio>',
        '                      <samplecharacteristics>',
        '                        <depth>16</depth>',
        f'                        <samplerate>{audio_sample_rate}</samplerate>',
        '                      </samplecharacteristics>',
        f'                      <channelcount>{audio_channels}</channelcount>',
        '                    </audio>',
        '                  </media>',
        '                </file>',
    ]

    def links(n):
        return [
            '                <link>',
            f'                  <linkclipref>clipitem-v{n}</linkclipref>',
            '                  <mediatype>video</mediatype>',
            '                  <trackindex>1</trackindex>',
            f'                  <clipindex>{n}</clipindex>',
            '                </link>',
            '                <link>',
            f'                  <linkclipref>clipitem-a{n}</linkclipref>',
            '                  <mediatype>audio</mediatype>',
            '                  <trackindex>1</trackindex>',
            f'                  <clipindex>{n}</clipindex>',
            '                </link>',
        ]

    def clipitem(kind, n, clip, in_f, out_f, start_f, end_f, file_lines, extra=()):
        return [
            f'              <clipitem id="clipitem-{kind}{n}">',
            f'                <name>{escape(f"Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}")}</name>',
            '                <enabled>TRUE</enabled>',
            f'                <duration>{src_dur_frames}</duration>',
            f'                {rate}',
            f'                <start>{start_f}</start>',
            f'                <end>{end_f}</end>',
            f'                <in>{in_f}</in>',
            f'                <out>{out_f}</out>',
            *file_lines,
            *extra,
            *links(n),
            '              </clipitem>',
        ]

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '  <sequence id="sequence-1">',
        f'    <name>{escape(video_path.stem)}_TrimmerCut</name>',
        f'    <duration>{seq_dur}</duration>',
        f'    {rate}',
        '    <media>',
        '      <video>',
        '        <format>',
        '          <samplecharacteristics>',
        f'            {rate}',
        f'            <width>{width}</width>',
        f'            <height>{height}</height>',
        '            <pixelaspectratio>square</pixelaspectratio>',
        '          </samplecharacteristics>',
        '        </format>',
        '        <track>',
    ]
    for n, clip, in_f, out_f, start_f, end_f in placements:
        file_lines = full_file if n == 1 else ['                <file id="file-1"/>']
        xml_lines.extend(clipitem("v", n, clip, in_f, out_f, start_f, end_f, file_lines))
    xml_lines.extend([
        '        </track>',
        '      </video>',
        '      <audio>',
        f'        <numOutputChannels>{audio_channels}</numOutputChannels>',
        '        <format>',
        '          <samplecharacteristics>',
        '            <depth>16</depth>',
        f'            <samplerate>{audio_sample_rate}</samplerate>',
        '          </samplecharacteristics>',
        '        </format>',
        '        <track>',
    ])
    source_track = [
        '                <sourcetrack>',
        '                  <mediatype>audio</mediatype>',
        '                  <trackindex>1</trackindex>',
        '                </sourcetrack>',
    ]
    for n, clip, in_f, out_f, start_f, end_f in placements:
        xml_lines.extend(clipitem("a", n, clip, in_f, out_f, start_f, end_f,
                                  ['                <file id="file-1"/>'], source_track))
    xml_lines.extend([
        '        </track>',
        '      </audio>',
        '    </media>',
        '  </sequence>',
        '</xmeml>',
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
        f'    <asset id="r2" name="{video_path.name}" src="{media_file_url(video_path)}" start="0s" duration="{total_src_frac}" hasVideo="1" hasAudio="1" format="r1"/>',
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

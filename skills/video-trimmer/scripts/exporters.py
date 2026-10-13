"""輸出格式產生：FCP 7 XML（Premiere / DaVinci 兩種口味）、FCPXML、CSV、Markdown 剪輯明細表。"""

import json
import subprocess
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

AUDIO_DEPTH = 16
AUDIO_SAMPLE_RATE = 48000

# FCP7 XML 口味：同一個 writer、同一套結構，只在少數節點上分歧（見 generate_fcp7_xml docstring）。
XML_PROFILES = ("premiere", "davinci")
_ZERO_TC = "00:00:00:00"


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


def get_video_timecode(video_path):
    """Extract embedded camera timecode from video metadata using ffprobe (`00:00:00:00` when absent)."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags=timecode:stream_tags=timecode",
            "-of", "json",
            str(video_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(res.stdout)
        tc = data.get("format", {}).get("tags", {}).get("timecode")
        if tc:
            return tc
        for s in data.get("streams", []):
            stc = s.get("tags", {}).get("timecode")
            if stc:
                return stc
    except Exception:  # noqa: BLE001
        pass
    return _ZERO_TC


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


def _resolve_davinci_start_tc(video_path, start_tc):
    """Camera start timecode for the DaVinci profile: explicit value first, then ffprobe, then 00:00:00:00."""
    if start_tc and start_tc != _ZERO_TC:
        return start_tc
    try:
        if video_path.is_file() and video_path.stat().st_size > 0:
            detected = get_video_timecode(video_path)
            if detected and detected != _ZERO_TC:
                return detected
    except OSError:
        pass
    return start_tc or _ZERO_TC


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976,
                      audio_channels=None, markers=None, start_tc=None, xml_profile="premiere"):
    """產生 Premiere Pro / DaVinci Resolve 相容的 FCP 7 XML。

    結構比照 Premiere Pro 自己匯出的 FCP7 XML：`<xmeml>` 下直接是 `<sequence>`（不包 `<project>`）、
    sequence 有 duration 與 timecode、file 有完整 media 描述（video/audio samplecharacteristics、
    channelcount）、每個聲道一條音軌並帶 sourcetrack/trackindex，video 與 audio clipitem 以 link
    互相連結，名稱做 XML 跳脫。

    xml_profile:
      - "premiere"（預設）：xmeml v4；ID 為 `file-1` / `clipitem-v{n}` / `clipitem-a{ch}-{n}`；
        `<file><timecode>` 寫 `00:00:00:00` 並帶 `<frame>0</frame>`。`start_tc` 在此口味不使用。
      - "davinci"：同一結構，套用 DaVinci Resolve 實測修法（FORK_UPSTREAM_SPECS.md 第一節）：
        xmeml v5；ID 命名空間 `{video}_master` / `{video}_v{idx}` / `{video}_a{idx}`
        （第 2 聲道起為 `{video}_a{idx}_ch{ch}`，idx 由 0 起算）；`<file><timecode><string>`
        直通相機時間碼（`start_tc` 未給或為 00:00:00:00 時以 ffprobe 自動偵測），且**不寫** `<frame>`，
        避免 Resolve 脫機與定格。
    兩種口味的 clipitem id 與 file id 都保證全域唯一、且彼此不重疊。

    markers: 選用，[{"name": str, "comment": str, "frame": int, "out": int|None}]，寫成 sequence marker，
    匯入後顯示在時間軸上方，剪輯師可直接跳到要確認的位置。兩種口味都支援。
    """
    profile = str(xml_profile or "premiere").lower()
    if profile not in XML_PROFILES:
        raise ValueError(f"xml_profile 必須是 {XML_PROFILES} 之一，收到: {xml_profile!r}")
    is_davinci = profile == "davinci"

    video_path = Path(video_path)
    timebase = int(round(fps))
    ntsc = "TRUE" if abs(fps - timebase) > 1e-3 else "FALSE"
    channels = int(audio_channels) if audio_channels else _probe_audio_channels(video_path)
    channels = max(1, min(channels, 8))

    def s2f(sec):
        return int(round(sec * fps))

    def rate_xml(indent):
        return f'{indent}<rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>'

    def attr(value):
        return _xml_escape(str(value), {'"': "&quot;"})

    file_name = _xml_escape(video_path.name)
    file_url = format_path_for_fcp7(video_path)
    file_frames = s2f(total_source_dur)
    reel_name = _xml_escape(video_path.stem)

    # ID 命名空間（clipitem id 與 file id 互斥；索引規則依口味而定）
    if is_davinci:
        id_prefix = video_path.name
        file_id = f"{id_prefix}_master"
        master_clip_id = f"{id_prefix}_masterclip"

        def video_clip_id(n):
            return f"{id_prefix}_v{n - 1}"

        def audio_clip_id(ch, n):
            return f"{id_prefix}_a{n - 1}" if ch == 1 else f"{id_prefix}_a{n - 1}_ch{ch}"
    else:
        file_id = "file-1"
        master_clip_id = "masterclip-1"

        def video_clip_id(n):
            return f"clipitem-v{n}"

        def audio_clip_id(ch, n):
            return f"clipitem-a{ch}-{n}"

    if is_davinci:
        file_tc = _resolve_davinci_start_tc(video_path, start_tc)
        tc_display = "DF" if ";" in file_tc else "NDF"
    else:
        file_tc = _ZERO_TC

    def file_timecode_xml(i):
        if is_davinci:
            # DaVinci Resolve：<string> 帶相機時間碼時若同時有 <frame>，Resolve 會把畫面鎖在第 0 幀（定格）
            return [
                f'{i}  <timecode>',
                rate_xml(i + "    "),
                f'{i}    <string>{_xml_escape(file_tc)}</string>',
                f'{i}    <displayformat>{tc_display}</displayformat>',
                f'{i}    <reel><name>{reel_name}</name></reel>',
                f'{i}  </timecode>',
            ]
        return [
            f'{i}  <timecode>',
            rate_xml(i + "    "),
            f'{i}    <string>{_ZERO_TC}</string>',
            f'{i}    <frame>0</frame>',
            f'{i}    <displayformat>NDF</displayformat>',
            f'{i}    <reel><name>{reel_name}</name></reel>',
            f'{i}  </timecode>',
        ]

    def file_node_full(indent):
        i = indent
        return [
            f'{i}<file id="{attr(file_id)}">',
            f'{i}  <name>{file_name}</name>',
            f'{i}  <pathurl>{file_url}</pathurl>',
            rate_xml(i + "  "),
            f'{i}  <duration>{file_frames}</duration>',
            *file_timecode_xml(i),
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

    file_ref = f'<file id="{attr(file_id)}"/>'

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
            f'{indent}  <linkclipref>{_xml_escape(video_clip_id(clip_index))}</linkclipref>',
            f'{indent}  <mediatype>video</mediatype>',
            f'{indent}  <trackindex>1</trackindex>',
            f'{indent}  <clipindex>{clip_index}</clipindex>',
            f'{indent}</link>',
        ]
        for ch in range(1, channels + 1):
            lines += [
                f'{indent}<link>',
                f'{indent}  <linkclipref>{_xml_escape(audio_clip_id(ch, clip_index))}</linkclipref>',
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
        f'<xmeml version="{5 if is_davinci else 4}">',
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
            f'          <clipitem id="{attr(video_clip_id(idx))}">',
            f'            <masterclipid>{_xml_escape(master_clip_id)}</masterclipid>',
            f'            <name>{clip_name(clip)}</name>',
            '            <enabled>TRUE</enabled>',
            f'            <duration>{file_frames}</duration>',
            rate_xml("            "),
            f'            <start>{start_f}</start>',
            f'            <end>{end_f}</end>',
            f'            <in>{in_f}</in>',
            f'            <out>{out_f}</out>',
        ]
        x += file_node_full("            ") if idx == 1 else [f'            {file_ref}']
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
                f'          <clipitem id="{attr(audio_clip_id(ch, idx))}">',
                f'            <masterclipid>{_xml_escape(master_clip_id)}</masterclipid>',
                f'            <name>{clip_name(clip)}</name>',
                '            <enabled>TRUE</enabled>',
                f'            <duration>{file_frames}</duration>',
                rate_xml("            "),
                f'            <start>{start_f}</start>',
                f'            <end>{end_f}</end>',
                f'            <in>{in_f}</in>',
                f'            <out>{out_f}</out>',
                f'            {file_ref}',
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
    video_path = Path(video_path)

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
            tr = c.get("transcript", "").replace('"', '""')
            vc = c.get("visual_check", "OK").replace('"', '""')
            ac = c.get("audio_check", "OK").replace('"', '""')
            cps = c.get("cps", 0.0)
            in_m = c.get("in_margin", 0.0)
            out_m = c.get("out_margin", 0.0)
            f.write(f'{c["clip_id"]},"{c.get("topic", "")}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},{cps:.2f},{in_m:.2f},{out_m:.2f},"{tr}","{vc}","{ac}"\n')


def generate_edl_report(edl, total_source_dur, report_path, base_name=""):
    """
    輸出 Markdown 格式的 EDL 剪輯明細表（CLI 寫到 `<base>_<tag>_edl_summary.md`；
    `<base>_<tag>_edl_report.md` 是 edl_auditor 的 8 維度審計報告）。
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

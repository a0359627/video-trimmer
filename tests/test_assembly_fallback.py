"""Full-programme assembly: compact output, one minimal-format retry, degraded fallback, unique timeline_position.

Every Gemini call goes to FakeClient. No test calls the cloud.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from google.genai import types as genai_types

from scripts import assembly
from scripts.assembly import (
    DEGRADED_BANNER,
    STATUS_ASSEMBLED,
    STATUS_ASSEMBLED_AFTER_RETRY,
    STATUS_DEGRADED,
    assemble_with_fallback,
    build_assembly_prompt,
    load_chunk_results,
)
from scripts.gemini_client import merge_chunked_edl

SKILL = Path(__file__).resolve().parent.parent / "skills" / "video-trimmer"
SHOW_PROMPT = SKILL / "prompts" / "video_cut_prompt_show.md"
MODEL = "gemini-3.8-flash"


def _response(text, finish="FinishReason.STOP", prompt=30000, cand=500, thoughts=4000):
    part = SimpleNamespace(text=text, thought=False)
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=finish, content=SimpleNamespace(parts=[part]))],
        usage_metadata=SimpleNamespace(prompt_token_count=prompt, candidates_token_count=cand,
                                       thoughts_token_count=thoughts, total_token_count=prompt + cand + thoughts),
        text=text,
    )


class FakeClient:
    """Stands in for genai.Client: returns queued responses and records each request. No network."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._responses:
            raise AssertionError("unexpected extra Gemini call")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _units():
    """Sentences 1..8 at 10 s steps, 4 s long. Sentence 7 is the off-screen voice."""
    return [
        {"id": i, "start": float(i * 10), "end": float(i * 10 + 4), "text": f"句子{i}", "words": [],
         "speaker_id": "OFFSCREEN" if i == 7 else "HOST", "is_target_speaker": True}
        for i in range(1, 9)
    ]


def _clip(cid, sids, tp, role="host", segment="主體"):
    return {"clip_id": cid, "role": role, "segment": segment, "timeline_position": tp, "topic": f"t{sids[0]}",
            "sentence_ids": sids, "transcript": "／".join(f"乾淨{s}" for s in sids),
            "take_selection_reason": f"r{sids[0]}", "visual_check": f"v{sids[0]}", "audio_check": f"a{sids[0]}"}


def _chunks(chunk2_order=((5, 6), (7,), (8,))):
    """Two chunks; each chunk model numbers timeline_position from 1."""
    c2 = []
    for pos, sids in enumerate(chunk2_order, start=1):
        c2.append(_clip(pos, list(sids), pos, role="offscreen" if sids == (7,) else "host"))
    return [
        {"final_edl": [_clip(1, [1, 2], 1, segment="開場"), _clip(2, [3], 2), _clip(3, [4], 3)]},
        {"final_edl": c2},
    ]


MINIMAL_OK = json.dumps({"final_edl": [
    {"cand": "1-1", "sentence_ids": [1, 2], "role": "host", "segment": "開場"},
    {"cand": "1-3", "sentence_ids": [4], "role": "host", "segment": "主體"},
    {"cand": "2-1", "sentence_ids": [5, 6], "role": "host", "segment": "主體"},
    {"cand": "2-2", "sentence_ids": [7], "role": "offscreen", "segment": "主體"},
    {"cand": "2-3", "sentence_ids": [8], "role": "host", "segment": "結尾"},
]}, ensure_ascii=False, separators=(",", ":"))


class TestAssemblyRetryAndFallback(unittest.TestCase):
    def test_max_tokens_then_minimal_retry_succeeds(self):
        truncated = _response('{"final_edl":[{"cand":"1-1","sentence_ids":[1,', finish="FinishReason.MAX_TOKENS",
                              cand=4400, thoughts=61136)
        client = FakeClient([truncated, _response(MINIMAL_OK, cand=300, thoughts=900)])
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            res = assemble_with_fallback(client, genai_types, MODEL, [SHOW_PROMPT], _chunks(), _units(), out,
                                         "raw_footage_t", on_attempt=seen.append)
            info = res["info"]
            assert info["status"] == STATUS_ASSEMBLED_AFTER_RETRY and info["degraded"] is False
            assert [a["status"] for a in info["attempts"]] == ["truncated", "ok"]
            assert [a["format"] for a in info["attempts"]] == ["compact", "minimal"]
            # Raw text (also the truncated one), usage and meta are on disk for every attempt
            assert (out / "raw_footage_t_assembly_raw.txt").read_text(encoding="utf-8").startswith('{"final_edl"')
            assert (out / "raw_footage_t_assembly_raw_retry.txt").read_text(encoding="utf-8") == MINIMAL_OK
            meta = json.loads((out / "raw_footage_t_assembly_meta.json").read_text(encoding="utf-8"))
            assert meta["status"] == STATUS_ASSEMBLED_AFTER_RETRY
            assert meta["attempts"][0]["usage"]["thoughts_tokens"] == 61136
            assert "MAX_TOKENS" in meta["attempts"][0]["finish_reason"]
        assert [r["status"] for r in seen] == ["truncated", "ok"]
        assert res["usage"]["thoughts_tokens"] == 61136 + 900

        # Request 1: compact schema, MEDIUM thinking. Request 2: minimal schema, LOW thinking.
        cfg1, cfg2 = client.calls[0]["config"], client.calls[1]["config"]
        assert cfg1.thinking_config.thinking_level == genai_types.ThinkingLevel.MEDIUM
        assert cfg1.thinking_config.thinking_budget is None
        assert cfg2.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW
        assert cfg1.max_output_tokens == 65536 and cfg1.temperature is None
        clip1 = cfg1.response_schema.properties["final_edl"].items.properties
        clip2 = cfg2.response_schema.properties["final_edl"].items.properties
        assert set(clip1) == {"cand", "sentence_ids", "role", "segment", "transcript", "note"}
        assert set(clip2) == {"cand", "sentence_ids", "role", "segment"}
        assert "dropped" not in cfg2.response_schema.properties
        # Vertex AI answers 400 INVALID_ARGUMENT to gemini-3.8-flash requests with a large max_items
        for cfg in (cfg1, cfg2):
            assert all(p.max_items is None for p in cfg.response_schema.properties.values() if p.type == genai_types.Type.ARRAY)
        assert "重試：改用最精簡輸出" in client.calls[1]["contents"][0]
        assert "重試：改用最精簡輸出" not in client.calls[0]["contents"][0]

        edl = res["model_edl"]["final_edl"]
        assert [c["sentence_ids"] for c in edl] == [[1, 2], [4], [5, 6], [7], [8]]
        assert [c["timeline_position"] for c in edl] == [1, 2, 3, 4, 5]
        # Text, reasons and visual checks come from the chunk candidate, times from the sentences
        assert edl[0]["transcript"] == "乾淨1／乾淨2" and edl[0]["visual_check"] == "v1"
        assert edl[0]["source_in"] == 10.0 and edl[0]["source_out"] == 24.0
        assert edl[2]["chunk_index"] == 2 and edl[2]["chunk_timeline_position"] == 1
        assert edl[3]["role"] == "offscreen"
        assert res["model_edl"]["assembly_unreferenced_candidates"] == ["1-2"]

    def test_two_failures_degrade_to_chunk_results_in_source_order(self):
        client = FakeClient([
            _response('{"final_edl":[', finish="FinishReason.MAX_TOKENS", cand=5000, thoughts=60536),
            _response('{"final_edl":[{"cand"', finish="FinishReason.MAX_TOKENS", cand=1000, thoughts=64536),
        ])
        # Chunk 2 lists its candidates out of source order
        chunks = _chunks(chunk2_order=((8,), (5, 6), (7,)))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            res = assemble_with_fallback(client, genai_types, MODEL, [SHOW_PROMPT], chunks, _units(), out, "raw_footage_t")
            info = res["info"]
            assert info["status"] == STATUS_DEGRADED and info["degraded"] is True
            assert "截斷" in info["error"]
            assert (out / "raw_footage_t_assembly_raw.txt").exists()
            assert (out / "raw_footage_t_assembly_raw_retry.txt").exists()
            meta = json.loads((out / "raw_footage_t_assembly_meta.json").read_text(encoding="utf-8"))
            assert meta["degraded"] is True and [a["status"] for a in meta["attempts"]] == ["truncated", "truncated"]
        edl = res["model_edl"]["final_edl"]
        assert res["model_edl"]["assembly_status"] == STATUS_DEGRADED
        # Source-time order, and timeline_position unique across the programme
        assert [c["sentence_ids"] for c in edl] == [[1, 2], [3], [4], [5, 6], [7], [8]]
        positions = [c["timeline_position"] for c in edl]
        assert positions == list(range(1, len(edl) + 1))
        assert [c["chunk_timeline_position"] for c in edl] == [1, 2, 3, 2, 3, 1]
        assert [c["chunk_index"] for c in edl] == [1, 1, 1, 2, 2, 2]
        assert res["usage"]["candidates_tokens"] == 6000

    def test_first_attempt_succeeds_without_retry(self):
        compact = json.dumps({
            "final_edl": [
                {"cand": "1-1", "sentence_ids": [1, 2], "role": "host", "segment": "開場"},
                {"cand": "2-1", "sentence_ids": [5], "role": "host", "segment": "主體", "transcript": "只留第五句",
                 "note": "刪掉句 6"},
            ],
            "dropped": [{"cand": "1-2", "why": "重複"}],
            "notes": "ok",
        }, ensure_ascii=False)
        client = FakeClient([_response(compact)])
        with tempfile.TemporaryDirectory() as tmp:
            res = assemble_with_fallback(client, genai_types, MODEL, [SHOW_PROMPT], _chunks(), _units(), Path(tmp), "x")
        assert res["info"]["status"] == STATUS_ASSEMBLED and len(client.calls) == 1
        edl = res["model_edl"]["final_edl"]
        assert edl[1]["transcript"] == "只留第五句" and edl[1]["assembly_note"] == "刪掉句 6"
        dropped = res["model_edl"]["assembly_dropped"]
        assert dropped[0]["sentence_ids"] == [3]  # candidate reference resolved to its sentences

    def test_api_error_then_retry(self):
        client = FakeClient([RuntimeError("503 UNAVAILABLE"), _response(MINIMAL_OK)])
        with mock.patch.object(assembly, "_generate_content", lambda c, **kw: c.models.generate_content(**kw)):
            with tempfile.TemporaryDirectory() as tmp:
                res = assemble_with_fallback(client, genai_types, MODEL, [SHOW_PROMPT], _chunks(), _units(), Path(tmp), "x")
                assert (Path(tmp) / "x_assembly_raw.txt").read_text(encoding="utf-8") == ""
        assert [a["status"] for a in res["info"]["attempts"]] == ["api_error", "ok"]
        assert res["info"]["status"] == STATUS_ASSEMBLED_AFTER_RETRY

    def test_prompt_lists_candidate_refs_and_compact_output(self):
        text = build_assembly_prompt([SHOW_PROMPT], _chunks(), _units())
        assert "cand=2-3" in text and "[區段 2 / 候選 3]" in text
        assert '"take_selection_reason"' not in text.split("## 輸出")[1]
        assert "候選共 6 個" in text
        assert "## 候選沒有涵蓋的主持人長句" not in text  # every sentence is in a candidate

    def test_uncovered_host_sentences_are_listed_and_dropped_ones_are_accounted(self):
        chunks = _chunks()
        chunks[0]["final_edl"] = [c for c in chunks[0]["final_edl"] if c["sentence_ids"] != [3]]
        compact = build_assembly_prompt([SHOW_PROMPT], chunks, _units())
        minimal = build_assembly_prompt([SHOW_PROMPT], chunks, _units(), output_format="minimal")
        assert "## 候選沒有涵蓋的主持人長句（1 句）" in compact and "Sentence ID 3（30.00s->34.00s）" in compact
        assert "## 候選沒有涵蓋的主持人長句" not in minimal
        answer = {"final_edl": [{"cand": "1-1", "sentence_ids": [1, 2], "role": "host", "segment": "開場"}],
                  "dropped": [{"cand": None, "sentence_ids": [3], "why": "NG"}]}
        out = assembly.normalize_assembled_edl(answer, _units(), chunks)
        assert 3 not in [u["id"] for u in out["assembly_unaccounted"]]
        silent = assembly.normalize_assembled_edl({"final_edl": answer["final_edl"]}, _units(), chunks)
        assert 3 in [u["id"] for u in silent["assembly_unaccounted"]]


class TestTimelinePositionWithoutAssembly(unittest.TestCase):
    def test_merge_renumbers_timeline_position_across_chunks(self):
        merged = merge_chunked_edl(_chunks(), project_title="p")["final_edl"]
        assert [c["timeline_position"] for c in merged] == [1, 2, 3, 4, 5, 6]
        assert [c["chunk_timeline_position"] for c in merged] == [1, 2, 3, 1, 2, 3]
        assert [c["chunk_index"] for c in merged] == [1, 1, 1, 2, 2, 2]

    def test_load_chunk_results_from_first_chunk_file_folder_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i, ch in enumerate(_chunks(), start=1):
                (d / f"raw_footage_t_chunk{i}_raw.txt").write_text(json.dumps(ch, ensure_ascii=False), encoding="utf-8")
            (d / "._raw_footage_t_chunk1_raw.txt").write_bytes(b"\x00\x05")  # macOS AppleDouble file is ignored
            from_file, files = load_chunk_results(d / "raw_footage_t_chunk1_raw.txt")
            from_dir, _ = load_chunk_results(d)
            assert len(from_file) == 2 and len(files) == 2 and from_dir == from_file
            (d / "raw_footage_t_chunks_raw.json").write_text(json.dumps(_chunks(), ensure_ascii=False), encoding="utf-8")
            from_json, _ = load_chunk_results(d / "raw_footage_t_chunks_raw.json")
            assert [len(c["final_edl"]) for c in from_json] == [3, 3]


def _make_test_video(path: Path, seconds: int = 92) -> None:
    """Black 160x120 video; a 220 Hz tone during [10k, 10k+4) s stands in for speech."""
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=160x120:r=24:d={seconds}",
        "-f", "lavfi", "-i", f"aevalsrc=if(lt(mod(t\\,10)\\,4)\\,0.3*sin(2*PI*220*t)\\,0):s=16000:d={seconds}",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ], check=True)


def _cached_whisper_raw():
    """One Whisper segment per sentence, with word timestamps, so the pipeline uses the cache."""
    segs = []
    for i in range(1, 9):
        start = float(i * 10)
        words = [{"word": w, "start": round(start + k * 0.8, 2), "end": round(start + k * 0.8 + 0.7, 2)}
                 for k, w in enumerate(["這是", "第", f"{i}", "句", "話。"])]
        segs.append({"id": i, "start": start, "end": words[-1]["end"], "text": f"這是第{i}句話。", "words": words})
    return segs


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class TestPipelineReuseChunksOffline(unittest.TestCase):
    """Runs video_trimmer.main() end to end on a synthetic video with --reuse-chunks (no upload, fake Gemini)."""

    def _setup(self, tmp: Path):
        from scripts.transcribe import transcribe_video_whisper

        video = tmp / "raw_footage.mp4"
        _make_test_video(video)
        src = tmp / "src"
        src.mkdir()
        (src / "raw_footage_whisper_raw.json").write_text(json.dumps(_cached_whisper_raw()), encoding="utf-8")
        _, sentences = transcribe_video_whisper(video, src / "raw_footage_whisper_raw.json", model_name="small")
        ids = [s["id"] for s in sentences]
        assert len(ids) >= 6, ids
        # Each chunk model numbers timeline_position from 1
        chunks = [
            {"final_edl": [_clip(1, [ids[0]], 1), _clip(2, [ids[1]], 2), _clip(3, [ids[2]], 3)]},
            {"final_edl": [_clip(1, [ids[3]], 1), _clip(2, [ids[4]], 2), _clip(3, [ids[5]], 3)]},
        ]
        for c in chunks:
            for clip in c["final_edl"]:
                clip["transcript"] = ""  # keep the Whisper words; the test checks ordering only
        for i, ch in enumerate(chunks, start=1):
            (src / f"raw_footage_old_chunk{i}_raw.txt").write_text(json.dumps(ch, ensure_ascii=False), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        shutil.copy(src / "raw_footage_whisper_raw.json", out / "raw_footage_whisper_raw.json")
        return video, src, out

    def _main(self, argv, client=None):
        from scripts import video_trimmer

        with mock.patch.object(sys, "argv", ["video_trimmer.py", *argv]), \
             mock.patch.object(video_trimmer, "load_env_file", lambda: None), \
             mock.patch.object(video_trimmer, "get_gemini_client",
                               lambda **kw: client if client is not None else self.fail("no Gemini call expected")):
            video_trimmer.main()

    def test_without_assemble_timeline_position_is_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, src, out = self._setup(Path(tmp))
            self._main(["-i", str(video), "-o", str(out), "--suffix", "t", "--xml-only",
                        "--whisper-model", "small", "--reuse-chunks", str(src / "raw_footage_old_chunk1_raw.txt")])
            edl = json.loads((out / "raw_footage_t_edl.json").read_text(encoding="utf-8"))["final_edl"]
            positions = [c["timeline_position"] for c in edl]
            assert len(edl) >= 6
            assert positions == list(range(1, len(edl) + 1)), positions
            assert {c["chunk_index"] for c in edl} == {1, 2}
            assert sorted({c["chunk_timeline_position"] for c in edl}) == [1, 2, 3]
            assert not (out / "usage_log.jsonl").exists()  # no Gemini call at all

    def test_assemble_failure_degrades_and_marks_reports(self):
        client = FakeClient([
            _response('{"final_edl":[', finish="FinishReason.MAX_TOKENS", cand=5000, thoughts=60536),
            _response('{"final_edl":[{', finish="FinishReason.MAX_TOKENS", cand=900, thoughts=64636),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            video, src, out = self._setup(Path(tmp))
            self._main(["-i", str(video), "-o", str(out), "--suffix", "t", "--xml-only", "--whisper-model", "small",
                        "--prompt", str(SHOW_PROMPT), "--assemble", "--allow-reorder", "--reuse-chunks", str(src)],
                       client=client)
            assert len(client.calls) == 2
            doc = json.loads((out / "raw_footage_t_edl.json").read_text(encoding="utf-8"))
            edl = doc["final_edl"]
            assert doc["assembly"]["status"] == STATUS_DEGRADED
            assert doc["inference_metrics"]["assembly_status"] == STATUS_DEGRADED
            assert [c["timeline_position"] for c in edl] == list(range(1, len(edl) + 1))
            starts = [c["source_in"] for c in edl]
            assert starts == sorted(starts)  # degraded output keeps source-time order
            report = json.loads((out / "raw_footage_t_edl_report.json").read_text(encoding="utf-8"))
            assert DEGRADED_BANNER in report["agent_verdict"]["warnings"][0]
            for md in ("raw_footage_t_edl_summary.md", "raw_footage_t_edl_report.md"):
                assert DEGRADED_BANNER in (out / md).read_text(encoding="utf-8"), md
            for name in ("raw_footage_t_assembly_raw.txt", "raw_footage_t_assembly_raw_retry.txt",
                         "raw_footage_t_assembly_meta.json", "raw_footage_t_chunks_raw.json"):
                assert (out / name).exists(), name
            log = [json.loads(line) for line in (out / "usage_log.jsonl").read_text(encoding="utf-8").splitlines()]
            assert [(e["mode"], e["attempt"], e["status"]) for e in log] == [
                ("assembly", 1, "truncated"), ("assembly", 2, "truncated")]
            assert sum(e["thoughts_tokens"] for e in log) == 60536 + 64636


if __name__ == "__main__":
    unittest.main()

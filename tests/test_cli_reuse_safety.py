"""CLI safety: --prompt resolution, --reuse-chunks sentence signature, complete remediation_cmd.

No test calls the cloud: Gemini, GCS staging and the client are replaced with local fakes.
"""

import contextlib
import json
import shlex
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from scripts import video_trimmer
from scripts.assembly import (
    SIG_SOURCE_INFERENCE,
    SIG_SOURCE_LEGACY_FILE,
    SIG_SOURCE_REUSE_CHECKED,
    chunk_signature_path,
    read_chunk_signature,
    sentence_signature,
)
from scripts.edl_auditor import audit_edl_quality
from scripts.exceptions import InvalidInputError
from scripts.gemini_client import load_prompt_template

from tests.test_assembly_fallback import FakeClient, _cached_whisper_raw, _clip, _make_test_video, _response

SKILL = Path(__file__).resolve().parent.parent / "skills" / "video-trimmer"
SHOW_PROMPT = SKILL / "prompts" / "video_cut_prompt_show.md"
DEFAULT_PROMPT = SKILL / "prompts" / "video_cut_prompt.md"
MODEL = "gemini-3.8-flash"


def _no_inference(*_args, **_kwargs):
    raise AssertionError("no Gemini call expected")


def _run_main(argv, client=None, inference=None, extra_patches=()):
    """Run video_trimmer.main() with every cloud entry point replaced by a local fake."""
    patches = [
        mock.patch.object(sys, "argv", ["video_trimmer.py", *argv]),
        mock.patch.object(video_trimmer, "load_env_file", lambda: None),
        mock.patch.object(video_trimmer, "get_gemini_client", lambda **kw: client if client is not None else object()),
        mock.patch.object(video_trimmer, "stage_video_to_gcs",
                          lambda *a, **k: ("gs://offline-test/raw/raw_footage.mp4", "video/mp4", False)),
        mock.patch.object(video_trimmer, "delete_gcs_blob", lambda *a, **k: None),
        mock.patch.object(video_trimmer, "run_gemini_inference", inference or _no_inference),
        *extra_patches,
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        video_trimmer.main()


# --------------------------------------------------------------------------------------------
# --prompt: a missing file stops the run; skill-relative paths work
# --------------------------------------------------------------------------------------------
class TestPromptArgument(unittest.TestCase):
    def test_skill_relative_path_finds_show_prompt_from_any_folder(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            resolved = video_trimmer._resolve_prompt_arg("prompts/video_cut_prompt_show.md")
            assert resolved == SHOW_PROMPT.resolve()
            cands = video_trimmer._video_cut_prompt_candidates(Namespace(prompt="prompts/video_cut_prompt_show.md"))
            assert cands[0] == SHOW_PROMPT.resolve()
            assert load_prompt_template(cands) == SHOW_PROMPT.read_text(encoding="utf-8")

    def test_plugin_root_relative_path_works_outside_plugin_root(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            resolved = video_trimmer._resolve_prompt_arg("skills/video-trimmer/prompts/video_cut_prompt_show.md")
            assert resolved == SHOW_PROMPT.resolve()

    def test_working_directory_file_wins(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            own = Path(tmp) / "prompts" / "video_cut_prompt_show.md"
            own.parent.mkdir()
            own.write_text("custom prompt", encoding="utf-8")
            assert video_trimmer._resolve_prompt_arg("prompts/video_cut_prompt_show.md") == own.resolve()

    def test_missing_prompt_is_an_error_and_never_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            with self.assertRaises(InvalidInputError) as ctx:
                video_trimmer._resolve_prompt_arg("prompts/video_cut_prompt_shw.md")
            assert "video_cut_prompt_show.md" in str(ctx.exception)  # the message names the right file
            with self.assertRaises(InvalidInputError):
                video_trimmer._video_cut_prompt_candidates(Namespace(prompt="prompts/video_cut_prompt_shw.md"))
            # Without --prompt the default prompt is used as before
            assert load_prompt_template(video_trimmer._video_cut_prompt_candidates(Namespace(prompt=None))) == \
                DEFAULT_PROMPT.read_text(encoding="utf-8")

    def test_main_stops_before_any_work_on_missing_input_files(self):
        def _fail(*_a, **_k):
            raise AssertionError("must stop before probing or transcription")

        guards = (mock.patch.object(video_trimmer, "probe_video", _fail),
                  mock.patch.object(video_trimmer, "transcribe_video_whisper", _fail))
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "raw_footage.mp4"
            video.write_bytes(b"")
            for bad in (["--prompt", "prompts/video_cut_prompt_shw.md"],
                        ["--reference-script", str(Path(tmp) / "missing_script.md")],
                        ["--script", str(Path(tmp) / "missing_script.md")]):
                with self.subTest(bad=bad), self.assertRaises(SystemExit) as ctx:
                    _run_main(["-i", str(video), "-o", str(Path(tmp) / "out"), *bad], extra_patches=guards)
                assert ctx.exception.code == 1
            assert not (Path(tmp) / "out").exists()


# --------------------------------------------------------------------------------------------
# --reuse-chunks: sentence signature
# --------------------------------------------------------------------------------------------
class TestSentenceSignature(unittest.TestCase):
    UNITS = [{"id": 1, "start": 0.0, "end": 1.5, "speaker_id": "HOST", "text": "a"},
             {"id": 2, "start": 2.0, "end": 3.25, "speaker_id": "OFFSCREEN", "text": "b"}]

    def test_signature_changes_with_id_time_and_role_only(self):
        base = sentence_signature(self.UNITS)
        assert base["sentence_count"] == 2
        same = [dict(u, text="other text", words=[1, 2]) for u in self.UNITS]
        assert sentence_signature(same) == base  # text and words are not part of the identity
        for key, value in (("id", 3), ("start", 0.1), ("end", 1.6), ("speaker_id", "HOST")):
            changed = [dict(self.UNITS[0]), dict(self.UNITS[1], **{key: value})]
            assert sentence_signature(changed) != base, key
        assert sentence_signature(self.UNITS[:1]) != base

    def test_missing_speaker_id_counts_as_merge_default(self):
        plain = [{"id": 1, "start": 0.0, "end": 1.0}]
        assert sentence_signature(plain) == sentence_signature([dict(plain[0], speaker_id="SPEAKER_00")])

    def test_signature_path_is_shared_by_chunk_text_and_json(self):
        d = Path("/x")
        assert chunk_signature_path(d / "raw_footage_t_chunk1_raw.txt") == d / "raw_footage_t_chunks_sig.json"
        assert chunk_signature_path(d / "raw_footage_t_chunk12_raw.txt") == d / "raw_footage_t_chunks_sig.json"
        assert chunk_signature_path(d / "raw_footage_t_chunks_raw.json") == d / "raw_footage_t_chunks_sig.json"
        assert chunk_signature_path(d / "my_results.json") == d / "my_results_chunks_sig.json"

    def test_damaged_signature_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw_footage_t_chunks_sig.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(InvalidInputError):
                read_chunk_signature(p)
            p.write_text(json.dumps({"format": "other"}), encoding="utf-8")
            with self.assertRaises(InvalidInputError):
                read_chunk_signature(p)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class TestReuseChunksSignatureOffline(unittest.TestCase):
    """End to end on a synthetic video. The chunk results stay in the output folder (the default layout),
    where each run writes <stem>_whisper_sentences.json again during transcription."""

    def _prepare(self, tmp: Path):
        from scripts.transcribe import transcribe_video_whisper

        video = tmp / "raw_footage.mp4"
        _make_test_video(video)
        probe = tmp / "probe"
        probe.mkdir()
        (probe / "raw_footage_whisper_raw.json").write_text(json.dumps(_cached_whisper_raw()), encoding="utf-8")
        _, sentences = transcribe_video_whisper(video, probe / "raw_footage_whisper_raw.json", model_name="small")
        ids = [s["id"] for s in sentences]
        assert len(ids) >= 6, ids
        out = tmp / "out"
        out.mkdir()
        self._write_cache(out, _cached_whisper_raw())
        chunk = {"final_edl": [_clip(1, [ids[0]], 1), _clip(2, [ids[1]], 2), _clip(3, [ids[2]], 3)]}
        for clip in chunk["final_edl"]:
            clip["transcript"] = ""  # keep the Whisper words
        return video, out, ids, chunk

    @staticmethod
    def _write_cache(out: Path, segs):
        (out / "raw_footage_whisper_raw.json").write_text(json.dumps(segs), encoding="utf-8")

    @staticmethod
    def _changed_cache():
        """Another transcription of the same video: one segment fewer, so every later Sentence ID moves."""
        segs = _cached_whisper_raw()
        del segs[1]
        return segs

    def _chunk_inference(self, chunk):
        def _infer(*_a, **_k):
            return json.dumps(chunk, ensure_ascii=False), {"total_tokens": 0}, 0.01
        return _infer

    def _first_run(self, video, out, chunk):
        _run_main(["-i", str(video), "-o", str(out), "--suffix", "a", "--xml-only", "--whisper-model", "small"],
                  inference=self._chunk_inference(chunk))

    def _reuse(self, video, out, suffix, *extra, client=None, extra_patches=()):
        _run_main(["-i", str(video), "-o", str(out), "--suffix", suffix, "--xml-only", "--whisper-model", "small",
                   "--reuse-chunks", str(out / "raw_footage_a_chunk1_raw.txt"), *extra],
                  client=client, extra_patches=extra_patches)

    def test_signature_blocks_changed_transcription_in_the_same_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, out, ids, chunk = self._prepare(Path(tmp))
            self._first_run(video, out, chunk)
            sig = read_chunk_signature(out / "raw_footage_a_chunks_sig.json")
            assert (out / "raw_footage_a_chunk1_raw.txt").exists()
            assert sig["source"] == SIG_SOURCE_INFERENCE
            assert sig["sentence_count"] == len(ids)
            assert sig["transcription"]["offscreen_channel"] == "none"

            # Different transcription, chunk results in the same folder
            self._write_cache(out, self._changed_cache())
            with self.assertLogs("scripts.video_trimmer", level="ERROR") as logs, \
                    self.assertRaises(SystemExit) as ctx:
                self._reuse(video, out, "b")
            assert ctx.exception.code == 1
            assert any("不一致" in line for line in logs.output), logs.output
            assert not (out / "raw_footage_b_edl.json").exists()
            # This is the trap of the old check: the run itself rewrote the sentences file next to the chunks
            rewritten = json.loads((out / "raw_footage_whisper_sentences.json").read_text(encoding="utf-8"))
            assert len(rewritten) == len(ids) - 1

            # Same transcription again: the reuse runs
            self._write_cache(out, _cached_whisper_raw())
            self._reuse(video, out, "c")
            assert (out / "raw_footage_c_edl.json").exists()

    def test_assemble_reuse_writes_checked_signature(self):
        client = FakeClient([
            _response('{"final_edl":[', finish="FinishReason.MAX_TOKENS", cand=5000, thoughts=100),
            _response('{"final_edl":[{', finish="FinishReason.MAX_TOKENS", cand=900, thoughts=100),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            video, out, ids, chunk = self._prepare(Path(tmp))
            self._first_run(video, out, chunk)
            self._reuse(video, out, "c", "--prompt", str(SHOW_PROMPT), "--assemble", client=client)
            assert len(client.calls) == 2  # degraded assembly, still offline
            sig = read_chunk_signature(out / "raw_footage_c_chunks_sig.json")
            assert sig["source"] == SIG_SOURCE_REUSE_CHECKED
            first = read_chunk_signature(out / "raw_footage_a_chunks_sig.json")
            assert sig["sentence_sha256"] == first["sentence_sha256"]
            assert (out / "raw_footage_c_chunks_raw.json").exists()

    def test_older_results_without_signature_are_pinned_before_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, out, ids, chunk = self._prepare(Path(tmp))
            self._first_run(video, out, chunk)
            (out / "raw_footage_a_chunks_sig.json").unlink()  # results made before signatures existed
            self._write_cache(out, self._changed_cache())
            for attempt in (1, 2):  # the second attempt must not trust the rewritten sentences file
                with self.subTest(attempt=attempt), self.assertRaises(SystemExit) as ctx:
                    self._reuse(video, out, "b")
                assert ctx.exception.code == 1
            pinned = read_chunk_signature(out / "raw_footage_a_chunks_sig.json")
            assert pinned["source"] == SIG_SOURCE_LEGACY_FILE
            assert pinned["sentence_count"] == len(ids)

    def test_run_passes_complete_remediation_flags_to_the_audit(self):
        seen = {}

        def _audit(*args, **kwargs):
            seen["flags"] = kwargs.get("extra_cli_flags", "")
            return audit_edl_quality(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            video, out, ids, chunk = self._prepare(Path(tmp))
            self._first_run(video, out, chunk)
            self._reuse(video, out, "c", "--keep-pause-sec", "1.0", "--tail-pad-sec", "0.25", "--pacing", "compact",
                        extra_patches=(mock.patch.object(video_trimmer, "audit_edl_quality", _audit),))
            tokens = shlex.split(seen["flags"])
            for opt, value in (("-o", str(out.resolve())), ("--suffix", "c"), ("--keep-pause-sec", "1.0"),
                               ("--tail-pad-sec", "0.25"), ("--pacing", "compact"), ("--whisper-model", "small"),
                               ("--reuse-chunks", str((out / "raw_footage_a_chunk1_raw.txt").resolve()))):
                assert opt in tokens and tokens[tokens.index(opt) + 1] == value, (opt, tokens)
            assert "--xml-only" in tokens


# --------------------------------------------------------------------------------------------
# remediation_cmd: every option of the run, parseable by the CLI
# --------------------------------------------------------------------------------------------
def _failing_audit(video_path, script_path, flags):
    """A clip that cuts its last word fails the quality gate, so the audit writes remediation_cmd."""
    units = [{"id": 2, "start": 10.0, "end": 15.0, "text": "第一段內容", "words": [], "is_target_speaker": True}]
    clips = [{"clip_id": 1, "topic": "A", "source_in": 10.0, "source_out": 11.0, "duration": 1.0, "t_last": 14.8,
              "sentence_ids": [2], "transcript": "第一段內容", "cps": 3.0, "in_margin": 0.1, "out_margin": 0.1}]
    report = audit_edl_quality(clips, units, 60.0, video_path, script_path=script_path, extra_cli_flags=flags)
    assert report["agent_verdict"]["pass_quality_gate"] is False
    return report["agent_verdict"]["remediation_cmd"]


class TestRemediationCommand(unittest.TestCase):
    def setUp(self):
        self.parser = video_trimmer._build_arg_parser(MODEL)

    def _round_trip(self, argv, out_dir):
        args = self.parser.parse_args(argv)
        flags = video_trimmer._remediation_cli_flags(args, out_dir)
        cmd = _failing_audit(Path(args.input), Path(args.script) if args.script else None, flags)
        tokens = shlex.split(cmd)
        assert tokens[:2] == ["python3", "skills/video-trimmer/scripts/video_trimmer.py"], tokens
        return args, self.parser.parse_args(tokens[2:]), cmd

    def test_every_option_survives_the_round_trip(self):
        root = Path("/footage/EP 2 醬油")
        out = root / "out put"
        argv = [
            "-i", str(root / "raw footage.mp4"), "-o", str(out), "--suffix", "show v2", "--model", "gemini-x",
            "--project", "my-proj", "--region", "global", "--bucket", "my-bucket", "--keep-gcs-upload",
            "--crf", "20", "--no-render", "--xml-profile", "davinci", "--pacing", "breathing", "--agentic",
            "--script", str(root / "it's script.md"), "--cached-json", str(root / "cached edl.json"),
            "--skip-whisper", "--whisper-model", "large-v3", "--strict", "--verbose",
            "--prompt", str(SHOW_PROMPT), "--reference-script", str(root / "ref 腳本.md"),
            "--offscreen-channel", "auto", "--analysis-video", str(root / "proxy 1080p.mp4"), "--assemble",
            "--reuse-chunks", str(root / "x_chunks_raw.json"), "--allow-reorder", "--xml-only",
            "--repeats-as-warnings", "--keep-pause-sec", "1.0", "--tail-pad-sec", "0.25",
        ]
        args, again, _ = self._round_trip(argv, out)
        assert vars(again) == vars(args), {k: (vars(args)[k], vars(again)[k])
                                           for k in vars(args) if vars(args)[k] != vars(again)[k]}

    def test_relative_paths_become_absolute(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            out = Path(tmp) / "output"
            args, again, cmd = self._round_trip(
                ["-i", "raw.mp4", "--reference-script", "ref.md", "--reuse-chunks", "o/x_chunks_raw.json",
                 "--keep-pause-sec", "1", "--tail-pad-sec", "0.25"], out)
            assert again.output_dir == str(out.resolve())
            assert again.reference_script == str((Path(tmp) / "ref.md").resolve())
            assert again.reuse_chunks == str((Path(tmp) / "o" / "x_chunks_raw.json").resolve())
            assert (again.keep_pause_sec, again.tail_pad_sec) == (1.0, 0.25)

    def test_default_options_are_not_repeated(self):
        args = self.parser.parse_args(["-i", "/v/raw.mp4"])
        flags = video_trimmer._remediation_cli_flags(args, Path("/v/output"))
        assert shlex.split(flags) == ["-o", "/v/output", "--model", MODEL]


# --------------------------------------------------------------------------------------------
# Documentation gate: the CLI tables follow argparse
# --------------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
README_FILES = ("README.md", "README.zh-TW.md", "README.zh-CN.md", "README.ja.md", "README.ko.md")
# Doc defaults that show the effective value where argparse stores None or an environment default
EFFECTIVE_DOC_DEFAULTS = {"--input": "*(Required)*", "--output-dir": "<input_dir>/output/",
                          "--model": MODEL, "--whisper-model": "large-v3"}


class TestDocsFollowCli(unittest.TestCase):
    def setUp(self):
        self.parser = video_trimmer._build_arg_parser(MODEL)
        self.options = {}
        for action in self.parser._actions:
            longs = [o for o in action.option_strings if o.startswith("--")]
            if longs and longs[0] != "--help":
                shorts = [o for o in action.option_strings if not o.startswith("--")]
                self.options[longs[0]] = (shorts[0] if shorts else "", action.default)
        self.show_options = {
            a.option_strings[0] for g in self.parser._action_groups if (g.title or "").startswith("Show Mode")
            for a in g._group_actions
        }

    def test_skill_cli_table_lists_every_option_with_its_default(self):
        rows = {}
        for line in (SKILL / "SKILL.md").read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and cells[0].startswith("`--"):
                rows[cells[0].strip("`")] = (cells[1].strip("`"), cells[2].replace("`", ""))
        assert set(rows) == set(self.options), (set(self.options) ^ set(rows))
        for opt, (short, default) in self.options.items():
            doc_short, doc_default = rows[opt]
            assert doc_short == short, (opt, doc_short, short)
            expected = EFFECTIVE_DOC_DEFAULTS.get(opt, str(default))
            assert doc_default == expected, (opt, doc_default, expected)

    def test_every_readme_show_mode_table_lists_every_show_mode_option(self):
        assert "--keep-pause-sec" in self.show_options and "--prompt" in self.show_options
        for name in README_FILES:
            text = (REPO / name).read_text(encoding="utf-8")
            section = text[text.index("## Show Mode"):]
            listed = set()
            for line in section.splitlines():
                cell = line.strip().strip("|").split("|")[0].strip()
                if cell.startswith("`--"):
                    listed.add(cell.strip("`").split()[0])
            assert self.show_options <= listed, (name, self.show_options - listed)
            assert "--prompt prompts/video_cut_prompt_show.md" not in text, name


if __name__ == "__main__":
    unittest.main()

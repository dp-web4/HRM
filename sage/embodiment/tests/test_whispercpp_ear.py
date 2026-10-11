"""The ear without torch: whisper.cpp's server behind the same guard (sage/embodiment/whispercpp.py).

A fake `whisper-server` (a few lines of http.server) stands in for the binary, so the real client is
exercised end to end: start on a free port, the multipart request with the fitted audio context, Whisper's
confidences carried through, stop. The ear's side: SAGE_LISTEN_BACKEND=whispercpp loads it, its words go
through the REAL judge_segments, unloading stops the process, and a missing binary is "unavailable", never
silence. One test runs the real binary when it is built (skipped elsewhere).
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sage.embodiment import listening  # noqa: E402
from sage.embodiment import whispercpp as wc  # noqa: E402

FAKE = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv; port = int(a[a.index("--port") + 1]); log = a[a.index("-m") + 1] + ".log"
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = self.rfile.read(n)
        ctype = self.headers["Content-Type"]; bd = ctype.split("boundary=")[1].encode()
        fields = {}
        for part in body.split(b"--" + bd):
            if b"name=\"" in part and b"\r\n\r\n" in part:
                head, val = part.split(b"\r\n\r\n", 1)
                name = head.split(b"name=\"")[1].split(b"\"")[0].decode()
                fields[name] = len(val) if name == "file" else val.rstrip(b"\r\n").decode()
        open(log, "a").write(json.dumps(fields) + "\n")
        out = {"text": "hey sprout hello", "segments": [
            {"text": " hey sprout hello", "start": 0, "end": 1, "no_speech_prob": 0.01, "avg_logprob": -0.2,
             "tokens": [1, 2], "words": []},
            {"text": " thank you", "start": 1, "end": 2, "no_speech_prob": 0.9, "avg_logprob": -0.3}]}
        b = json.dumps(out).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture
def fake(tmp_path):
    exe = tmp_path / "whisper-server"
    exe.write_text(f"#!{sys.executable}\n" + FAKE)
    exe.chmod(0o755)
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"x")
    return exe, model


def _log(model):
    return [json.loads(line) for line in open(str(model) + ".log")]


def test_audio_context_fits_the_clip_and_never_exceeds_whisper():
    assert wc.audio_ctx(2.0) == 228
    assert wc.audio_ctx(30.0) == 1500 and wc.audio_ctx(60.0) == 1500


def test_a_model_name_is_a_file_in_the_models_dir_and_a_path_is_kept():
    assert wc.model_path("tiny.en").endswith(os.path.join("models", "ggml-tiny.en.bin")) or \
        wc.model_path("tiny.en").endswith("ggml-tiny.en.bin")
    assert wc.model_path("/x/y.bin") == "/x/y.bin"


def test_the_client_starts_asks_with_a_fitted_context_and_keeps_whispers_confidences(fake):
    exe, model = fake
    s = wc.WhisperCpp(str(model), bin_path=str(exe), full_ctx=False)
    try:
        segs = s.segments(b"\x00\x00" * 16000 * 2)          # starts on first use
        assert s.alive()
        assert [x["no_speech_prob"] for x in segs] == [0.01, 0.9]
        assert set(segs[0]) == {"text", "start", "end", "no_speech_prob", "avg_logprob"}
        sent = _log(model)[-1]
        assert sent["response_format"] == "verbose_json" and sent["audio_ctx"] == "228"
        assert sent["file"] > 64000                          # the WAV, header and 2 s of PCM
    finally:
        s.stop()
    assert not s.alive()


def test_full_context_sends_no_audio_ctx(fake):
    exe, model = fake
    s = wc.WhisperCpp(str(model), bin_path=str(exe), full_ctx=True)
    try:
        s.segments(b"\x00\x00" * 1600)
        assert "audio_ctx" not in _log(model)[-1]
    finally:
        s.stop()


def test_a_dead_server_is_restarted_on_the_next_utterance(fake):
    exe, model = fake
    s = wc.WhisperCpp(str(model), bin_path=str(exe))
    try:
        s.start()
        s.proc.kill(); s.proc.wait()
        assert s.segments(b"\x00\x00" * 1600)
    finally:
        s.stop()


def test_a_missing_binary_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="whisper-server"):
        wc.WhisperCpp(str(tmp_path / "m.bin"), bin_path=str(tmp_path / "nope")).start()


# ---- the ear --------------------------------------------------------------------------------

def _ear(monkeypatch, tmp_path, exe, model):
    monkeypatch.setattr(listening, "BACKEND", "whispercpp")
    monkeypatch.setattr(wc, "BIN", str(exe))
    t = listening.Transcriber(source="test", heard_path=str(tmp_path / "heard.jsonl"), model_name=str(model))
    return t


def test_the_ear_loads_whispercpp_and_its_guard_is_the_real_one(monkeypatch, tmp_path, fake):
    exe, model = fake
    t = _ear(monkeypatch, tmp_path, exe, model)
    try:
        t._load()
        assert t.status == "ready" and t.model.alive()
        rec = t.transcribe(b"\x00\x00" * 16000)
        # the 0.9 no-speech segment ("thank you", whisper's stock phrase) is dropped by judge_segments, kept as dropped
        assert rec["text"] == "hey sprout hello"
        assert rec["dropped"][0]["why"] == "silence" and rec["seconds"] == 1.0
    finally:
        if t.model:
            t.model.stop()


def test_unloading_the_ear_stops_the_server(monkeypatch, tmp_path, fake):
    exe, model = fake
    t = _ear(monkeypatch, tmp_path, exe, model)
    monkeypatch.setattr(listening, "UNLOAD_IDLE_S", 10.0)
    t._load()
    server = t.model
    t._last_use = time.time() - 60
    assert t.unload_if_idle() is True
    assert t.model is None and t.status == "idle" and not server.alive()


def test_a_missing_whispercpp_is_unavailable_never_silence(monkeypatch, tmp_path):
    monkeypatch.setattr(listening, "BACKEND", "whispercpp")
    monkeypatch.setattr(wc, "BIN", str(tmp_path / "absent"))
    t = listening.Transcriber(source="test", heard_path=str(tmp_path / "h.jsonl"), model_name="base.en")
    t._load()
    assert t.model is None and t.status.startswith("unavailable: FileNotFoundError")
    assert listening.ear_state(True, t.status, {"listening": True})[2] == "recognizer"


def test_the_default_backend_is_unchanged():
    assert listening.BACKEND == "torch" or os.environ.get("SAGE_LISTEN_BACKEND")


REAL_BIN = Path(wc.BIN)
REAL_MODEL = Path(wc.model_path("tiny.en"))
SAMPLE = Path(wc.HOME) / "samples" / "jfk.wav"


@pytest.mark.skipif(not (REAL_BIN.exists() and REAL_MODEL.exists() and SAMPLE.exists()),
                    reason="whisper.cpp is not built here")
def test_the_real_server_hears_words_with_confidences():
    import wave
    with wave.open(str(SAMPLE)) as w:
        pcm = w.readframes(w.getnframes())
    s = wc.WhisperCpp("tiny.en")
    try:
        rec = listening.judge_segments(s.segments(pcm), seconds=11.0, source="test")
        assert "fellow americans" in rec["text"].lower()
        assert s.peak_rss_mb() and s.peak_rss_mb() < 400
    finally:
        s.stop()


def test_a_word_split_across_segments_is_one_word_again_and_judged_whole():
    raw = [{"text": " how lumin", "start": 0, "end": 1, "no_speech_prob": 0.004, "avg_logprob": -0.03, "tokens": [1, 2, 3]},
           {"text": "ous these", "start": 1, "end": 2, "no_speech_prob": 0.0, "avg_logprob": -0.07, "tokens": [4]},
           {"text": " criticisms are", "start": 2, "end": 3, "no_speech_prob": 0.0, "avg_logprob": -0.1, "tokens": [5]}]
    segs = wc.join_split_words(raw)
    assert [s["text"] for s in segs] == [" how luminous these", " criticisms are"]
    assert segs[0]["no_speech_prob"] == 0.004 and segs[0]["end"] == 2
    assert abs(segs[0]["avg_logprob"] - (-0.03 * 3 - 0.07) / 4) < 1e-9
    assert set(segs[0]) == {"text", "start", "end", "no_speech_prob", "avg_logprob"}
    assert listening.judge_segments(segs)["text"] == "how luminous these criticisms are"


def test_a_continuation_carries_the_more_cautious_no_speech_into_the_guard():
    raw = [{"text": " hello", "no_speech_prob": 0.1, "avg_logprob": -0.2},
           {"text": "world", "no_speech_prob": 0.9, "avg_logprob": -0.2}]
    rec = listening.judge_segments(wc.join_split_words(raw))
    assert rec["text"] == "" and rec["dropped"][0]["text"] == "helloworld"

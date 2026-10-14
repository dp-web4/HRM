"""The body's audio plumbing per platform: Linux unchanged, a Mac through AVFoundation/afplay/say.

McNugget, 2026-10-14: a USB mic, a USB camera and the monitor's speakers arrive for a Mac body, and
sage/embodiment named PipeWire programs directly (pw-record, pw-play, pw-dump, espeak-ng). These pin:
  * pipewire argv is byte-identical to what Sprout ran before this module existed;
  * avfoundation captures with ffmpeg as RAW s16le (no WAV header, so audio.py skips no header),
    plays with afplay, and its fallback voice keeps the words out of argv (`say -f <file>`);
  * the inventory reads system_profiler's JSON into pw-dump's {sinks, sources} shape;
  * the backend is $SAGE_AUDIO_BACKEND when set, else avfoundation on Darwin only.
No device is touched: subprocess.run is replaced where a program would run.

Plain asserts, so a failure fails under pytest as well as the script runner.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sage.embodiment import platform_av as pav  # noqa: E402

REAL_SP = ('{"SPAudioDataType":[{"_items":['
           '{"_name":"M156F1","coreaudio_device_output":2,"coreaudio_device_transport":"coreaudio_device_type_displayport"},'
           '{"_name":"Mac mini Speakers","coreaudio_default_audio_output_device":"spaudio_yes","coreaudio_device_output":2,'
           '"coreaudio_device_transport":"coreaudio_device_type_builtin"},'
           '{"_name":"USB Mic","coreaudio_device_input":1,"coreaudio_default_audio_input_device":"spaudio_yes",'
           '"coreaudio_device_transport":"coreaudio_device_type_usb"},'
           '{"_name":"AirPods","coreaudio_device_input":1,"coreaudio_device_output":2,'
           '"coreaudio_device_transport":"coreaudio_device_type_bluetooth"}]}]}')


class Backend:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.saved = os.environ.get("SAGE_AUDIO_BACKEND")
        os.environ["SAGE_AUDIO_BACKEND"] = self.value
        return self

    def __exit__(self, *a):
        if self.saved is None:
            os.environ.pop("SAGE_AUDIO_BACKEND", None)
        else:
            os.environ["SAGE_AUDIO_BACKEND"] = self.saved


def test_pipewire_argv_is_what_sprout_ran_before():
    with Backend("pipewire"):
        argv, header = pav.capture_cmd("bluez_input.41_42_5A_A0_6B_ED.0", 16000)
        assert argv == ["pw-record", "--target", "bluez_input.41_42_5A_A0_6B_ED.0", "--rate", "16000",
                        "--channels", "1", "--format", "s16", "-"] and header is True
        assert pav.play_cmd("/x.wav", "node") == ["pw-play", "--target", "node", "/x.wav"]
        assert pav.play_cmd("/x.wav") == ["pw-play", "/x.wav"]
        assert pav.voice_tools() == ("espeak-ng", "pw-play")
        assert pav.capture_source("bluez_input.x") == "bluez_input.x"


def test_avfoundation_captures_raw_pcm_and_plays_with_afplay():
    with Backend("avfoundation"):
        argv, header = pav.capture_cmd("ignored", 16000)
        assert argv[0] == "ffmpeg" and "avfoundation" in argv and ":default" in argv
        assert argv[-3:] == ["-f", "s16le", "-"] and "16000" in argv and header is False
        assert pav.play_cmd("/x.wav", "ignored") == ["afplay", "/x.wav"]
        assert pav.voice_tools() == ("say", "afplay")
        assert pav.capture_source("ignored") == "avfoundation:default"


def test_the_mac_device_is_chosen_by_name_or_index():
    saved = os.environ.get("SAGE_AUDIO_DEVICE")
    os.environ["SAGE_AUDIO_DEVICE"] = "USB Mic"
    try:
        with Backend("avfoundation"):
            assert ":USB Mic" in pav.capture_cmd("x", 16000)[0]
    finally:
        if saved is None:
            os.environ.pop("SAGE_AUDIO_DEVICE", None)
        else:
            os.environ["SAGE_AUDIO_DEVICE"] = saved


def test_the_mac_fallback_voice_never_puts_the_words_in_argv():
    seen = []
    real = subprocess.run
    subprocess.run = lambda argv, **kw: seen.append((list(argv), open(argv[argv.index("-f") + 1]).read()))
    try:
        with Backend("avfoundation"):
            pav.synth_fallback("rm -rf / ; hello", "/tmp/out.wav", 10)
    finally:
        subprocess.run = real
    argv, text = seen[0]
    assert argv[0] == "say" and "-o" in argv and "/tmp/out.wav" in argv
    assert "rm -rf / ; hello" not in " ".join(argv) and text == "rm -rf / ; hello"


def test_the_mac_inventory_has_pw_dumps_shape():
    inv = pav.mac_audio(raw=REAL_SP)
    sinks = {s["name"]: s for s in inv["sinks"]}
    sources = {s["name"]: s for s in inv["sources"]}
    assert set(sinks) == {"M156F1", "Mac mini Speakers", "AirPods"}
    assert set(sources) == {"USB Mic", "AirPods"}
    assert sinks["Mac mini Speakers"]["default"] is True and sinks["M156F1"]["default"] is False
    assert sinks["AirPods"]["kind"] == "bluetooth" and sources["USB Mic"]["kind"] == "wired"
    assert pav.mac_audio(raw="not json") == {}


def test_the_backend_follows_the_platform_unless_set():
    saved, real = os.environ.pop("SAGE_AUDIO_BACKEND", None), sys.platform
    try:
        sys.platform = "darwin"
        assert pav.backend() == "avfoundation"
        sys.platform = "linux"
        assert pav.backend() == "pipewire"
    finally:
        sys.platform = real
        if saved is not None:
            os.environ["SAGE_AUDIO_BACKEND"] = saved


def test_audio_reads_no_header_from_a_headerless_capture():
    from sage.embodiment import audio
    with Backend("avfoundation"):
        assert audio._capture_has_header() is False
    with Backend("pipewire"):
        assert audio._capture_has_header() is True


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"{'FAILED' if fails else 'ok'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)

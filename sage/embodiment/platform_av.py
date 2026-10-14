"""The body's audio plumbing, per platform: one place that knows which program captures and plays.

Sprout's body is PipeWire on Linux: `pw-record` from the Airhug's HFP mic, `pw-play`, `espeak-ng`,
and `pw-dump` for the inventory. McNugget's body (2026-10-15: a USB mic, a USB camera and the monitor's
speakers) is a Mac: no PipeWire. The ear, the voice and the inventory keep their designs -- a capture
SUBPROCESS writing 16 kHz mono s16le to stdout and respawned if it dies, a played file, a measured
inventory -- and only the programs change:

  | act             | pipewire (Linux)               | avfoundation (macOS)                                 |
  |-----------------|--------------------------------|------------------------------------------------------|
  | capture         | pw-record --target <node>      | ffmpeg -f avfoundation -i :<device> -f s16le (no header) |
  | play a file     | pw-play [--target <node>]      | afplay (the system default output)                   |
  | fallback voice  | espeak-ng -w out.wav -- text   | say -o out.wav -f <text file> (the words never in argv) |
  | inventory       | pw-dump                        | system_profiler -json SPAudioDataType                |

Chosen like the launchd wake path (SAGE #359): $SAGE_AUDIO_BACKEND ("pipewire" | "avfoundation")
when set, otherwise avfoundation on Darwin and pipewire everywhere else. Nothing here changes what
Linux runs. ffmpeg writes RAW PCM, not WAV: its WAV header carries a LIST chunk that would put
`data` past audio.py's 64-byte header skip, so the capture reports whether a header follows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def backend() -> str:
    explicit = os.environ.get("SAGE_AUDIO_BACKEND", "").strip().lower()
    if explicit in ("pipewire", "avfoundation"):
        return explicit
    return "avfoundation" if sys.platform == "darwin" else "pipewire"


def mac_device() -> str:
    """The AVFoundation audio input: $SAGE_AUDIO_DEVICE (a name or an index), else the default input."""
    return os.environ.get("SAGE_AUDIO_DEVICE", "").strip() or "default"


def capture_cmd(source: str, rate: int) -> Tuple[List[str], bool]:
    """(argv, has_wav_header) for a process writing `rate` Hz mono s16le to stdout."""
    if backend() == "avfoundation":
        return (["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-f", "avfoundation", "-i", f":{mac_device()}",
                 "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"], False)
    return (["pw-record", "--target", source, "--rate", str(rate),
             "--channels", "1", "--format", "s16", "-"], True)


def capture_source(default_source: str) -> str:
    """The label the ear records its words under: the PipeWire node, or the AVFoundation device."""
    return f"avfoundation:{mac_device()}" if backend() == "avfoundation" else default_source


def play_cmd(path: str, target: Optional[str] = None) -> List[str]:
    """argv that plays one audio file. afplay has no device choice: it plays on the system default
    output, so on a Mac the speaker is chosen in System Settings (or the default output)."""
    if backend() == "avfoundation":
        return ["afplay", str(path)]
    return ["pw-play"] + (["--target", target] if target else []) + [str(path)]


def voice_tools() -> Tuple[str, str]:
    """(synthesizer, player) the fallback voice needs installed."""
    return ("say", "afplay") if backend() == "avfoundation" else ("espeak-ng", "pw-play")


def synth_fallback(text: str, out_wav: str, timeout: float) -> None:
    """Render `text` to a WAV with the platform's built-in voice. Raises on failure.
    The words are never a shell string; on macOS they never reach argv at all (`say -f`)."""
    if backend() == "avfoundation":
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as tf:
            tf.write(text)
            tf.flush()
            subprocess.run(["say", "-o", out_wav, "--file-format=WAVE", "--data-format=LEI16@22050",
                            "-f", tf.name], check=True, capture_output=True, timeout=timeout)
        return
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "160", "-w", out_wav, "--", text],
                   check=True, capture_output=True, timeout=timeout)


def mac_audio(timeout: float = 6.0, raw: Optional[str] = None) -> Dict:
    """Audio sinks and sources as macOS sees them, in the shape pw-dump gives: {} when unreadable.
    `raw` is a test seam (system_profiler's JSON)."""
    try:
        if raw is None:
            raw = subprocess.run(["system_profiler", "-json", "SPAudioDataType"],
                                 capture_output=True, text=True, timeout=timeout).stdout
        data = json.loads(raw)
    except Exception:
        return {}
    sinks, sources = [], []
    for group in data.get("SPAudioDataType") or []:
        for d in group.get("_items") or []:
            name = str(d.get("_name") or "?")
            transport = str(d.get("coreaudio_device_transport") or "")
            kind = "bluetooth" if "bluetooth" in transport.lower() else "wired"
            entry = {"name": name, "kind": kind, "node": name}
            if d.get("coreaudio_device_output"):
                sinks.append(dict(entry, default=d.get("coreaudio_default_audio_output_device") == "spaudio_yes"))
            if d.get("coreaudio_device_input"):
                sources.append(dict(entry, default=d.get("coreaudio_default_audio_input_device") == "spaudio_yes"))
    return {"sinks": sinks, "sources": sources}

#!/usr/bin/env python3
"""Hearing — the Airhug mic as Sprout's ear.

Continuous ambient sound level + onset detection, folded into perception (see + feel + HEAR) and
salience (a sudden sound is a surprise). Enables cross-modal binding: a sound AND a visual motion
together is more real than either alone — the second cross-check, after vision↔IMU.

Captures via `pw-record` (pipewire) from the Airhug HFP source, 16 kHz mono s16le, in fixed 0.1s
windows, computing RMS loudness. The mic is mono, so hearing gives LEVEL + ONSET, not direction.
Fails open: no mic → vision/IMU still work.
"""
from __future__ import annotations
import subprocess, threading, time, struct, math

SOURCE = "bluez_input.41_42_5A_A0_6B_ED.0"   # Airhug HFP mic (stable node name)
RATE = 16000
WIN = 1600                # 0.1s windows
FULL_SCALE = 3000.0       # RMS ~this reads as "loud" (level 1.0); ambient quiet room ~30
ONSET_JUMP = 0.08         # level rising this far above the running ambient baseline = a sound onset
ONSET_FLOOR = 0.04        # ...and only if it's above this absolute level (ignore noise-floor jitter)


class Hearing(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.level = 0.0       # normalized current loudness 0..1
        self.baseline = 0.0    # EMA ambient level
        self.onset = False     # a sudden sound above the ambient baseline this window
        self.ok = False
        self.running = True
        self.proc = None
        self._warmed = False
        try:
            self.proc = subprocess.Popen(
                ["pw-record", "--target", SOURCE, "--rate", str(RATE),
                 "--channels", "1", "--format", "s16", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            self.ok = True
        except Exception:
            pass

    def _read_exact(self, nbytes: int) -> bytes:
        buf = b""
        while self.running and len(buf) < nbytes:
            chunk = self.proc.stdout.read(nbytes - len(buf))
            if not chunk:
                return b""   # EOF → pw-record died (BT drop)
            buf += chunk
        return buf

    def run(self):
        if not self.ok:
            return
        # pw-record writes a WAV header before the PCM; skip past the 'data' chunk once.
        head = self._read_exact(64)
        if not head:
            self.ok = False; return
        i = head.find(b"data")
        leftover = head[i + 8:] if i >= 0 else head[44:]
        want = WIN * 2
        while self.running:
            need = want - len(leftover)
            data = self._read_exact(need) if need > 0 else b""
            if need > 0 and not data:
                self.ok = False; return   # mic died — fail open (vision/IMU continue)
            buf = (leftover + data)[:want]
            leftover = b""
            n = len(buf) // 2
            if n == 0:
                continue
            s = struct.unpack("<%dh" % n, buf[:n * 2])
            rms = math.sqrt(sum(x * x for x in s) / n)
            level = min(1.0, rms / FULL_SCALE)
            # first ~0.5s the HFP link warms up (level 0) — let the baseline settle before onsets
            self._warmed = self._warmed or level > 0.0
            self.onset = self._warmed and level > self.baseline + ONSET_JUMP and level > ONSET_FLOOR
            self.level = round(level, 3)
            self.baseline = 0.95 * self.baseline + 0.05 * level

    def state(self) -> dict:
        return {"level": round(self.level, 3), "onset": bool(self.onset),
                "baseline": round(self.baseline, 3), "ok": self.ok}

    def stop(self):
        self.running = False
        time.sleep(0.05)
        try:
            self.proc.terminate()
        except Exception:
            pass

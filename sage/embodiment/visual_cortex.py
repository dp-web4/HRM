#!/usr/bin/env python3
"""Sprout's visual cortex — dual IMX219 → a compact, word-shaped PERCEPTUAL STATE.

The pixels→words organ (embodied-raising plan P1). Symbolic-first: no VLM, no VAE
checkpoint. Turns each camera's frame into (motion, attention ROI, vision-trust) and
the pair into a short natural-language scene descriptor the 0.8B can reason about.

Faithful to the surveyed early-vision mechanisms:
  - capture: nvarguscamerasrc → nvvidconv → BGR → appsink (proven 30fps dual)
  - attention: Gen-2 "gravity" — 8×8 tile motion field (75th-pct + sigmoid), a fixed
    focus window that leaky-integrates toward the peak-motion tile, homing to center.
  - vision trust: compact frame-quality (sharpness + exposure + contrast).

Writes the perceptual state as JSON (atomic) at ~POLL_HZ; optional overlay display.
Run:  ~/arc-venv/bin/python -m sage.embodiment.visual_cortex --display
"""
from __future__ import annotations
import cv2, numpy as np, time, json, os, threading, argparse, tempfile
from sage.embodiment.proprioception import Proprioception
from sage.embodiment.salience import SalienceFilter

STATE_PATH = os.path.expanduser("~/.sprout/perception.json")
POLL_HZ = 4.0            # perceptual-state emit rate
GRID = 8                 # 8×8 attention tiles
FOCUS_W, FOCUS_H = 4, 3  # focus window size in tiles
GRAVITY = 0.6            # leaky-integrate approach rate
MOTION_TH = 0.12         # tile motion gate (post-sigmoid)


def gst_pipeline(sid: int, w: int = 640, h: int = 360) -> str:
    return (f"nvarguscamerasrc sensor-id={sid} ! "
            f"video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
            f"nvvidconv ! video/x-raw,width={w},height={h},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


class Camera(threading.Thread):
    """Threaded latest-frame grabber for one CSI camera."""
    def __init__(self, sid: int):
        super().__init__(daemon=True)
        self.sid = sid
        self.cap = cv2.VideoCapture(gst_pipeline(sid), cv2.CAP_GSTREAMER)
        self.frame = None
        self.ok = self.cap.isOpened()
        self.running = True

    def run(self):
        while self.running and self.ok:
            r, f = self.cap.read()
            if r and f is not None:
                self.frame = f
            else:
                time.sleep(0.005)

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def vision_trust(gray: np.ndarray) -> float:
    """Compact frame-quality trust in [0,1]: sharpness + exposure + contrast.
    (A distilled camera_trust_score; enrich with edges/noise/stereo later.)"""
    sharp = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0)     # focus energy
    contrast = min(1.0, gray.std() / 60.0)                              # dynamic range
    clip = (np.mean(gray < 8) + np.mean(gray > 247))                    # under/over-exposed frac
    exposure = max(0.0, 1.0 - clip / 0.30)                             # punish >30% clipped
    return float(np.clip(0.5 * sharp + 0.2 * contrast + 0.3 * exposure, 0, 1))


def motion_field(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """8×8 tile motion scores (0..1): 75th-pct per tile, sigmoid-amplified."""
    m = cv2.GaussianBlur(cv2.absdiff(prev_gray, gray), (5, 5), 0)
    h, w = gray.shape
    th, tw = h // GRID, w // GRID
    scores = np.zeros((GRID, GRID), np.float32)
    for ty in range(GRID):
        for tx in range(GRID):
            tile = m[ty*th:(ty+1)*th, tx*tw:(tx+1)*tw]
            s = np.percentile(tile, 75) / 255.0
            scores[ty, tx] = 1.0 / (1.0 + np.exp(-10.0 * (s - 0.1)))
    return scores


class GravityFocus:
    """Single focus window that gravitates toward the peak-motion tile."""
    def __init__(self):
        self.fx = (GRID - FOCUS_W) / 2.0
        self.fy = (GRID - FOCUS_H) / 2.0

    def update(self, scores: np.ndarray):
        peak = float(scores.max())
        if peak > MOTION_TH:
            my, mx = np.unravel_index(int(scores.argmax()), scores.shape)
            tx = np.clip(mx - FOCUS_W // 2, 0, GRID - FOCUS_W)
            ty = np.clip(my - FOCUS_H // 2, 0, GRID - FOCUS_H)
            self.fx += GRAVITY * (tx - self.fx)
            self.fy += GRAVITY * (ty - self.fy)
        else:  # home to center at rest
            self.fx += 0.1 * ((GRID - FOCUS_W) / 2.0 - self.fx)
            self.fy += 0.1 * ((GRID - FOCUS_H) / 2.0 - self.fy)
        return peak

    def roi(self) -> dict:  # normalized 0..1 rectangle
        return {"cx": round((self.fx + FOCUS_W/2) / GRID, 3),
                "cy": round((self.fy + FOCUS_H/2) / GRID, 3),
                "w": round(FOCUS_W / GRID, 3), "h": round(FOCUS_H / GRID, 3)}


class BinocularCorrelator:
    """Two-eyes-together despite MECHANICAL MISALIGNMENT (the cameras aren't a calibrated
    stereo rig — they point a little differently and may differ in focus). Matches eye-0's
    attention patch *anywhere* in eye-1 (2D, misalignment-tolerant).

    `agreement` (robust) = did both eyes confirm the same feature → it's a real thing in the
    shared field, not sensor noise. Depth is deliberately humble: we auto-calibrate the
    cameras' fixed resting offset from confident matches, then treat *deviation* from it as
    parallax — a distinct object 'standing out in depth' from the background. No metric
    distance and no near/far direction is claimed (that needs calibration we don't have)."""
    S = 0.5   # match at half-res for speed
    PS = 48   # patch size (downscaled px)

    def __init__(self):
        self.base = None  # learned fixed misalignment [dx, dy], downscaled px

    def correlate(self, gray0: np.ndarray, gray1: np.ndarray, roi0: dict) -> dict:
        g0 = cv2.resize(gray0, None, fx=self.S, fy=self.S)
        g1 = cv2.resize(gray1, None, fx=self.S, fy=self.S)
        h, w = g0.shape
        cx, cy = roi0["cx"] * w, roi0["cy"] * h
        x0 = int(np.clip(cx - self.PS//2, 0, w - self.PS)); y0 = int(np.clip(cy - self.PS//2, 0, h - self.PS))
        patch = g0[y0:y0+self.PS, x0:x0+self.PS]
        if patch.shape[0] < self.PS or patch.shape[1] < self.PS:
            return {"agreement": 0.0, "depth": "unknown", "offset": [0, 0]}
        res = cv2.matchTemplate(g1, patch, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        agreement = float(max(0.0, maxval))
        dx, dy = maxloc[0] - x0, maxloc[1] - y0
        out = {"agreement": round(agreement, 3), "offset": [int(dx/self.S), int(dy/self.S)], "depth": "unknown"}
        if agreement >= 0.5:  # confident correspondence → learn the resting misalignment
            if self.base is None:
                self.base = [float(dx), float(dy)]
            else:
                self.base = [0.9*self.base[0] + 0.1*dx, 0.9*self.base[1] + 0.1*dy]
            parallax = abs(dx - self.base[0]) + abs(dy - self.base[1])
            out["parallax"] = round(parallax / self.S, 1)
            out["depth"] = "standout" if parallax > 12 else "background"
        return out


def _dir_word(cx: float, cy: float) -> str:
    h = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    v = "upper" if cy < 0.38 else "lower" if cy > 0.62 else ""
    return (f"{v} {h}").strip() if h != "center" or v else "center"


def describe(eyes: list[dict], binoc: dict, prop: dict) -> str:
    """Deterministic symbolic scene descriptor from the two eyes, their correlation,
    and proprioception (reafference: is the motion the world's, or my own?)."""
    mot = max(e["motion"] for e in eyes)
    trust = min(e["trust"] for e in eyes)
    if mot < 0.15:
        motion_clause = "the scene is still"
    else:
        strong = max(eyes, key=lambda e: e["motion"])
        level = "strong" if mot > 0.5 else "gentle" if mot < 0.3 else "clear"
        dirw = _dir_word(strong["attention"]["cx"], strong["attention"]["cy"])
        both = eyes[0]["motion"] > 0.15 and eyes[1]["motion"] > 0.15
        confirmed = both and binoc.get("agreement", 0) > 0.45  # both eyes see a correlated feature
        if confirmed:
            standout = " and standing out from the background" if binoc.get("depth") == "standout" else ""
            motion_clause = f"{level} motion to the {dirw}, both eyes on it{standout}"
        elif both:
            motion_clause = f"{level} motion to the {dirw}, both eyes (uncorrelated)"
        else:
            motion_clause = f"{level} motion to the {dirw} (one eye only)"
    # reafference: attribute the motion to me or to the world
    sm = prop.get("self_motion", "unknown")
    if mot >= 0.15 and sm == "still":
        self_clause = "; I'm still, so this is the world moving"
    elif mot >= 0.15 and sm in ("moving", "rotating"):
        self_clause = f"; but I'm {sm}, so some of it is my own motion"
    elif sm in ("moving", "rotating"):
        self_clause = f"; I'm {sm}"
    else:
        self_clause = ""
    view = "clear view" if trust > 0.6 else "murky view" if trust > 0.3 else "almost no view (dark or blurred)"
    return f"{motion_clause}{self_clause}; {view}"


JOURNAL_PATH = os.path.expanduser("~/.sprout/perception_journal.jsonl")
JOURNAL_MAX = 2000  # keep last N events


class Journal:
    """Logs the SALIENT fraction of the stream (habituation-filtered), not every frame —
    so the raising loop reflects on signal, not noise. Each entry carries its SNARC
    breakdown. A cooldown prevents a single salient burst from flooding the log; a slow
    heartbeat marks that the senses were open even through the quiet stretches."""
    HEARTBEAT_S = 900
    COOLDOWN_S = 8

    def __init__(self):
        self.last_log = 0.0
        self._since_trim = 0

    def observe(self, s: dict, sal: dict):
        now = s["ts"]; ev = None
        if sal.get("salient") and now - self.last_log >= self.COOLDOWN_S:
            ev = {"kind": "salient", "salience": sal["salience"],
                  "snarc": {k: sal[k] for k in ("surprise", "novelty", "arousal", "conflict")}}
        elif now - self.last_log > self.HEARTBEAT_S:
            ev = {"kind": "heartbeat", "salience": sal.get("salience", 0.0)}
        if ev is not None:
            ev.update({"ts": now, "descriptor": s["descriptor"]})
            self._append(ev); self.last_log = now

    def _append(self, ev: dict):
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        with open(JOURNAL_PATH, "a") as f:
            f.write(json.dumps(ev) + "\n")
        self._since_trim += 1
        if self._since_trim >= 200:  # trim to last JOURNAL_MAX
            self._since_trim = 0
            try:
                lines = open(JOURNAL_PATH).read().splitlines()
                if len(lines) > JOURNAL_MAX:
                    with open(JOURNAL_PATH, "w") as f:
                        f.write("\n".join(lines[-JOURNAL_MAX:]) + "\n")
            except Exception:
                pass


class VisualCortex:
    def __init__(self, display: bool = False):
        self.cams = [Camera(0), Camera(1)]
        self.focus = [GravityFocus(), GravityFocus()]
        self.prev = [None, None]
        self.binoc = BinocularCorrelator()
        self.prop = Proprioception()
        self.salience = SalienceFilter()
        self.journal = Journal()
        self.display = display

    def start(self):
        for c in self.cams:
            if not c.ok:
                raise RuntimeError(f"camera {c.sid} failed to open")
            c.start()
        self.prop.start()  # inner ear (fails open if no IMU)
        time.sleep(1.0)  # warm up (auto-exposure)

    def _perceive_one(self, i: int, frame) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if self.prev[i] is not None:
            scores = motion_field(self.prev[i], gray)
            motion = float(self.focus[i].update(scores))
        self.prev[i] = gray
        return {"motion": round(motion, 3), "attention": self.focus[i].roi(),
                "trust": round(vision_trust(gray), 3)}, gray

    def _emit(self, state: dict):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE_PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)  # atomic

    def run(self):
        self.start()
        period = 1.0 / POLL_HZ
        try:
            while True:
                t0 = time.time()
                frames = [c.frame for c in self.cams]
                if any(f is None for f in frames):
                    time.sleep(0.05); continue
                perceived = [self._perceive_one(i, frames[i]) for i in range(2)]
                eyes = [p[0] for p in perceived]; grays = [p[1] for p in perceived]
                binoc = self.binoc.correlate(grays[0], grays[1], eyes[0]["attention"])
                prop = self.prop.state()
                state = {"ts": round(time.time(), 2), "cameras": {str(i): eyes[i] for i in range(2)},
                         "dominant_eye": int(np.argmax([e["motion"] for e in eyes])),
                         "binocular": binoc, "proprioception": prop,
                         "descriptor": describe(eyes, binoc, prop)}
                sal = self.salience.score(state)
                state["salience"] = sal
                self._emit(state)
                self.journal.observe(state, sal)
                if self.display:
                    self._draw(frames, eyes, state)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            pass
        finally:
            for c in self.cams:
                c.stop()
            self.prop.stop()
            if self.display:
                cv2.destroyAllWindows()

    def _draw(self, frames, eyes, state):
        panels = []
        for i, f in enumerate(frames):
            f = f.copy(); h, w = f.shape[:2]; e = eyes[i]
            r = e["attention"]
            x1 = int((r["cx"]-r["w"]/2)*w); y1 = int((r["cy"]-r["h"]/2)*h)
            x2 = int((r["cx"]+r["w"]/2)*w); y2 = int((r["cy"]+r["h"]/2)*h)
            col = (0, 200, 255) if e["motion"] > MOTION_TH else (120, 120, 120)
            cv2.rectangle(f, (x1, y1), (x2, y2), col, 2)
            cv2.putText(f, f"eye{i} mot={e['motion']:.2f} trust={e['trust']:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            panels.append(f)
        combo = np.hstack(panels)
        cv2.putText(combo, state["descriptor"], (8, combo.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Sprout visual cortex", combo)
        cv2.waitKey(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", action="store_true", help="show overlay on screen")
    args = ap.parse_args()
    VisualCortex(display=args.display).run()

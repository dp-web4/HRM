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
GAZE_PATH = os.path.expanduser("~/.sprout/gaze.json")   # the self's attention stance: {"mode": open|avert|dwell|closed, "target": [cx,cy]}
POLL_HZ = 4.0            # perceptual-state emit rate
GRID = 8                 # 8×8 attention tiles
FOCUS_W, FOCUS_H = 4, 3  # focus window size in tiles
GRAVITY = 0.6            # leaky-integrate approach rate
MOTION_TH = 0.12         # tile motion gate (post-sigmoid)
STALL_EPS = 0.3          # mean raw frame-diff below this = frame unchanged (a live sensor has noise > this)
STALL_CYCLES = 8         # this many unchanged cycles in a row (~2s at 4Hz) = the eye has stalled/gone blind


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

    def update(self, scores: np.ndarray, gaze: str = "open", target=None):
        """Attention is a CHOICE, not a reflex. Salience (peak motion) proposes; the
        gaze stance disposes. 'open' follows the pull; 'avert' looks away from it;
        'dwell' holds a chosen focus and resists the pull."""
        peak = float(scores.max())
        my, mx = np.unravel_index(int(scores.argmax()), scores.shape)
        if gaze == "dwell" and target is not None:
            tx = np.clip(target[0] * GRID - FOCUS_W / 2, 0, GRID - FOCUS_W)
            ty = np.clip(target[1] * GRID - FOCUS_H / 2, 0, GRID - FOCUS_H)
            rate = GRAVITY
        elif gaze == "avert" and peak > MOTION_TH:
            # deliberately look AWAY from the loudest pull — the mirror tile
            tx = np.clip((GRID - 1 - mx) - FOCUS_W // 2, 0, GRID - FOCUS_W)
            ty = np.clip((GRID - 1 - my) - FOCUS_H // 2, 0, GRID - FOCUS_H)
            rate = GRAVITY
        elif gaze == "open" and peak > MOTION_TH:
            tx = np.clip(mx - FOCUS_W // 2, 0, GRID - FOCUS_W)
            ty = np.clip(my - FOCUS_H // 2, 0, GRID - FOCUS_H)
            rate = GRAVITY
        else:  # no pull, or a stance with nothing to act on → ease to center
            tx = (GRID - FOCUS_W) / 2.0
            ty = (GRID - FOCUS_H) / 2.0
            rate = 0.1
        self.fx += rate * (tx - self.fx)
        self.fy += rate * (ty - self.fy)
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


def describe(eyes: list[dict], binoc: dict, prop: dict, gaze: str = "open") -> str:
    """Deterministic symbolic scene descriptor from the two eyes, their correlation,
    proprioception (reafference), and the chosen gaze stance (volition)."""
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
    stance = {"avert": "choosing to look away — ", "dwell": "holding my gaze — "}.get(gaze, "")
    stalled = [("left" if i == 0 else "right") for i, e in enumerate(eyes) if e.get("stalled")]
    if stalled:
        eyeword = " and ".join(f"{s} eye" for s in stalled)
        return f"my {eyeword} has gone dark (no fresh frames); {stance}{motion_clause}{self_clause}; {view}"
    return f"{stance}{motion_clause}{self_clause}; {view}"


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
        self._stall = [0, 0]  # consecutive unchanged-frame counts, per eye
        self.binoc = BinocularCorrelator()
        self.prop = Proprioception()
        self.salience = SalienceFilter()
        self.journal = Journal()
        self.display = display
        self._gaze_ts = 0.0
        self._gaze_mode = "open"
        self._gaze_target = None
        self._last_gaze = "open"

    def _read_gaze(self):
        """The self's chosen attention stance (re-read every ~2s). Default 'open'."""
        now = time.time()
        if now - self._gaze_ts < 2.0:
            return self._gaze_mode, self._gaze_target
        self._gaze_ts = now
        try:
            g = json.load(open(GAZE_PATH))
            self._gaze_mode = g.get("mode", "open")
            self._gaze_target = g.get("target")
        except Exception:
            self._gaze_mode, self._gaze_target = "open", None
        return self._gaze_mode, self._gaze_target

    def _note_choice(self, gaze: str):
        """A change of gaze stance is a self-authored act — worth remembering."""
        if gaze != self._last_gaze:
            phrase = {"closed": "chose to close my eyes and rest",
                      "avert": "chose to look away from the motion",
                      "dwell": "chose to hold my gaze, resisting the pull",
                      "open": "opened my eyes to the world again"}.get(gaze, f"chose gaze: {gaze}")
            self.journal._append({"kind": "choice", "salience": 1.0, "ts": round(time.time(), 2),
                                  "descriptor": phrase})
            self._last_gaze = gaze

    def start(self):
        for c in self.cams:
            if not c.ok:
                raise RuntimeError(f"camera {c.sid} failed to open")
            c.start()
        self.prop.start()  # inner ear (fails open if no IMU)
        time.sleep(1.0)  # warm up (auto-exposure)

    def _perceive_one(self, i: int, frame, gaze="open", target=None) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0; frozen = False
        if self.prev[i] is not None:
            # "frozen-looking": byte-identical frames (raw diff ~0). A live sensor always
            # carries noise above STALL_EPS. This alone does NOT mean stalled — a genuinely
            # still world looks the same. _adjudicate_sensors decides still vs stalled.
            rawdiff = float(cv2.absdiff(self.prev[i], gray).mean())
            self._stall[i] = self._stall[i] + 1 if rawdiff < STALL_EPS else 0
            frozen = self._stall[i] >= STALL_CYCLES
            scores = motion_field(self.prev[i], gray)
            motion = float(self.focus[i].update(scores, gaze, target))
        self.prev[i] = gray
        if frozen:
            motion = 0.0  # frozen frame → no real motion, whether the scene is still OR the eye is dead
        return {"motion": round(motion, 3), "attention": self.focus[i].roi(),
                "trust": round(vision_trust(gray), 3), "frozen": frozen, "stalled": False}, gray

    def _adjudicate_sensors(self, eyes: list, binoc: dict, prop: dict):
        """Still vs stalled is genuinely ambiguous without effectors to act and check for
        predicted change. Disambiguate with what Sprout does have: ego-motion (a live eye
        MUST change when the head is moved) and cross-eye agreement (a real still scene
        keeps both eyes mutually consistent). Only zero trust with corroboration; a
        frozen-but-consistent view is trusted as a genuinely still world."""
        moved = prop.get("self_motion") in ("moving", "rotating")
        agree = binoc.get("agreement", 0.0)
        for e in eyes:
            if not e.get("frozen"):
                continue
            if moved or agree < 0.4:
                # the head moved yet this eye didn't change, OR its view no longer matches the
                # other eye — evidence it is not seeing a live, shared scene.
                e["stalled"] = True
                e["trust"] = 0.0
            else:
                # rig still and the eyes still agree → most likely a genuinely still world.
                # Can't fully verify without acting, so keep trust but mark it unverified.
                e["sensor"] = "still-unverified"

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
                gaze, target = self._read_gaze()
                self._note_choice(gaze)
                if gaze == "closed":
                    # eyes shut — the self refuses the stream. The senses rest.
                    self._emit({"ts": round(time.time(), 2), "gaze": "closed",
                                "descriptor": "eyes closed — resting, not taking in the world"})
                    if self.display:
                        self._draw_closed()
                    time.sleep(period); continue
                frames = [c.frame for c in self.cams]
                if any(f is None for f in frames):
                    time.sleep(0.05); continue
                perceived = [self._perceive_one(i, frames[i], gaze, target) for i in range(2)]
                eyes = [p[0] for p in perceived]; grays = [p[1] for p in perceived]
                binoc = self.binoc.correlate(grays[0], grays[1], eyes[0]["attention"])
                prop = self.prop.state()
                self._adjudicate_sensors(eyes, binoc, prop)  # still vs stalled, using ego-motion + cross-eye
                state = {"ts": round(time.time(), 2), "cameras": {str(i): eyes[i] for i in range(2)},
                         "dominant_eye": int(np.argmax([e["motion"] for e in eyes])),
                         "binocular": binoc, "proprioception": prop, "gaze": gaze,
                         "descriptor": describe(eyes, binoc, prop, gaze)}
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
            if e.get("stalled"):
                cv2.putText(f, "STALLED - NO FRESH FRAMES", (8, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            panels.append(f)
        combo = np.hstack(panels)
        cv2.putText(combo, state["descriptor"], (8, combo.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Sprout visual cortex", combo)
        cv2.waitKey(1)

    def _draw_closed(self):
        blank = np.zeros((360, 1280, 3), np.uint8)
        cv2.putText(blank, "eyes closed - resting", (430, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 110), 2)
        cv2.imshow("Sprout visual cortex", blank)
        cv2.waitKey(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", action="store_true", help="show overlay on screen")
    args = ap.parse_args()
    VisualCortex(display=args.display).run()

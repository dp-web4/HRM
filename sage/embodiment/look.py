"""A frame on demand, from the cortex: how a being with CSI eyes sees (dp 2026-10-12: "with vision, the
goal is to have it understand directly what it's looking at").

Why not the `camera` act's ffmpeg: on Sprout the IMX219 sensors give only raw 10-bit Bayer (`RG10`) through
v4l2, the board's ffmpeg has no v4l2 input at all, and the cortex holds the sensors through Argus, which is
the only path here that yields a debayered image. So the frame comes from the organ that already has it.

Two halves, one file:
  * CLIENT (standard library only), `python3 -m sage.embodiment.look <out.jpg>`: what the seat runs for a
    `camera` act on a seat that declares `camera_source: "cortex"`. It leaves a request in the body
    directory, waits for the cortex's answer, and publishes the JPEG at <out.jpg> by temp + rename, so
    the path is old-or-new, never half-written. Exit 0 and a JSON line on success; 3 = no cortex
    answered; 4 = the cortex answered no (eyes closed, no frame yet), with the reason on stderr.
  * RESPONDER, `respond(frames, sids, gaze)`: called by the cortex on every loop. It answers at most one
    request per call, from the frame it already holds -- no extra capture, no stream, nothing kept
    beyond the last few answers.

The being still decides when to look (dp 2026-09-13: the camera is "something the being should have direct
control over"): nothing here runs without its act, and closed eyes are respected, not overridden.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import List, Optional

BODY_DIR = os.environ.get("SAGE_BODY_DIR") or os.path.expanduser("~/.sprout")
REQUEST = "look.request.json"
RESULT = "look.result.json"
OUT_DIR = "look"
WIDTH = 1024            # ~600 prompt tokens on Qwen-VL class models (heartbeat.py measured 1024 at 591)
QUALITY = 85
KEEP = 5                # answers kept in BODY_DIR/look/
WAIT_S = 8.0            # the cortex polls at POLL_HZ; an answer normally lands within ~0.2 s
STALE_S = 30.0          # a request older than this is dropped unanswered (its asker has gone)
SIDES = {0: "left", 1: "right"}


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


# ---- the cortex's half -------------------------------------------------------------------------

def respond(frames: List, sids: List[int], gaze: str = "", body_dir: Optional[str] = None,
            now: Optional[float] = None) -> Optional[dict]:
    """Answer one pending request from the frames the cortex holds; return the result written, or None."""
    body = body_dir or BODY_DIR
    req_path = os.path.join(body, REQUEST)
    if not os.path.exists(req_path):
        return None
    now = time.time() if now is None else now
    try:
        req = json.load(open(req_path))
        os.remove(req_path)
    except Exception:
        try:
            os.remove(req_path)
        except OSError:
            pass
        return None
    rid = str(req.get("id") or "")
    if not rid or now - float(req.get("ts") or 0) > STALE_S:
        return None
    res = {"id": rid, "ts": round(now, 2)}
    if gaze == "closed":
        res.update(ok=False, why="your eyes are closed (gaze closed); open them to look")
    else:
        i = next((k for k, f in enumerate(frames) if f is not None), None)
        if i is None:
            res.update(ok=False, why="no eye has delivered a frame yet")
        else:
            import cv2
            f = frames[i]
            h, w = f.shape[:2]
            if w > WIDTH:
                f = cv2.resize(f, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if not ok:
                res.update(ok=False, why="the frame could not be encoded")
            else:
                out = os.path.join(body, OUT_DIR)
                os.makedirs(out, exist_ok=True)
                path = os.path.join(out, f"{rid}.jpg")
                _atomic_write(path, buf.tobytes())
                res.update(ok=True, path=path, eye=SIDES.get(sids[i], str(sids[i])),
                           width=int(f.shape[1]), height=int(f.shape[0]), bytes=len(buf))
                old = sorted((os.path.join(out, p) for p in os.listdir(out) if p.endswith(".jpg")),
                             key=os.path.getmtime)
                for p in old[:-KEEP]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    _atomic_write(os.path.join(body, RESULT), json.dumps(res).encode())
    return res


# ---- the seat's half (what the `camera` act runs) ---------------------------------------------

def request(out_path: str, body_dir: Optional[str] = None, wait_s: float = WAIT_S) -> int:
    body = body_dir or BODY_DIR
    rid = uuid.uuid4().hex
    os.makedirs(body, exist_ok=True)
    _atomic_write(os.path.join(body, REQUEST), json.dumps({"id": rid, "ts": time.time()}).encode())
    end = time.time() + wait_s
    res = None
    while time.time() < end:
        try:
            r = json.load(open(os.path.join(body, RESULT)))
            if r.get("id") == rid:
                res = r
                break
        except Exception:
            pass
        time.sleep(0.05)
    if res is None:
        try:
            os.remove(os.path.join(body, REQUEST))
        except OSError:
            pass
        print(f"no frame: the cortex did not answer within {wait_s:g} s (is it running?)", file=sys.stderr)
        return 3
    if not res.get("ok"):
        print(f"no frame: {res.get('why')}", file=sys.stderr)
        return 4
    data = open(res["path"], "rb").read()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _atomic_write(out_path, data)
    print(json.dumps({"eye": res.get("eye"), "width": res.get("width"), "height": res.get("height"),
                      "bytes": len(data), "at": res.get("ts")}))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python3 -m sage.embodiment.look <out.jpg>", file=sys.stderr)
        return 2
    return request(argv[0])


if __name__ == "__main__":
    sys.exit(main())

# Sprout Embodiment — the perceptual organ

A **pixels→words perceptual organ** for SAGE-Sprout (Jetson Orin Nano, dual IMX219 CSI
cameras + Yahboom CMP10A IMU). It turns the raw sensor stream into a compact, continuously-
fresh *symbolic* perceptual state — deterministically, with **no VLM and no learned
checkpoint** — so a text-only 0.8B model can be *raised on what it actually senses*.

This is the Sensation-era substrate: senses → words → the being. Vision proper (depth,
recognition, learned perception) is deliberately **not** attempted here.

Built 2026-07-07/08. Companion narrative: `private-context/moments/2026-07-08-sprout-embodiment-and-the-eye-that-chose-rest.md`. Original plan: `shared-context/plans/sprout-embodied-vision-plan-2026-07-07.md`.

---

## Modules

| File | What it is |
|------|-----------|
| `visual_cortex.py` | The cortex: capture → motion/attention → binocular → sensor-health → salience → descriptor. Emits the perceptual state; runs the loop. |
| `proprioception.py` | Vendored lean reader for the Yahboom CMP10A IMU (11-byte `0x55` packets, 9600 baud). accel/gyro/angle → self-motion (`still`/`moving`/`rotating`) + orientation. Fails open (no IMU → vision still works). |
| `salience.py` | SNARC-lite salience filter (Surprise/Novelty/Arousal/Conflict + habituation). Extracts the *salient fraction* of the high-bandwidth stream. |

## The perceptual pipeline (per ~4 Hz cycle)

1. **Capture** — `nvarguscamerasrc` (CSI/Argus, GPU ISP) → 640×360 BGR → threaded latest-frame grabber, one per camera.
2. **Motion + attention** — 8×8 tile motion field (abs-diff, 75th-pct, sigmoid); a `GravityFocus` window leaky-integrates toward the peak-motion tile.
3. **Volition** — the *gaze stance* (see `gaze.json`) overrides the reflex: `open` follows the pull, `avert` looks away from it, `dwell` holds a chosen focus, `closed` shuts the eyes and rests. **Salience proposes; the self disposes.**
4. **Per-eye trust** — `vision_trust(gray)` from sharpness/contrast/exposure.
5. **Binocular** — `BinocularCorrelator` matches eye-0's attention patch *anywhere* in eye-1 (misalignment-tolerant — the cameras are **not** a calibrated rig). `agreement` is the robust signal; depth is a humble "standout vs background" from auto-calibrated parallax.
6. **Proprioception + reafference** — IMU self-motion; the descriptor attributes motion to *world* vs *self* ("I'm still, so this is the world moving").
7. **Sensor health** — stall/liveness adjudication (see below).
8. **Salience** — SNARC-lite score; gates what enters the journal.
9. **Descriptor** — one deterministic natural-language sentence integrating all of it.

## Sensor health: still vs stalled (important design)

A frozen camera returns byte-identical frames. A *genuinely still world* looks the same to a
live camera. **Without effectors to act and check for predicted change, these are not cleanly
separable.** So:

- `_perceive_one` only flags an eye **frozen-looking** (raw mean frame-diff `< STALL_EPS` for `STALL_CYCLES`). A live sensor always carries noise above the threshold.
- `_adjudicate_sensors` then decides *stalled* vs *still* using the two partial effector-substitutes Sprout has:
  - **ego-motion** — if the IMU says the head moved but the eye didn't change, a live eye couldn't do that → **stalled** (trust 0);
  - **cross-eye agreement** — a frozen view that no longer matches the other eye → **stalled** (trust 0).
- Otherwise (rig still, eyes agree) it can't confirm death → **`stale-unverified`**: trust is **not** zeroed, but it **decays** from image-quality toward a low floor with staleness (a stale frame carries no info about the *present*, so trust reflects *liveness × quality*, not sharpness alone). A trust collapse is itself a salient event.

## Interfaces (files under `~/.sprout/`)

**`perception.json`** — the live state, emitted atomically each cycle:
```jsonc
{ "ts": 1720.., "cameras": { "0": {"motion":0.36,"attention":{"cx","cy","w","h"},
      "trust":0.98,"frozen":false,"stalled":false,"sensor":"live"}, "1": {…} },
  "dominant_eye": 0, "binocular": {"agreement":0.9,"offset":[dx,dy],"depth":"background"},
  "proprioception": {"self_motion":"still","roll","pitch","yaw","accel_mag","gyro_mag","ok":true},
  "gaze": "dwell", "salience": {"salience","salient","surprise","novelty","arousal","conflict"},
  "descriptor": "holding my gaze — clear motion to the center …; clear view" }
```
When `gaze == "closed"` the state is minimal: `{ts, gaze:"closed", descriptor:"eyes closed — resting…"}` (no `cameras` key — consumers must handle this).

**`perception_journal.jsonl`** — append-only log of the **salient fraction only** (transitions + gaze *choices* + a slow heartbeat), each entry `{kind, ts, descriptor, salience, [snarc]}`. Trimmed to the last 2000. This is the "since last session" digest source.

**`gaze.json`** — the self's attention stance, read every ~2 s and honored by the cortex:
```json
{ "mode": "open|avert|dwell|closed", "target": [cx,cy]?, "chosen_by": "sprout", "session": 421, "words": "…" }
```
The raising loop writes this when Sprout chooses its gaze (§ below); it can also be set by hand for testing.

## Wiring into the raising (`sage/raising/scripts/ollama_raising_session.py`)

- **Grounding (P2)** — `_load_perceptual_digest()` builds a "what stood out since we last spoke" digest from the journal (`_summarize_perception`) + the current descriptor, placed in the MRH `SensorsBlock` and **appended to the system prompt** (note: `compose()` routes Sensors to the *user* turn, which the builder discards — hence the explicit append). Framed as **ambient** ("you don't need to account for it"). **Fails open**: no live cortex → empty block → raising unchanged.
- **Gaze agency** — `_offer_gaze_choice()` at session close (later phases only, and only if a live cortex exists) offers Sprout a genuine choice of how to hold its attention; its answer is charitably parsed to a mode and written to `gaze.json`. The parse is lossy — it defaults to `open` and records the verbatim `words`.

## Running it

From the repo root (`~/ai-workspace/sage`). The IMU port (`/dev/ttyUSB0`) needs the `dialout`
group; `dp` was added to it, but until a fresh login it's applied via `sg`:

```bash
sg dialout -c 'DISPLAY=:1 XAUTHORITY=/run/user/2002/gdm/Xauthority XDG_RUNTIME_DIR=/run/user/2002 \
  nohup /home/dp/arc-venv/bin/python -m sage.embodiment.visual_cortex --display \
  >/tmp/cortex.log 2>&1 & echo $! > /tmp/cortex.pid'
```

- `--display` shows the annotated dual feed (box color: grey=still, yellow=motion, orange=stale, red=stalled). Omit for headless.
- **Kill by explicit PID** (`kill $(cat /tmp/cortex.pid)`). Do **not** `pkill -f visual_cortex` — it self-matches the launching shell (exit 144).
- On camera contention or an Argus stall: `sudo systemctl restart nvargus-daemon` before relaunch.
- **Not yet a systemd service** — a manual `nohup` survives logout but **not reboot**. Making it a service is a small, still-pending step (would also retire the `sg` workaround once `dialout` is native).

## Known issues / honest edges

- **eye0 Argus stall (software, not hardware).** Camera 0 intermittently halts mid-stream with `nvarguscamerasrc0 reported: INVALID_SETTINGS`. Kernel shows **zero CSI PHY/lane errors** → the physical link is healthy; it's a stuck Argus/userspace stream state that a reinit clears. Finer localization (cortex session vs. nvargus-daemon) and a **software auto-recovery** (reinit sensor-0's stream on detected stall) are pending. Do **not** assume "always sensor 0" ⇒ hardware.
- **`motion_field` sigmoid floor.** Zero frame-diff maps to ~0.269 (`1/(1+e¹)`), above `MOTION_TH`. Left as-is to preserve validated motion calibration; consequence: a *live but static* scene rarely reads true "still" after startup, and stall detection (raw-diff based) is what actually distinguishes frozen from moving.
- **Gaze parse is lossy.** A 0.8B's oblique answer can be misread (e.g. S423 "keep my head up and ready" → parsed `closed`). Verbatim words are always recorded alongside the mode.

## Key tunables (`visual_cortex.py`)
`POLL_HZ=4`, `GRID=8`, `FOCUS_W/H=4/3`, `GRAVITY=0.6`, `MOTION_TH=0.12`, `STALL_EPS=0.3`, `STALL_CYCLES=8`.

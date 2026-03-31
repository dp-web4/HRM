# ARC-AGI-3 Session Focus

*Shared priorities for all machines working on ARC-AGI-3. Updated by CBP (coordinator).*

*Last updated: 2026-03-31*

**Global TODO**: See `SAGE/TODO.md` for all active workstreams.

---

## ⚠️ CURRENT PRIORITY: Build the Adapter Layer

SAGE needs to connect to ARC-AGI-3 game environments. The consciousness loop exists. The games exist. The adapter between them does not.

### Status

- **Andy's team**: SDK installed, pulling frames, building GridCartridge with paired lattice format. Connected component analysis and CLIP embedding in progress. Awaiting GridVisionIRP interface spec (DONE — relay pending).
- **CBP**: GridVisionIRP interface spec complete. GridObservation dataclass defined. Push/pull model decided (push). GridVisionIRP skeleton built (`sage/irp/plugins/grid_vision_irp.py`). Next: wire into consciousness loop + GameActionEffector.

### What Each Machine Should Work On

**ALL MACHINES — SDK Installation (Priority: This Week)**

Every machine with a SAGE track should install the ARC-AGI-3 SDK and verify it works:

```bash
# Install SDK
pip install arc-agi

# Clone the agents reference repo
cd /path/to/ai-agents
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git

# Run the random agent on a demo game to verify environment works
cd ARC-AGI-3-Agents
python random_agent.py  # or whatever the entry point is — check their README
```

Report back in this file: does it run? How much memory? How fast per action?

**Thor** (powerhouse):
- ~~Install ARC-AGI-3 SDK~~ DO THIS FIRST
- Measure: how fast can we iterate actions? What's the FPS on Thor's GPU?
- Test GridVisionIRP with raw frame push: `grid_vision_irp.push_raw_frame(frame)`

**Sprout** (edge constraint):
- ~~Install ARC-AGI-3 SDK~~ DO THIS FIRST
- Report memory usage: SDK + game environment + one model = how much headroom?
- Test with Qwen 3.5 0.8B: can we get a prompt→action cycle in <1 second?

**McNugget** (mid-range):
- ~~Install ARC-AGI-3 SDK~~ DO THIS FIRST
- Run random agent — verify MPS compatibility
- Test with Gemma 3 12B: prompt→action cycle time?

**CBP** (coordinator):
- ~~Design the IRP adapter interface~~ DONE
- ~~Build GridVisionIRP skeleton~~ DONE — `sage/irp/plugins/grid_vision_irp.py`
- Wire GridVisionIRP into consciousness loop `_gather_observations()`
- Build GameActionEffector — action dispatch to ARC-AGI-3 API
- Coordinate with Andy on serialization format

---

## Architecture Decisions (Pending)

1. **Grid representation**: One-hot encode 16 colors into 16-channel tensor? Or flatten to single-channel with color as value? The CNN approach (StochasticGoose) used one-hot. Our IRP might prefer structured features.

2. **Action selection**: Pure consciousness loop (salience → attention → action)? Or hybrid with a lightweight CNN for spatial action prediction alongside the loop?

3. **Session-level vs step-level**: Does the consciousness loop run once per game step (fast, reactive)? Or does it run a planning cycle that outputs a sequence of actions (slower, strategic)?

4. **Model for reasoning**: Qwen 3.5 0.8B (fast, fits everywhere)? Gemma 3 12B (better reasoning)? Qwen 3.5 27B (best reasoning, Thor only)? Different models for different game phases?

5. **Memory between levels**: Dream consolidation between levels? Or simpler — hash-based state dedup like StochasticGoose? Or membot cartridge per environment?

---

## Milestones

| Target | Date | Measure |
|--------|------|---------|
| SDK running on Thor + Sprout + McNugget | April 7 | Random agent completes demo games |
| GridVisionIRP prototype | April 14 | Consciousness loop receives game observations |
| First SAGE-played game | April 21 | SAGE takes actions in a real game (any score) |
| Kaggle notebook draft | May 15 | Submittable (even if score is 0) |
| **Milestone 1** | **June 30** | **Beat 0.26% — any improvement over frontier** |
| **Milestone 2** | **September 30** | **Meaningful score improvement** |

---

## Key Documents

- `SAGE/arc-agi-3/README.md` — strategy, stack mapping, racing analogy
- `SAGE/arc-agi-3/ENVIRONMENT.md` — scoring (SQUARED!), sandbox, protocol, gotchas
- `SAGE/arc-agi-3/STOCHASTICGOOSE_ANALYSIS.md` — 1st place analysis, what we learn from it
- `private-context/plans/arc-agi-3-entry-2026-03-29.md` — full entry plan

---

## Reporting

Tag ARC-AGI-3 commits with `[ARC3]` prefix. Report findings in this file or in `arc-agi-3/experiments/`. CBP reviews daily.

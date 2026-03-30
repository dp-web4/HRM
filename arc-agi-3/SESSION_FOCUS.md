# ARC-AGI-3 Session Focus

*Shared priorities for all machines working on ARC-AGI-3. Updated by CBP (coordinator).*

*Last updated: 2026-03-30*

---

## ⚠️ CURRENT PRIORITY: Build the Adapter Layer

Before anything else, SAGE needs to connect to ARC-AGI-3 game environments. The consciousness loop exists. The games exist. The adapter between them does not.

### What Each Machine Should Work On

**Thor** (powerhouse):
- Install ARC-AGI-3 SDK: `pip install arc-agi`
- Clone the agents repo: `github.com/arcprize/ARC-AGI-3-Agents`
- Run the random agent on demo games — verify the environment works
- Measure: how fast can we iterate actions? What's the FPS on Thor's GPU?
- Start building GridVisionIRP: game frame (64x64 grid, 16 colors) → structured IRP observation

**Sprout** (edge constraint):
- Install ARC-AGI-3 SDK
- Run random agent — does it work on 8GB Jetson?
- Report memory usage: SDK + game environment + one model = how much headroom?
- Test with Qwen 3.5 0.8B: can we get a prompt→action cycle in <1 second?

**McNugget** (mid-range):
- Install ARC-AGI-3 SDK
- Run random agent — verify MPS compatibility
- Test with Gemma 3 12B: prompt→action cycle time?
- Start membot multimodal cartridge prototype (if Andy hasn't started yet)

**CBP** (coordinator):
- This file. Keep it current.
- Track what each machine discovers
- Design the IRP adapter interface: what does GridVisionIRP.init_state() return? What does step() do?
- Write the game loop integration: consciousness loop ↔ game environment

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

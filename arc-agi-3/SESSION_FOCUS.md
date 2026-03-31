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

**Thor** (powerhouse): ✅ SDK VERIFIED + BENCHMARKED 2026-03-31
- ~~Install ARC-AGI-3 SDK~~ DONE — arc-agi 0.9.6 + arcengine 0.9.3 in .venv-arc
- ~~Test GridVisionIRP + GameActionEffector~~ DONE — both functional, 100% effector efficiency
- SDK performance: reset 0.002s, step 0.000s (essentially instant)
- **Qwen 3.5 27B reasoning: ~24s/action** (17.8s, 25.9s, 28.0s measured)
  - Model size: 26.9B parameters (19GB), Q5_K_M quantization
  - Hypothesis formation working, action-outcome memory tracking
  - **TOO SLOW for competition** (need <1s/action, ideally <100ms)
- Next: Test with smaller models (Gemma 3 12B or Qwen 3.5 0.8B) for speed vs reasoning tradeoff
- Next: Prompt optimization to reduce reasoning time (currently verbose)

**Sprout** (edge constraint): ✅ SDK VERIFIED + FIRST SCORE 2026-03-31
- ~~Install ARC-AGI-3 SDK~~ DONE — arc-agi 0.9.6 + arcengine 0.9.3 (Python 3.10 patches: `typing_extensions.Self`, `--ignore-requires-python`)
- SDK performance: ~3,000 steps/sec, 38MB RSS — game engine is NOT the bottleneck
- **Qwen 3.5 0.8B LLM game runner: ~5.6s/action** (ultra-compact prompts, `think: false` critical)
  - `/api/chat` with `think: False` at top level (NOT in `options`) — without this, empty responses
  - Pre-seeded JSON format reduces parsing overhead
  - LLM runner: 0 levels (too slow for tight step budgets, random action selection)
- **Smart clicker (no LLM): lp85 Level 1/8 at step 4** — FIRST SPROUT SCORE
  - Two-phase: EXPLORE (probe colors) → EXPLOIT (target effective colors)
  - Color 8 = 100% effectiveness on lp85 (consistent across runs)
  - 24 other games: 0 levels (tight step budgets, color patterns vary per API run)
- Memory: SDK 38MB + Ollama 2.3GB + system = ~5.3GB total (within 8GB budget)
- Next: Improve scoring beyond lp85, test if 0.8B can reason about grid structure pre-click

**McNugget** (mid-range): ✅ SDK + GAME RUNNER OPERATIONAL 2026-03-31
- ~~Install ARC-AGI-3 SDK~~ DONE — arc-agi 0.9.6 + arcengine 0.9.3
- ~~Run random agent~~ DONE — local execution works, anonymous API key auto-generated
- ~~GridVisionIRP~~ DONE — frame diff, change detection, multi-frame SDK handling
- ~~GameActionEffector~~ DONE — action dispatch, efficiency tracking
- ~~Full game runner~~ DONE — v2 with sequence planning + reflection cycles
- **Performance (fast mode): 1.9s/step, 4.5s/sequence** (Gemma 3 12B via Ollama)
- **Performance (verbose): 16s/step** (full reasoning with grid crops)
- SDK: 5,910 steps/sec (0.2ms/step), 101 MB RSS — game engine is NOT the bottleneck
- Tested on: tu93, dc22, sc25, wa30, ls20
- **Key finding**: LLM discovers real game rules through pure exploration
  - sc25: Found UP→UP→UP→ACTION6 changes color 14→0 in specific columns
  - wa30: Identified SELECT as "primary driver of change"
  - ls20: Discovered UP changes colors (11→3), column range depends on position
- **Gap**: 0 levels completed across all games. Needs goal detection.
- Next: lighter model test (can 0.8B or 4B complete levels faster?)

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

4. **Model for reasoning**: ⚠️ **SPEED IS CRITICAL** — competition requires <1s/action, ideally <100ms
   - **Qwen 3.5 27B**: Best reasoning (hypothesis formation, memory tracking) but ~24s/action = impractical
   - **Gemma 3 12B**: Middle ground — 1.9s/step fast mode (McNugget), 3.8s baseline
   - **Phi-4 14B**: Best reasoning/GB — downloading to Thor, see `MODEL_EVALUATION_PLAN.md`
   - **Qwen 3.5 0.8B**: ~~test needed~~ MEASURED — **5.6s/action** on Sprout (Jetson, ~20 tok/s). NOT sub-second. `think:false` mandatory.
   - **Key insight**: No-LLM clicker outperforms LLM runner on all games tested (faster + targeted)
   - **Hybrid approach**: Different models for different phases? Fast model for actions, slow model for strategy?
   - **Prompt optimization**: Current prompts verbose — can we reduce reasoning time 3-5x with tighter prompts?

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
